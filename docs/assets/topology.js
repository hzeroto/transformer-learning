(() => {
  const root = typeof window === 'undefined' ? globalThis : window;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const geometry = {
    nodeWidth: 154,
    nodeHeight: 50,
    rankGap: 186,
    nodeGap: 10,
    laneGap: 12,
    laneHeaderHeight: 43,
    lanePaddingBottom: 14,
    graphPaddingLeft: 176,
    graphPaddingRight: 58,
    graphPaddingTop: 54,
    graphPaddingBottom: 38
  };

  function createSvgElement(name, attributes = {}, text = '') {
    const element = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) {
      element.setAttribute(key, String(value));
    }
    if (text) element.textContent = text;
    return element;
  }

  function calculateRanks(nodes) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const ranks = new Map();
    const visiting = new Set();

    function visit(node) {
      if (ranks.has(node.id)) return ranks.get(node.id);
      if (visiting.has(node.id)) throw new Error(`课程依赖中存在环：${node.id}`);
      visiting.add(node.id);
      let rank = 0;
      for (const prerequisiteId of node.prerequisites) {
        const prerequisite = byId.get(prerequisiteId);
        if (!prerequisite) throw new Error(`知识点 ${node.id} 引用了不存在的先修 ${prerequisiteId}`);
        rank = Math.max(rank, visit(prerequisite) + 1);
      }
      visiting.delete(node.id);
      ranks.set(node.id, rank);
      return rank;
    }

    for (const node of nodes) visit(node);
    return ranks;
  }

  function createTopologyLayout(nodes, stages) {
    const ranks = calculateRanks(nodes);
    const maxRank = Math.max(0, ...ranks.values());
    const buckets = new Map();

    for (const node of nodes) {
      const key = `${node.stageIndex}:${ranks.get(node.id)}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(node);
    }
    for (const bucket of buckets.values()) {
      bucket.sort((a, b) => a.nodeIndexInStage - b.nodeIndexInStage);
    }

    const laneLayouts = [];
    let laneTop = geometry.graphPaddingTop;
    for (const [stageIndex, stage] of stages.entries()) {
      let maxBucketSize = 1;
      for (let rank = 0; rank <= maxRank; rank += 1) {
        maxBucketSize = Math.max(maxBucketSize, buckets.get(`${stageIndex}:${rank}`)?.length ?? 0);
      }
      const height = geometry.laneHeaderHeight
        + maxBucketSize * geometry.nodeHeight
        + Math.max(0, maxBucketSize - 1) * geometry.nodeGap
        + geometry.lanePaddingBottom;
      laneLayouts.push({
        id: stage.id,
        title: stage.title,
        index: stageIndex,
        y: laneTop,
        height,
        nodeCount: stage.nodes.length
      });
      laneTop += height + geometry.laneGap;
    }

    const positions = new Map();
    for (const node of nodes) {
      const rank = ranks.get(node.id);
      const bucket = buckets.get(`${node.stageIndex}:${rank}`);
      const bucketIndex = bucket.findIndex((entry) => entry.id === node.id);
      const lane = laneLayouts[node.stageIndex];
      positions.set(node.id, {
        id: node.id,
        rank,
        x: geometry.graphPaddingLeft + rank * geometry.rankGap,
        y: lane.y + geometry.laneHeaderHeight + bucketIndex * (geometry.nodeHeight + geometry.nodeGap),
        width: geometry.nodeWidth,
        height: geometry.nodeHeight
      });
    }

    const edges = nodes.flatMap((node) => node.prerequisites.map((sourceId) => ({
      sourceId,
      targetId: node.id
    })));

    return {
      ranks,
      positions,
      edges,
      lanes: laneLayouts,
      maxRank,
      width: geometry.graphPaddingLeft
        + maxRank * geometry.rankGap
        + geometry.nodeWidth
        + geometry.graphPaddingRight,
      height: laneTop - geometry.laneGap + geometry.graphPaddingBottom
    };
  }

  function collectRelationshipSets(selectedNodeId, nodes) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const directPrerequisites = new Set(byId.get(selectedNodeId)?.prerequisites ?? []);
    const dependents = new Map(nodes.map((node) => [node.id, []]));
    for (const node of nodes) {
      for (const prerequisiteId of node.prerequisites) dependents.get(prerequisiteId)?.push(node.id);
    }
    const directDependents = new Set(dependents.get(selectedNodeId) ?? []);

    function traverse(initialIds, nextIds) {
      const visited = new Set();
      const queue = [...initialIds];
      while (queue.length > 0) {
        const id = queue.shift();
        if (visited.has(id)) continue;
        visited.add(id);
        for (const nextId of nextIds(id)) queue.push(nextId);
      }
      return visited;
    }

    return {
      directPrerequisites,
      directDependents,
      ancestors: traverse(directPrerequisites, (id) => byId.get(id)?.prerequisites ?? []),
      descendants: traverse(directDependents, (id) => dependents.get(id) ?? [])
    };
  }

  function truncateLabel(value, maxUnits = 20) {
    let units = 0;
    let output = '';
    for (const character of value) {
      const weight = character.codePointAt(0) > 255 ? 2 : 1;
      if (units + weight > maxUnits) return `${output}…`;
      output += character;
      units += weight;
    }
    return output;
  }

  function createTopologyGraph(options) {
    const {
      container,
      nodes,
      stages,
      statusMeta,
      initialNodeId = nodes.find((node) => node.progress.status === 'current')?.id ?? nodes[0]?.id,
      onSelect,
      onTransform
    } = options;
    const layout = createTopologyLayout(nodes, stages);
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const nodeElements = new Map();
    const edgeElements = [];
    let selectedNodeId = null;
    let transform = { x: 0, y: 0, scale: 1 };
    let pointerStart = null;
    let viewMode = nodeById.has(initialNodeId) ? 'focus' : 'all';
    let focusedNodeId = initialNodeId;
    let previousViewportSize = null;
    let resizeFrame = null;
    const instanceId = `learning-topology-${Math.random().toString(36).slice(2, 9)}`;

    container.replaceChildren();
    const canvas = document.createElement('div');
    canvas.className = 'topology-canvas';
    const svg = createSvgElement('svg', {
      class: 'topology-svg',
      role: 'group',
      'aria-labelledby': `${instanceId}-title ${instanceId}-description`
    });
    svg.append(
      createSvgElement('title', { id: `${instanceId}-title` }, 'Transformer 知识依赖拓扑图'),
      createSvgElement('desc', { id: `${instanceId}-description` }, `${nodes.length} 个知识点按依赖深度从左向右排列，箭头从先修指向后续知识点。`)
    );

    const defs = createSvgElement('defs');
    const markers = {
      normal: createMarker(`${instanceId}-arrow`, 'topology-arrow-normal'),
      upstream: createMarker(`${instanceId}-arrow-upstream`, 'topology-arrow-upstream'),
      downstream: createMarker(`${instanceId}-arrow-downstream`, 'topology-arrow-downstream')
    };
    defs.append(markers.normal, markers.upstream, markers.downstream);
    svg.appendChild(defs);

    const viewport = createSvgElement('g', { class: 'topology-viewport' });
    const lanesLayer = createSvgElement('g', { class: 'topology-lanes' });
    const edgesLayer = createSvgElement('g', { class: 'topology-edges' });
    const nodesLayer = createSvgElement('g', { class: 'topology-nodes' });
    viewport.append(lanesLayer, edgesLayer, nodesLayer);
    svg.appendChild(viewport);
    canvas.appendChild(svg);
    container.appendChild(canvas);

    const stageCounts = new Map(stages.map((stage) => [
      stage.id,
      stage.nodes.filter((node) => node.progress.status === 'mastered').length
    ]));
    for (const lane of layout.lanes) {
      const hasCurrentNode = stages[lane.index].nodes.some((node) => node.progress.status === 'current');
      const laneGroup = createSvgElement('g', {
        class: `topology-lane${hasCurrentNode ? ' is-current' : ''}`,
        'data-stage-id': lane.id
      });
      const background = createSvgElement('rect', {
        class: 'topology-lane-background',
        x: 10,
        y: lane.y,
        width: layout.width - 20,
        height: lane.height,
        rx: 12
      });
      const number = createSvgElement('text', {
        class: 'topology-lane-number',
        x: 28,
        y: lane.y + 28
      }, String(lane.index + 1).padStart(2, '0'));
      const title = createSvgElement('text', {
        class: 'topology-lane-title',
        x: 60,
        y: lane.y + 27
      }, truncateLabel(lane.title.split('：')[0], 16));
      const count = createSvgElement('text', {
        class: 'topology-lane-count',
        x: geometry.graphPaddingLeft - 18,
        y: lane.y + 27,
        'text-anchor': 'end'
      }, `${stageCounts.get(lane.id)}/${lane.nodeCount}`);
      laneGroup.append(background, number, title, count);
      lanesLayer.appendChild(laneGroup);
    }

    for (let rank = 0; rank <= layout.maxRank; rank += 1) {
      const x = geometry.graphPaddingLeft + rank * geometry.rankGap;
      const label = createSvgElement('text', {
        class: 'topology-rank-label',
        x: x + geometry.nodeWidth / 2,
        y: 31,
        'text-anchor': 'middle'
      }, rank === 0 ? '路线起点' : `依赖层 ${rank}`);
      const guide = createSvgElement('line', {
        class: 'topology-rank-guide',
        x1: x + geometry.nodeWidth / 2,
        y1: 39,
        x2: x + geometry.nodeWidth / 2,
        y2: layout.height - geometry.graphPaddingBottom + 5
      });
      lanesLayer.append(guide, label);
    }

    for (const edge of layout.edges) {
      const source = layout.positions.get(edge.sourceId);
      const target = layout.positions.get(edge.targetId);
      const startX = source.x + source.width;
      const startY = source.y + source.height / 2;
      const endX = target.x;
      const endY = target.y + target.height / 2;
      const curve = Math.max(24, (endX - startX) * 0.46);
      const path = createSvgElement('path', {
        class: 'topology-edge',
        d: `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`,
        'data-source-id': edge.sourceId,
        'data-target-id': edge.targetId,
        'marker-end': `url(#${instanceId}-arrow)`
      });
      edgesLayer.appendChild(path);
      edgeElements.push({ ...edge, element: path });
    }

    for (const node of nodes) {
      const position = layout.positions.get(node.id);
      const group = createSvgElement('g', {
        class: `topology-node status-${node.progress.status}`,
        transform: `translate(${position.x} ${position.y})`,
        role: 'button',
        tabindex: '0',
        'aria-label': `${node.title}，${statusMeta[node.progress.status].label}`,
        'aria-pressed': 'false',
        'data-node-id': node.id
      });
      const title = createSvgElement('title', {}, `${node.title} · ${statusMeta[node.progress.status].label}`);
      const card = createSvgElement('rect', {
        class: 'topology-node-card',
        width: geometry.nodeWidth,
        height: geometry.nodeHeight,
        rx: 9
      });
      const statusBar = createSvgElement('rect', {
        class: 'topology-node-status-bar',
        width: 5,
        height: geometry.nodeHeight,
        rx: 3
      });
      const titleText = createSvgElement('text', {
        class: 'topology-node-title',
        x: 15,
        y: 21
      }, truncateLabel(node.title));
      const metaText = createSvgElement('text', {
        class: 'topology-node-meta',
        x: 15,
        y: 38
      }, `${node.stageIndex + 1}.${node.nodeIndexInStage + 1} · ${statusMeta[node.progress.status].shortLabel}`);
      const stateMark = createSvgElement('text', {
        class: 'topology-node-state-mark',
        x: geometry.nodeWidth - 12,
        y: 20,
        'text-anchor': 'end'
      }, statusMark(node.progress.status));
      group.append(title, card, statusBar, titleText, metaText, stateMark);
      group.addEventListener('click', () => onSelect(node.id));
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect(node.id);
        }
      });
      nodesLayer.appendChild(group);
      nodeElements.set(node.id, group);
    }

    function createMarker(id, className) {
      const marker = createSvgElement('marker', {
        id,
        class: className,
        viewBox: '0 0 8 8',
        refX: 7,
        refY: 4,
        markerWidth: 6,
        markerHeight: 6,
        orient: 'auto-start-reverse'
      });
      marker.appendChild(createSvgElement('path', { d: 'M 0 0 L 8 4 L 0 8 z' }));
      return marker;
    }

    function statusMark(status) {
      return {
        mastered: '✓',
        current: '◆',
        verify: '?',
        relearn: '↺',
        pending: '○'
      }[status] ?? '○';
    }

    function emitTransform() {
      previousViewportSize = getViewportSize();
      viewport.setAttribute('transform', `translate(${transform.x} ${transform.y}) scale(${transform.scale})`);
      canvas.classList.toggle('is-overview', transform.scale < 0.5);
      onTransform?.({ ...transform });
    }

    function getViewportSize() {
      const bounds = svg.getBoundingClientRect();
      return { width: bounds.width, height: bounds.height };
    }

    function fit() {
      const viewportSize = getViewportSize();
      if (viewportSize.width === 0 || viewportSize.height === 0) return;
      const padding = 22;
      const scale = Math.min(
        1,
        (viewportSize.width - padding * 2) / layout.width,
        (viewportSize.height - padding * 2) / layout.height
      );
      transform = {
        scale,
        x: (viewportSize.width - layout.width * scale) / 2,
        y: (viewportSize.height - layout.height * scale) / 2
      };
      viewMode = 'all';
      emitTransform();
    }

    function zoomAt(scaleFactor, clientX, clientY) {
      const bounds = svg.getBoundingClientRect();
      const pointX = clientX - bounds.left;
      const pointY = clientY - bounds.top;
      const nextScale = Math.min(1.8, Math.max(0.18, transform.scale * scaleFactor));
      const worldX = (pointX - transform.x) / transform.scale;
      const worldY = (pointY - transform.y) / transform.scale;
      transform = {
        scale: nextScale,
        x: pointX - worldX * nextScale,
        y: pointY - worldY * nextScale
      };
      viewMode = 'manual';
      emitTransform();
    }

    function zoom(scaleFactor) {
      const bounds = svg.getBoundingClientRect();
      zoomAt(scaleFactor, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
    }

    function focusNode(nodeId, { moveFocus = true } = {}) {
      const position = layout.positions.get(nodeId);
      if (!position) return;
      focusedNodeId = nodeId;
      viewMode = 'focus';
      const viewportSize = getViewportSize();
      if (viewportSize.width === 0 || viewportSize.height === 0) return;
      // 保证文字可读；远处依赖通过拖动或“查看全图”浏览，不再用整图尺寸压缩当前节点。
      const nextScale = 1.15;
      transform = {
        scale: nextScale,
        x: viewportSize.width / 2 - (position.x + position.width / 2) * nextScale,
        y: viewportSize.height / 2 - (position.y + position.height / 2) * nextScale
      };
      emitTransform();
      if (moveFocus) nodeElements.get(nodeId)?.focus({ preventScroll: true });
    }

    function refreshViewport() {
      const size = getViewportSize();
      if (!size.width || !size.height) return;
      if (viewMode === 'all') fit();
      else if (viewMode === 'focus') focusNode(focusedNodeId, { moveFocus: false });
      else {
        // 手动浏览时保留画布中心的世界坐标，适配全屏和容器尺寸变化。
        if (previousViewportSize) {
          transform.x += (size.width - previousViewportSize.width) / 2;
          transform.y += (size.height - previousViewportSize.height) / 2;
        }
        emitTransform();
      }
    }

    function selectNode(nodeId) {
      if (!nodeById.has(nodeId)) return;
      selectedNodeId = nodeId;
      const relationships = collectRelationshipSets(nodeId, nodes);
      const upstreamPath = new Set([...relationships.ancestors, nodeId]);
      const downstreamPath = new Set([...relationships.descendants, nodeId]);

      for (const [id, element] of nodeElements) {
        element.setAttribute('aria-pressed', String(id === nodeId));
        element.classList.toggle('is-selected', id === nodeId);
        element.classList.toggle('is-upstream', relationships.ancestors.has(id));
        element.classList.toggle('is-downstream', relationships.descendants.has(id));
        element.classList.toggle('is-direct-upstream', relationships.directPrerequisites.has(id));
        element.classList.toggle('is-direct-downstream', relationships.directDependents.has(id));
        element.classList.toggle(
          'is-unrelated',
          id !== nodeId && !relationships.ancestors.has(id) && !relationships.descendants.has(id)
        );
      }

      for (const edge of edgeElements) {
        const isUpstream = upstreamPath.has(edge.sourceId) && upstreamPath.has(edge.targetId);
        const isDownstream = downstreamPath.has(edge.sourceId) && downstreamPath.has(edge.targetId);
        const isDirect = (edge.targetId === nodeId && relationships.directPrerequisites.has(edge.sourceId))
          || (edge.sourceId === nodeId && relationships.directDependents.has(edge.targetId));
        edge.element.classList.toggle('is-upstream', isUpstream);
        edge.element.classList.toggle('is-downstream', isDownstream);
        edge.element.classList.toggle('is-direct', isDirect);
        edge.element.classList.toggle('is-unrelated', !isUpstream && !isDownstream);
        edge.element.setAttribute(
          'marker-end',
          `url(#${isUpstream ? `${instanceId}-arrow-upstream` : isDownstream ? `${instanceId}-arrow-downstream` : `${instanceId}-arrow`})`
        );
      }
    }

    function setFilter({ query = '', status = 'all' }) {
      const normalizedQuery = query.trim().toLocaleLowerCase('zh-CN');
      let matches = 0;
      for (const node of nodes) {
        const searchText = [node.title, node.summary, node.stageTitle, ...node.masteryCriteria]
          .join(' ')
          .toLocaleLowerCase('zh-CN');
        const isMatch = (!normalizedQuery || searchText.includes(normalizedQuery))
          && (status === 'all' || node.progress.status === status);
        const element = nodeElements.get(node.id);
        element?.classList.toggle('is-filtered-out', !isMatch);
        element?.setAttribute('tabindex', isMatch ? '0' : '-1');
        element?.setAttribute('aria-hidden', String(!isMatch));
        if (isMatch) matches += 1;
      }
      for (const edge of edgeElements) {
        const sourceHidden = nodeElements.get(edge.sourceId)?.classList.contains('is-filtered-out');
        const targetHidden = nodeElements.get(edge.targetId)?.classList.contains('is-filtered-out');
        edge.element.classList.toggle('is-filtered-out', sourceHidden || targetHidden);
      }
      return matches;
    }

    svg.addEventListener('wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX, event.clientY);
    }, { passive: false });

    svg.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || event.target.closest('.topology-node')) return;
      pointerStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        originX: transform.x,
        originY: transform.y
      };
      svg.setPointerCapture(event.pointerId);
      canvas.classList.add('is-dragging');
    });
    svg.addEventListener('pointermove', (event) => {
      if (!pointerStart || pointerStart.pointerId !== event.pointerId) return;
      transform.x = pointerStart.originX + event.clientX - pointerStart.x;
      transform.y = pointerStart.originY + event.clientY - pointerStart.y;
      viewMode = 'manual';
      emitTransform();
    });
    function stopDragging(event) {
      if (!pointerStart || pointerStart.pointerId !== event.pointerId) return;
      pointerStart = null;
      canvas.classList.remove('is-dragging');
    }
    svg.addEventListener('pointerup', stopDragging);
    svg.addEventListener('pointercancel', stopDragging);

    const resizeObserver = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(() => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(refreshViewport);
      });
    resizeObserver?.observe(svg);
    requestAnimationFrame(refreshViewport);

    return {
      fit,
      refreshViewport,
      zoomIn: () => zoom(1.2),
      zoomOut: () => zoom(1 / 1.2),
      focusNode,
      selectNode,
      setFilter,
      getLayout: () => layout,
      getScale: () => transform.scale,
      destroy: () => resizeObserver?.disconnect()
    };
  }

  root.LearningTopology = {
    calculateRanks,
    collectRelationshipSets,
    createTopologyGraph,
    createTopologyLayout
  };
})();
