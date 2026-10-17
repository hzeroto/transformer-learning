(() => {
  const root = typeof window === 'undefined' ? globalThis : window;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const geometry = { nodeWidth: 200, nodeHeight: 82, columnGap: 28, rowGap: 20, rankGap: 44, padding: 20 };

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

  function createTopologyLayout(nodes, stages, { width = 1200, stageId = null } = {}) {
    // 阶段图重新计算阶段内依赖层，不继承全局的巨大坐标偏移。
    const scoped = nodes.filter((node) => stageId === null || node.stageId === stageId);
    const ids = new Set(scoped.map((node) => node.id));
    const localNodes = scoped.map((node) => ({ ...node, prerequisites: node.prerequisites.filter((id) => ids.has(id)) }));
    const ranks = calculateRanks(localNodes);
    const maxRank = Math.max(0, ...ranks.values());
    const buckets = Array.from({ length: maxRank + 1 }, () => []);
    for (const node of localNodes) buckets[ranks.get(node.id)].push(node);
    for (const bucket of buckets) bucket.sort((a, b) => a.stageIndex - b.stageIndex || a.nodeIndexInStage - b.nodeIndexInStage);
    const maxBucket = Math.max(1, ...buckets.map((bucket) => bucket.length));
    const { nodeHeight, columnGap, rowGap, rankGap, padding } = geometry;
    const canvasWidth = Math.max(240, width);
    const horizontalWidth = (maxRank + 1) * geometry.nodeWidth + maxRank * rankGap + padding * 2;
    const direction = stageId !== null && horizontalWidth <= canvasWidth ? 'horizontal' : 'vertical';
    const positions = new Map();
    const levels = [];
    let height;
    if (direction === 'horizontal') {
      height = maxBucket * nodeHeight + (maxBucket - 1) * rowGap + padding * 2;
      const step = (canvasWidth - padding * 2 - geometry.nodeWidth) / Math.max(1, maxRank);
      for (const [rank, bucket] of buckets.entries()) {
        const startY = (height - bucket.length * nodeHeight - (bucket.length - 1) * rowGap) / 2;
        for (const [index, node] of bucket.entries()) positions.set(node.id, {
          id: node.id, rank, x: maxRank ? padding + rank * step : (canvasWidth - geometry.nodeWidth) / 2,
          y: startY + index * (nodeHeight + rowGap), width: geometry.nodeWidth, height: nodeHeight
        });
      }
    } else {
      const columns = Math.min(maxBucket, Math.max(1, Math.floor((canvasWidth - padding * 2 + columnGap) / (180 + columnGap))));
      const nodeWidth = Math.min(260, (canvasWidth - padding * 2 - (columns - 1) * columnGap) / columns);
      let top = padding;
      for (const [rank, bucket] of buckets.entries()) {
        if (!bucket.length) continue;
        levels.push({ rank, y: top });
        top += 24;
        for (let start = 0; start < bucket.length; start += columns) {
          const row = bucket.slice(start, start + columns);
          const left = (canvasWidth - row.length * nodeWidth - (row.length - 1) * columnGap) / 2;
          for (const [index, node] of row.entries()) positions.set(node.id, {
            id: node.id, rank, x: left + index * (nodeWidth + columnGap), y: top,
            width: nodeWidth, height: nodeHeight
          });
          top += nodeHeight + rowGap;
        }
        top += rankGap - rowGap;
      }
      height = Math.max(nodeHeight, top - rankGap + padding);
    }
    return {
      ranks, positions, levels, maxRank, direction, width: canvasWidth, height,
      edges: localNodes.flatMap((node) => node.prerequisites.map((sourceId) => ({ sourceId, targetId: node.id })))
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

  function wrapLabel(value, maxUnits) {
    const lines = [];
    let line = '';
    let units = 0;
    for (const character of value) {
      const weight = character.codePointAt(0) > 255 ? 2 : 1;
      if (units + weight > maxUnits && line) {
        lines.push(line);
        line = '';
        units = 0;
      }
      line += character;
      units += weight;
    }
    if (line) lines.push(line);
    return lines;
  }

  function createTopologyGraph({ container, nodes, stages, statusMeta, initialNodeId = null, onSelect, onTransform }) {
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const nodeElements = new Map();
    const edgeElements = [];
    let layout = createTopologyLayout(nodes, stages);
    let filter = { query: '', status: 'all', stageId: null };
    let selectedNodeId = null;
    let scale = 1;
    let offset = { x: 0, y: 0 };
    let pointerStart = null;
    let resizeFrame;
    let pendingFocusId = initialNodeId;
    let previousSize = null;
    const instanceId = 'learning-topology-' + Math.random().toString(36).slice(2, 9);
    const canvas = document.createElement('div');
    canvas.className = 'topology-canvas';
    canvas.setAttribute('tabindex', '0');
    canvas.setAttribute('aria-label', '知识依赖图，可滚动浏览全部节点');
    const svg = createSvgElement('svg', {
      class: 'topology-svg', role: 'group',
      'aria-labelledby': instanceId + '-title ' + instanceId + '-description'
    });
    svg.append(
      createSvgElement('title', { id: instanceId + '-title' }, 'Transformer 知识依赖拓扑图'),
      createSvgElement('desc', { id: instanceId + '-description' }, '全局按依赖层从上向下排列，阶段内按箭头方向阅读；滚动可查看完整课程，点击节点查看先修与后续。')
    );
    const defs = createSvgElement('defs');
    for (const kind of ['normal', 'upstream', 'downstream']) {
      const marker = createSvgElement('marker', {
        id: instanceId + '-arrow-' + kind, class: 'topology-arrow-' + kind,
        viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse'
      });
      marker.appendChild(createSvgElement('path', { d: 'M 0 0 L 8 4 L 0 8 z' }));
      defs.appendChild(marker);
    }
    const viewport = createSvgElement('g', { class: 'topology-viewport' });
    const levelsLayer = createSvgElement('g', { class: 'topology-levels' });
    const edgesLayer = createSvgElement('g', { class: 'topology-edges' });
    const nodesLayer = createSvgElement('g', { class: 'topology-nodes' });
    viewport.append(levelsLayer, edgesLayer, nodesLayer);
    svg.append(defs, viewport);
    canvas.appendChild(svg);
    container.replaceChildren(canvas);

    for (const node of nodes) {
      for (const sourceId of node.prerequisites) {
        const path = createSvgElement('path', {
          class: 'topology-edge', 'data-source-id': sourceId, 'data-target-id': node.id,
          'marker-end': 'url(#' + instanceId + '-arrow-normal)'
        });
        edgesLayer.appendChild(path);
        edgeElements.push({ sourceId, targetId: node.id, element: path });
      }
      const group = createSvgElement('g', {
        class: 'topology-node status-' + node.progress.status, role: 'button', tabindex: '0',
        'aria-label': node.title + '，' + node.stageTitle + '，' + statusMeta[node.progress.status].label,
        'aria-pressed': 'false', 'data-node-id': node.id
      });
      const title = createSvgElement('title', {}, node.title + ' · ' + statusMeta[node.progress.status].label);
      const card = createSvgElement('rect', { class: 'topology-node-card', rx: 8 });
      const bar = createSvgElement('rect', { class: 'topology-node-status-bar', width: 4, rx: 2 });
      const label = createSvgElement('text', { class: 'topology-node-title', x: 13 });
      const meta = createSvgElement('text', { class: 'topology-node-meta', x: 13 });
      group.append(title, card, bar, label, meta);
      group.addEventListener('click', () => onSelect(node.id));
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect(node.id);
        }
      });
      nodesLayer.appendChild(group);
      nodeElements.set(node.id, { group, card, bar, label, meta });
    }

    function getSize() {
      return { width: canvas.clientWidth, height: canvas.clientHeight };
    }

    function applyScale() {
      const size = getSize();
      const width = Math.max(size.width, layout.width * scale);
      const height = Math.max(size.height, layout.height * scale);
      offset = {
        x: (width - layout.width * scale) / 2,
        y: filter.stageId === null ? 0 : (height - layout.height * scale) / 2
      };
      svg.style.width = width + 'px';
      svg.style.height = height + 'px';
      viewport.setAttribute('transform', 'translate(' + offset.x + ' ' + offset.y + ') scale(' + scale + ')');
      onTransform?.({ scale });
    }

    function redrawLayout() {
      const size = getSize();
      if (!size.width || !size.height) return false;
      layout = createTopologyLayout(nodes, stages, { width: size.width, stageId: filter.stageId });
      container.classList.toggle('is-stage-scope', filter.stageId !== null);
      container.style.setProperty('--topology-content-height', (layout.height + 48) + 'px');
      svg.setAttribute('data-direction', layout.direction);
      levelsLayer.replaceChildren();
      for (const level of layout.levels) {
        levelsLayer.append(
          createSvgElement('text', { class: 'topology-rank-label', x: 20, y: level.y + 12 }, level.rank === 0 ? '起点 · 无本图内先修' : '依赖层 ' + level.rank),
          createSvgElement('line', { class: 'topology-rank-guide', x1: 140, x2: layout.width - 20, y1: level.y + 8, y2: level.y + 8 })
        );
      }
      for (const node of nodes) {
        const position = layout.positions.get(node.id);
        const { group, card, bar, label, meta } = nodeElements.get(node.id);
        const search = [node.title, node.summary, node.stageTitle, ...node.masteryCriteria].join(' ').toLocaleLowerCase('zh-CN');
        const matches = Boolean(position) && (!filter.query || search.includes(filter.query))
          && (filter.status === 'all' || node.progress.status === filter.status);
        group.classList.toggle('is-outside-stage', !position);
        group.classList.toggle('is-filtered-out', !matches);
        group.setAttribute('aria-hidden', String(!matches));
        group.setAttribute('tabindex', matches ? '0' : '-1');
        if (!position) continue;
        group.setAttribute('transform', 'translate(' + position.x + ' ' + position.y + ')');
        card.setAttribute('width', position.width);
        card.setAttribute('height', position.height);
        bar.setAttribute('height', position.height);
        label.replaceChildren();
        const lines = wrapLabel(node.title, Math.floor((position.width - 26) / 7.5));
        const startY = lines.length === 1 ? 33 : 24;
        for (const [index, line] of lines.entries()) {
          label.appendChild(createSvgElement('tspan', { x: 13, y: startY + index * 17 }, line));
        }
        const outsidePrerequisites = node.prerequisites.filter((id) => !layout.positions.has(id)).length;
        meta.setAttribute('y', position.height - 12);
        // 编号对应侧栏阶段；不要为塞入长阶段名而横向压扁文字。
        meta.textContent = (node.stageIndex + 1) + '.' + (node.nodeIndexInStage + 1) + ' · '
          + statusMeta[node.progress.status].label
          + (filter.stageId !== null && outsidePrerequisites ? ' · ' + outsidePrerequisites + ' 个跨阶段先修' : '');
      }
      for (const [edgeIndex, edge] of edgeElements.entries()) {
        const source = layout.positions.get(edge.sourceId);
        const target = layout.positions.get(edge.targetId);
        const visible = source && target;
        edge.element.classList.toggle('is-outside-stage', !visible);
        edge.element.classList.toggle('is-filtered-out', !visible
          || nodeElements.get(edge.sourceId).group.classList.contains('is-filtered-out')
          || nodeElements.get(edge.targetId).group.classList.contains('is-filtered-out'));
        if (!visible) continue;
        edge.element.classList.toggle('is-long-edge', layout.direction === 'horizontal'
          ? target.rank > source.rank + 1
          : target.y - source.y - source.height > geometry.rankGap + 24);
        let path;
        if (layout.direction === 'horizontal') {
          const x1 = source.x + source.width;
          const y1 = source.y + source.height / 2;
          const x2 = target.x;
          const y2 = target.y + target.height / 2;
          const curve = Math.max(20, (x2 - x1) / 2);
          if (target.rank === source.rank + 1) {
            path = 'M ' + x1 + ' ' + y1 + ' C ' + (x1 + curve) + ' ' + y1 + ', ' + (x2 - curve) + ' ' + y2 + ', ' + x2 + ' ' + y2;
          } else {
            const rail = 5 + (edgeIndex % 5) * 2;
            path = 'M ' + x1 + ' ' + y1 + ' H ' + (x1 + 10) + ' V ' + rail + ' H ' + (x2 - 10) + ' V ' + y2 + ' H ' + x2;
          }
        } else {
          const x1 = source.x + source.width / 2;
          const y1 = source.y + source.height;
          const x2 = target.x + target.width / 2;
          const y2 = target.y;
          const curve = Math.max(20, (y2 - y1) / 2);
          if (y2 - y1 <= geometry.rankGap + 24) {
            path = 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + (y1 + curve) + ', ' + x2 + ' ' + (y2 - curve) + ', ' + x2 + ' ' + y2;
          } else {
            // 跨行/跨层边从卡片间隙绕行，避免穿过无关节点造成错误的依赖暗示。
            const inset = 5 + (edgeIndex % 5) * 2;
            const rail = x1 + x2 < layout.width ? inset : layout.width - inset;
            path = 'M ' + x1 + ' ' + y1 + ' V ' + (y1 + 8) + ' H ' + rail + ' V ' + (y2 - 8) + ' H ' + x2 + ' V ' + y2;
          }
        }
        edge.element.setAttribute('d', path);
      }
      previousSize = size;
      applyScale();
      return true;
    }

    function fit() {
      scale = 1;
      pendingFocusId = null;
      redrawLayout();
      canvas.scrollLeft = 0;
      canvas.scrollTop = 0;
    }

    function focusNode(nodeId, { moveFocus = true } = {}) {
      if (!nodeById.has(nodeId)) return;
      pendingFocusId = nodeId;
      const position = layout.positions.get(nodeId);
      if (!position) return;
      const size = getSize();
      if (!size.width || !size.height) return;
      canvas.scrollLeft = Math.max(0, (position.x + position.width / 2) * scale + offset.x - size.width / 2);
      canvas.scrollTop = Math.max(0, (position.y + position.height / 2) * scale + offset.y - size.height / 2);
      pendingFocusId = null;
      if (moveFocus) nodeElements.get(nodeId).group.focus({ preventScroll: true });
    }

    function refreshViewport() {
      const size = getSize();
      if (!size.width || !size.height) return;
      const scrollFraction = canvas.scrollTop / Math.max(1, canvas.scrollHeight - (previousSize?.height || size.height));
      const selectedPosition = layout.positions.get(selectedNodeId);
      const selectedWasVisible = selectedPosition && selectedPosition.y * scale + offset.y >= canvas.scrollTop
        && selectedPosition.y * scale + offset.y < canvas.scrollTop + size.height;
      if (!redrawLayout()) return;
      if (pendingFocusId || selectedWasVisible) focusNode(pendingFocusId || selectedNodeId, { moveFocus: false });
      else canvas.scrollTop = scrollFraction * Math.max(0, canvas.scrollHeight - size.height);
    }

    function zoomAt(factor, clientX, clientY) {
      const bounds = canvas.getBoundingClientRect();
      const x = clientX - bounds.left;
      const y = clientY - bounds.top;
      const worldX = (canvas.scrollLeft + x - offset.x) / scale;
      const worldY = (canvas.scrollTop + y - offset.y) / scale;
      // 用户可主动缩小，但默认/阶段导航始终回到 100% 可读字号。
      scale = Math.max(0.6, Math.min(1.8, scale * factor));
      applyScale();
      canvas.scrollLeft = Math.max(0, worldX * scale + offset.x - x);
      canvas.scrollTop = Math.max(0, worldY * scale + offset.y - y);
    }

    function zoom(factor) {
      const bounds = canvas.getBoundingClientRect();
      zoomAt(factor, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
    }

    function selectNode(nodeId) {
      if (nodeId !== null && !nodeById.has(nodeId)) return;
      selectedNodeId = nodeId;
      const relationships = collectRelationshipSets(nodeId, nodes);
      const upstream = new Set([...relationships.ancestors, nodeId]);
      const downstream = new Set([...relationships.descendants, nodeId]);
      for (const [id, { group }] of nodeElements) {
        group.setAttribute('aria-pressed', String(id === nodeId));
        group.classList.toggle('is-selected', id === nodeId);
        group.classList.toggle('is-upstream', relationships.ancestors.has(id));
        group.classList.toggle('is-downstream', relationships.descendants.has(id));
        group.classList.toggle('is-direct-upstream', relationships.directPrerequisites.has(id));
        group.classList.toggle('is-direct-downstream', relationships.directDependents.has(id));
        group.classList.toggle('is-unrelated', nodeId !== null && !upstream.has(id) && !downstream.has(id));
      }
      for (const edge of edgeElements) {
        const isUpstream = upstream.has(edge.sourceId) && upstream.has(edge.targetId);
        const isDownstream = downstream.has(edge.sourceId) && downstream.has(edge.targetId);
        edge.element.classList.toggle('is-upstream', isUpstream);
        edge.element.classList.toggle('is-downstream', isDownstream);
        edge.element.classList.toggle('is-direct', edge.targetId === nodeId || edge.sourceId === nodeId);
        edge.element.classList.toggle('is-unrelated', nodeId !== null && !isUpstream && !isDownstream);
        edge.element.setAttribute('marker-end', 'url(#' + instanceId + '-arrow-' + (isUpstream ? 'upstream' : isDownstream ? 'downstream' : 'normal') + ')');
      }
    }

    function setFilter({ query = '', status = 'all', stageId = null }) {
      const scopeChanged = filter.stageId !== stageId;
      filter = { query: query.trim().toLocaleLowerCase('zh-CN'), status, stageId };
      if (scopeChanged) fit();
      else redrawLayout();
      return nodes.filter((node) => (stageId === null || node.stageId === stageId)
        && (status === 'all' || node.progress.status === status)
        && (!filter.query || [node.title, node.summary, node.stageTitle, ...node.masteryCriteria].join(' ').toLocaleLowerCase('zh-CN').includes(filter.query))).length;
    }

    canvas.addEventListener('wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX, event.clientY);
    }, { passive: false });
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || event.pointerType === 'touch' || event.target.closest('.topology-node')) return;
      pointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY, left: canvas.scrollLeft, top: canvas.scrollTop };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add('is-dragging');
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!pointerStart || pointerStart.id !== event.pointerId) return;
      canvas.scrollLeft = pointerStart.left - (event.clientX - pointerStart.x);
      canvas.scrollTop = pointerStart.top - (event.clientY - pointerStart.y);
    });
    function stopDragging() {
      pointerStart = null;
      canvas.classList.remove('is-dragging');
    }
    canvas.addEventListener('pointerup', stopDragging);
    canvas.addEventListener('pointercancel', stopDragging);
    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(refreshViewport);
    });
    resizeObserver?.observe(container);
    requestAnimationFrame(refreshViewport);
    return {
      fit, refreshViewport, focusNode, selectNode, setFilter,
      zoomIn: () => zoom(1.2), zoomOut: () => zoom(1 / 1.2),
      getLayout: () => layout, getScale: () => scale,
      destroy: () => { resizeObserver?.disconnect(); cancelAnimationFrame(resizeFrame); }
    };
  }

  root.LearningTopology = {
    calculateRanks,
    collectRelationshipSets,
    createTopologyGraph,
    createTopologyLayout
  };
})();
