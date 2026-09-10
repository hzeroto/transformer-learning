(() => {
  const data = window.LEARNING_DATA;
  if (!data) {
    document.body.textContent = '学习数据未生成，请先运行 npm run learning:build';
    return;
  }

  const statusMeta = {
    mastered: { label: '已掌握', shortLabel: '掌握' },
    current: { label: '当前学习', shortLabel: '当前' },
    verify: { label: '待验证', shortLabel: '验证' },
    relearn: { label: '待系统重学', shortLabel: '重学' },
    pending: { label: '未开始', shortLabel: '未开始' }
  };

  const allNodes = data.stages.flatMap((stage, stageIndex) =>
    stage.nodes.map((node, nodeIndexInStage) => ({
      ...node,
      stageId: stage.id,
      stageTitle: stage.title,
      stageIndex,
      nodeIndexInStage
    }))
  );
  const nodeIndex = new Map(allNodes.map((node) => [node.id, node]));
  const dependents = new Map(allNodes.map((node) => [node.id, []]));
  for (const node of allNodes) {
    for (const prerequisiteId of node.prerequisites) {
      dependents.get(prerequisiteId)?.push(node.id);
    }
  }

  const elements = {
    goalDescription: document.getElementById('goal-description'),
    curriculumSize: document.getElementById('curriculum-size'),
    updatedAt: document.getElementById('updated-at'),
    masteryCount: document.getElementById('mastery-count'),
    masteryPercentage: document.getElementById('mastery-percentage'),
    masteryCaption: document.getElementById('mastery-caption'),
    masteryDial: document.getElementById('mastery-dial'),
    progressFill: document.getElementById('progress-fill'),
    progressTrack: document.querySelector('.progress-track'),
    currentNode: document.getElementById('current-node'),
    currentStage: document.getElementById('current-stage'),
    currentSummary: document.getElementById('current-summary'),
    currentReadiness: document.getElementById('current-readiness'),
    currentDetailButton: document.getElementById('current-detail-button'),
    currentStageButton: document.getElementById('current-stage-button'),
    reviewCount: document.getElementById('review-count'),
    verifyCount: document.getElementById('verify-count'),
    relearnCount: document.getElementById('relearn-count'),
    readyCount: document.getElementById('ready-count'),
    stageSummary: document.getElementById('stage-summary'),
    stageOverview: document.getElementById('stage-overview'),
    goalTitle: document.getElementById('goal-title'),
    graduationCriteria: document.getElementById('graduation-criteria'),
    searchInput: document.getElementById('search-input'),
    statusFilters: document.getElementById('status-filters'),
    resetFilters: document.getElementById('reset-filters'),
    toggleStages: document.getElementById('toggle-stages'),
    mapPanel: document.getElementById('map-panel'),
    mapViewSwitch: document.querySelector('.map-view-switch'),
    topologyPanel: document.getElementById('topology-panel'),
    stagesPanel: document.getElementById('stages-panel'),
    topology: document.getElementById('learning-topology'),
    topologyFullscreen: document.getElementById('topology-fullscreen'),
    topologyFit: document.getElementById('topology-fit'),
    topologyCurrent: document.getElementById('topology-current'),
    topologyZoomIn: document.getElementById('topology-zoom-in'),
    topologyZoomOut: document.getElementById('topology-zoom-out'),
    topologyScale: document.getElementById('topology-scale'),
    topologyEmpty: document.getElementById('topology-empty'),
    map: document.getElementById('learning-map'),
    visibleCount: document.getElementById('visible-count'),
    filterEmpty: document.getElementById('filter-empty'),
    detailPosition: document.getElementById('detail-position'),
    detailTitle: document.getElementById('detail-title'),
    detailDot: document.getElementById('detail-dot'),
    detailStatus: document.getElementById('detail-status'),
    detailSummary: document.getElementById('detail-summary'),
    detailReadiness: document.getElementById('detail-readiness'),
    detailNote: document.getElementById('detail-note'),
    detailPrerequisites: document.getElementById('detail-prerequisites'),
    detailDependents: document.getElementById('detail-dependents'),
    prerequisiteCount: document.getElementById('prerequisite-count'),
    dependentCount: document.getElementById('dependent-count'),
    detailCriteria: document.getElementById('detail-criteria'),
    detailEvidence: document.getElementById('detail-evidence'),
    locateNode: document.getElementById('locate-node'),
    recordsList: document.getElementById('records-list'),
    recordsToggle: document.getElementById('records-toggle')
  };

  const hashNodeId = decodeURIComponent(window.location.hash.slice(1));
  let selectedNodeId = nodeIndex.has(hashNodeId)
    ? hashNodeId
    : data.progress.currentNodeId || allNodes[0]?.id;
  let activeStatus = 'all';
  let activeMapView = 'topology';
  let showAllRecords = false;
  let topologyGraph = null;
  let pseudoFullscreen = false;
  const nodeButtons = new Map();
  const stageSections = new Map();
  const expandedStages = new Set(
    data.stages
      .filter((stage) => stage.nodes.some((node) => node.progress.status !== 'pending'))
      .map((stage) => stage.id)
  );

  function formatDate(value) {
    if (!value) return '尚未更新';
    const date = new Date(value);
    const options = {
      month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hour12: false
    };
    if (date.getFullYear() !== new Date().getFullYear()) options.year = 'numeric';
    return new Intl.DateTimeFormat('zh-CN', options).format(date);
  }

  function createStatusDot(status) {
    const dot = document.createElement('span');
    dot.className = `status-dot ${status}`;
    dot.setAttribute('aria-hidden', 'true');
    return dot;
  }

  function createStatusBadge(status) {
    const badge = document.createElement('span');
    badge.className = `mini-status ${status}`;
    badge.append(createStatusDot(status), statusMeta[status].shortLabel);
    return badge;
  }

  function getMissingPrerequisites(node) {
    return node.prerequisites.filter(
      (id) => nodeIndex.get(id)?.progress.status !== 'mastered'
    );
  }

  function getStageCounts(stage) {
    const counts = { mastered: 0, current: 0, verify: 0, relearn: 0, pending: 0 };
    for (const node of stage.nodes) counts[node.progress.status] += 1;
    return counts;
  }

  function setHash(nodeId) {
    const hash = `#${encodeURIComponent(nodeId)}`;
    if (window.location.hash !== hash) history.replaceState(null, '', hash);
  }

  function showNode(nodeId, options = {}) {
    const node = nodeIndex.get(nodeId);
    if (!node) return;
    selectedNodeId = nodeId;
    setHash(nodeId);

    for (const [id, button] of nodeButtons) {
      button.setAttribute('aria-pressed', String(id === nodeId));
      button.classList.toggle('is-prerequisite', node.prerequisites.includes(id));
      button.classList.toggle('is-dependent', (dependents.get(node.id) ?? []).includes(id));
    }
    topologyGraph?.selectNode(nodeId);

    const status = node.progress.status;
    const missing = getMissingPrerequisites(node);
    elements.detailPosition.textContent = `阶段 ${node.stageIndex + 1} / ${data.stages.length} · ${node.stageTitle}`;
    elements.detailTitle.textContent = node.title;
    elements.detailDot.className = `status-dot ${status}`;
    elements.detailStatus.textContent = statusMeta[status].label;
    elements.detailStatus.className = `status-label ${status}`;
    elements.detailSummary.textContent = node.summary;

    elements.detailReadiness.className = 'detail-readiness';
    if (status === 'mastered') {
      elements.detailReadiness.classList.add('is-ready');
      elements.detailReadiness.textContent = '已有可核验的掌握证据';
    } else if (missing.length === 0) {
      elements.detailReadiness.classList.add('is-ready');
      elements.detailReadiness.textContent = '直接先修均已掌握，可以推进';
    } else {
      elements.detailReadiness.classList.add('is-blocked');
      elements.detailReadiness.textContent = `还有 ${missing.length} 个直接先修未掌握`;
    }

    const note = node.progress.note;
    elements.detailNote.hidden = !note;
    elements.detailNote.textContent = note ? `当前备注：${note}` : '';

    elements.prerequisiteCount.textContent = node.prerequisites.length
      ? `${node.prerequisites.length - missing.length}/${node.prerequisites.length} 已掌握`
      : '无先修';
    elements.dependentCount.textContent = `${dependents.get(node.id)?.length ?? 0} 个直接后续`;
    renderDependencies(elements.detailPrerequisites, node.prerequisites, '这是一个路线起点，没有直接先修。');
    renderDependencies(elements.detailDependents, dependents.get(node.id) ?? [], '这是当前路线的终点之一。');

    elements.detailCriteria.replaceChildren();
    for (const criterion of node.masteryCriteria) {
      const item = document.createElement('li');
      item.className = status === 'mastered' ? 'is-complete' : '';
      const marker = document.createElement('span');
      marker.setAttribute('aria-hidden', 'true');
      marker.textContent = status === 'mastered' ? '✓' : '';
      const text = document.createElement('span');
      text.textContent = criterion;
      item.append(marker, text);
      elements.detailCriteria.appendChild(item);
    }

    elements.detailEvidence.replaceChildren();
    const evidence = node.progress.evidence ?? [];
    if (evidence.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'evidence-empty';
      empty.innerHTML = '<span aria-hidden="true">○</span><div><strong>尚无掌握证据</strong><p>需要通过解释、实现、测试或排错来证明。</p></div>';
      elements.detailEvidence.appendChild(empty);
    } else {
      for (const entry of [...evidence].reverse()) {
        const item = document.createElement('div');
        item.className = 'evidence-item';
        const marker = document.createElement('span');
        marker.className = 'evidence-check';
        marker.textContent = '✓';
        marker.setAttribute('aria-hidden', 'true');
        const content = document.createElement('div');
        const text = document.createElement('p');
        text.textContent = entry.text;
        const time = document.createElement('time');
        time.dateTime = entry.at;
        time.textContent = formatDate(entry.at);
        content.append(text, time);
        item.append(marker, content);
        elements.detailEvidence.appendChild(item);
      }
    }

    if (options.locate) locateSelectedNode();
  }

  function renderDependencies(container, ids, emptyText) {
    container.replaceChildren();
    if (ids.length === 0) {
      const empty = document.createElement('span');
      empty.className = 'empty-state';
      empty.textContent = emptyText;
      container.appendChild(empty);
      return;
    }
    for (const id of ids) {
      const dependency = nodeIndex.get(id);
      if (!dependency) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'dependency-button';
      const title = document.createElement('span');
      title.textContent = dependency.title;
      button.append(title, createStatusBadge(dependency.progress.status));
      button.addEventListener('click', () => showNode(id));
      container.appendChild(button);
    }
  }

  function renderOverview() {
    const counts = data.progress.statusCounts;
    const percentage = data.progress.totalNodes === 0
      ? 0
      : Math.round((counts.mastered / data.progress.totalNodes) * 100);
    const readyNodes = allNodes.filter(
      (node) => node.progress.status !== 'mastered' && getMissingPrerequisites(node).length === 0
    );
    const touchedStages = data.stages.filter(
      (stage) => stage.nodes.some((node) => node.progress.status !== 'pending')
    ).length;

    elements.goalDescription.textContent = data.goal.description;
    elements.curriculumSize.textContent = `${data.stages.length} 个阶段 · ${data.progress.totalNodes} 个知识点`;
    elements.updatedAt.textContent = `更新于 ${formatDate(data.progress.updatedAt || data.generatedAt)}`;
    elements.masteryCount.textContent = `${counts.mastered} / ${data.progress.totalNodes}`;
    elements.masteryPercentage.textContent = `${percentage}%`;
    elements.masteryCaption.textContent = percentage === 0
      ? '从第一个可核验能力开始'
      : '知识点形成了可核验能力';
    elements.masteryDial.style.setProperty('--progress', `${percentage * 3.6}deg`);
    elements.progressFill.style.width = `${percentage}%`;
    elements.progressTrack.setAttribute('aria-valuenow', String(percentage));
    elements.reviewCount.textContent = String(counts.verify + counts.relearn);
    elements.verifyCount.textContent = String(counts.verify);
    elements.relearnCount.textContent = String(counts.relearn);
    elements.readyCount.textContent = `${readyNodes.length} 个知识点`;
    elements.stageSummary.textContent = `已触达 ${touchedStages} / ${data.stages.length} 个阶段`;

    const current = nodeIndex.get(data.progress.currentNodeId);
    elements.currentNode.textContent = current?.title ?? '尚未指定当前知识点';
    elements.currentStage.textContent = current
      ? `阶段 ${current.stageIndex + 1} · ${current.stageTitle}`
      : '请通过学习上报脚本设置当前入口';
    elements.currentSummary.textContent = current?.summary ?? '选定唯一的当前知识点后，这里会显示本次学习焦点。';
    if (!current) {
      document.getElementById('current-status').textContent = '未设置';
      document.getElementById('current-status').className = 'status-label pending';
      elements.currentReadiness.textContent = '当前入口尚未设置';
      elements.currentDetailButton.disabled = true;
      elements.currentStageButton.disabled = true;
    } else {
      const missing = getMissingPrerequisites(current);
      elements.currentReadiness.textContent = missing.length === 0
        ? '✓ 直接先修已就绪'
        : `${missing.length} 个直接先修尚未掌握`;
    }

    elements.goalTitle.textContent = data.goal.title;
    elements.graduationCriteria.replaceChildren();
    for (const [index, criterion] of data.goal.graduationCriteria.entries()) {
      const item = document.createElement('li');
      const number = document.createElement('span');
      number.textContent = String(index + 1).padStart(2, '0');
      const text = document.createElement('p');
      text.textContent = criterion;
      item.append(number, text);
      elements.graduationCriteria.appendChild(item);
    }

    for (const button of elements.statusFilters.querySelectorAll('[data-status-filter]')) {
      const status = button.dataset.statusFilter;
      const count = status === 'all' ? data.progress.totalNodes : counts[status];
      button.querySelector('span').textContent = String(count);
    }
  }

  function renderStageOverview() {
    elements.stageOverview.replaceChildren();
    for (const [index, stage] of data.stages.entries()) {
      const counts = getStageCounts(stage);
      const percentage = Math.round((counts.mastered / stage.nodes.length) * 100);
      const isCurrent = stage.nodes.some((node) => node.progress.status === 'current');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `stage-summary-card${isCurrent ? ' is-current' : ''}`;
      button.dataset.stageTarget = stage.id;
      button.setAttribute('aria-label', `查看阶段 ${index + 1}：${stage.title}，已掌握 ${counts.mastered}/${stage.nodes.length}`);

      const top = document.createElement('span');
      top.className = 'stage-card-top';
      const number = document.createElement('span');
      number.className = 'stage-number';
      number.textContent = String(index + 1).padStart(2, '0');
      const state = document.createElement('span');
      state.className = 'stage-state';
      state.textContent = isCurrent ? '进行中' : percentage === 100 ? '完成' : `${percentage}%`;
      top.append(number, state);

      const title = document.createElement('strong');
      title.textContent = stage.title.split('：')[0];
      const track = document.createElement('span');
      track.className = 'mini-progress';
      const fill = document.createElement('span');
      fill.style.width = `${percentage}%`;
      track.appendChild(fill);
      const count = document.createElement('span');
      count.className = 'stage-card-count';
      count.textContent = `${counts.mastered}/${stage.nodes.length} 掌握`;
      button.append(top, title, track, count);
      button.addEventListener('click', () => revealStage(stage.id));
      elements.stageOverview.appendChild(button);
    }
  }

  function renderMap() {
    for (const [stageIndex, stage] of data.stages.entries()) {
      const counts = getStageCounts(stage);
      const section = document.createElement('section');
      section.className = 'stage';
      section.dataset.stageId = stage.id;

      const headerButton = document.createElement('button');
      headerButton.type = 'button';
      headerButton.className = 'stage-header';
      headerButton.setAttribute('aria-expanded', String(expandedStages.has(stage.id)));
      const marker = document.createElement('span');
      marker.className = 'stage-map-number';
      marker.textContent = String(stageIndex + 1).padStart(2, '0');
      const heading = document.createElement('span');
      heading.className = 'stage-heading-copy';
      const title = document.createElement('strong');
      title.textContent = stage.title;
      const description = document.createElement('span');
      description.textContent = stage.description;
      heading.append(title, description);
      const completion = document.createElement('span');
      completion.className = 'stage-completion';
      const completionValue = document.createElement('strong');
      completionValue.textContent = `${counts.mastered}/${stage.nodes.length}`;
      const completionLabel = document.createElement('span');
      completionLabel.textContent = '已掌握';
      completion.append(completionValue, completionLabel);
      const chevron = document.createElement('span');
      chevron.className = 'stage-chevron';
      chevron.setAttribute('aria-hidden', 'true');
      chevron.textContent = '⌄';
      headerButton.append(marker, heading, completion, chevron);

      const body = document.createElement('div');
      body.className = 'stage-body';
      const statusLine = document.createElement('div');
      statusLine.className = 'stage-status-line';
      for (const status of ['mastered', 'current', 'verify', 'relearn', 'pending']) {
        if (counts[status] === 0) continue;
        const item = document.createElement('span');
        item.append(createStatusDot(status), `${statusMeta[status].shortLabel} ${counts[status]}`);
        statusLine.appendChild(item);
      }

      const grid = document.createElement('div');
      grid.className = 'node-grid';
      for (const [nodeInStageIndex, node] of stage.nodes.entries()) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'node-button';
        button.dataset.nodeId = node.id;
        button.dataset.status = node.progress.status;
        button.dataset.search = [
          node.title,
          node.summary,
          stage.title,
          ...node.masteryCriteria
        ].join(' ').toLocaleLowerCase('zh-CN');
        button.setAttribute('aria-pressed', String(node.id === selectedNodeId));

        const top = document.createElement('span');
        top.className = 'node-topline';
        const number = document.createElement('span');
        number.className = 'node-number';
        number.textContent = `${stageIndex + 1}.${nodeInStageIndex + 1}`;
        top.append(number, createStatusBadge(node.progress.status));

        const title = document.createElement('strong');
        title.className = 'node-title';
        title.textContent = node.title;
        const summary = document.createElement('span');
        summary.className = 'node-summary';
        summary.textContent = node.summary;
        const missing = getMissingPrerequisites(node);
        const meta = document.createElement('span');
        meta.className = `node-meta${missing.length === 0 ? ' is-ready' : ''}`;
        meta.textContent = node.progress.status === 'mastered'
          ? '✓ 已有掌握证据'
          : missing.length === 0
            ? '✓ 先修已就绪'
            : `锁定 · ${missing.length} 个先修未掌握`;
        button.append(top, title, summary, meta);
        button.addEventListener('click', () => showNode(node.id));
        nodeButtons.set(node.id, button);
        grid.appendChild(button);
      }

      body.append(statusLine, grid);
      section.append(headerButton, body);
      headerButton.addEventListener('click', () => toggleStage(stage.id));
      stageSections.set(stage.id, { section, headerButton });
      elements.map.appendChild(section);
    }
    syncStageExpansion();
  }

  function renderTopology() {
    if (!window.LearningTopology) {
      elements.topology.textContent = '拓扑模块加载失败，请刷新页面重试。';
      return;
    }
    topologyGraph = window.LearningTopology.createTopologyGraph({
      container: elements.topology,
      nodes: allNodes,
      stages: data.stages,
      statusMeta,
      initialNodeId: selectedNodeId,
      onSelect: (nodeId) => showNode(nodeId),
      onTransform: ({ scale }) => {
        elements.topologyScale.textContent = `${Math.round(scale * 100)}%`;
      }
    });
  }

  function setMapView(view) {
    if (!['topology', 'stages'].includes(view)) return;
    activeMapView = view;
    const showTopology = view === 'topology';
    elements.topologyPanel.hidden = !showTopology;
    elements.stagesPanel.hidden = showTopology;
    elements.toggleStages.hidden = showTopology;
    for (const button of elements.mapViewSwitch.querySelectorAll('[data-map-view]')) {
      const isActive = button.dataset.mapView === view;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', String(isActive));
      button.tabIndex = isActive ? 0 : -1;
    }
    if (showTopology) requestAnimationFrame(() => topologyGraph?.refreshViewport());
  }

  function scheduleTopologyRefresh() {
    requestAnimationFrame(() => requestAnimationFrame(() => topologyGraph?.refreshViewport()));
  }

  function syncFullscreenState() {
    const isNativeFullscreen = document.fullscreenElement === elements.topologyPanel;
    const isFullscreen = isNativeFullscreen || pseudoFullscreen;
    elements.topologyPanel.classList.toggle('is-pseudo-fullscreen', pseudoFullscreen);
    document.body.classList.toggle('topology-fullscreen-open', pseudoFullscreen);
    elements.topologyFullscreen.textContent = isFullscreen ? '退出全屏' : '全屏查看';
    elements.topologyFullscreen.setAttribute('aria-pressed', String(isFullscreen));
    elements.topologyFullscreen.setAttribute('aria-label', isFullscreen ? '退出拓扑图全屏' : '全屏查看拓扑图');
    scheduleTopologyRefresh();
  }

  async function toggleTopologyFullscreen() {
    if (document.fullscreenElement === elements.topologyPanel) {
      try {
        await document.exitFullscreen?.();
      } catch (_error) {
        // 全屏状态仍由 fullscreenchange 与实际 DOM 状态校准。
      }
      syncFullscreenState();
      return;
    }
    if (pseudoFullscreen) {
      pseudoFullscreen = false;
      syncFullscreenState();
      return;
    }
    if (elements.topologyPanel.requestFullscreen) {
      try {
        await elements.topologyPanel.requestFullscreen();
        syncFullscreenState();
        return;
      } catch (_error) {
        // 浏览器拒绝原生全屏时，退化为覆盖整个页面的沉浸视图。
      }
    }
    pseudoFullscreen = true;
    syncFullscreenState();
  }

  function toggleStage(stageId) {
    if (expandedStages.has(stageId)) expandedStages.delete(stageId);
    else expandedStages.add(stageId);
    syncStageExpansion();
  }

  function syncStageExpansion(filtersActive = false) {
    for (const [stageId, { section, headerButton }] of stageSections) {
      const expanded = filtersActive || expandedStages.has(stageId);
      section.classList.toggle('is-collapsed', !expanded);
      headerButton.setAttribute('aria-expanded', String(expanded));
    }
    const allExpanded = expandedStages.size === data.stages.length;
    elements.toggleStages.textContent = allExpanded ? '收起未来阶段' : '展开全部阶段';
  }

  function revealStage(stageId) {
    setMapView('stages');
    expandedStages.add(stageId);
    if (activeStatus !== 'all' || elements.searchInput.value) clearFilters();
    syncStageExpansion();
    const target = stageSections.get(stageId)?.section;
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function locateSelectedNode() {
    const node = nodeIndex.get(selectedNodeId);
    if (!node) return;
    if (activeStatus !== 'all' && node.progress.status !== activeStatus) setStatusFilter('all');
    const query = elements.searchInput.value.trim().toLocaleLowerCase('zh-CN');
    if (query && !nodeButtons.get(node.id)?.dataset.search.includes(query)) {
      elements.searchInput.value = '';
      applyFilters();
    }
    if (activeMapView === 'topology') {
      topologyGraph?.focusNode(node.id);
      return;
    }
    expandedStages.add(node.stageId);
    syncStageExpansion(activeStatus !== 'all' || Boolean(elements.searchInput.value));
    nodeButtons.get(node.id)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    window.setTimeout(() => nodeButtons.get(node.id)?.focus({ preventScroll: true }), 350);
  }

  function setStatusFilter(status) {
    activeStatus = status;
    for (const button of elements.statusFilters.querySelectorAll('[data-status-filter]')) {
      button.setAttribute('aria-pressed', String(button.dataset.statusFilter === status));
    }
    applyFilters();
  }

  function applyFilters() {
    const query = elements.searchInput.value.trim().toLocaleLowerCase('zh-CN');
    const filtersActive = Boolean(query) || activeStatus !== 'all';
    let visible = 0;
    for (const button of nodeButtons.values()) {
      const matchesQuery = !query || button.dataset.search.includes(query);
      const matchesStatus = activeStatus === 'all' || button.dataset.status === activeStatus;
      const show = matchesQuery && matchesStatus;
      button.classList.toggle('is-hidden', !show);
      if (show) visible += 1;
    }
    for (const { section } of stageSections.values()) {
      const hasVisibleNode = [...section.querySelectorAll('.node-button')]
        .some((button) => !button.classList.contains('is-hidden'));
      section.classList.toggle('is-filtered-out', !hasVisibleNode);
    }
    syncStageExpansion(filtersActive);
    elements.visibleCount.textContent = filtersActive
      ? `筛选结果 ${visible} / ${allNodes.length}`
      : `共 ${allNodes.length} 个知识点`;
    elements.filterEmpty.hidden = visible !== 0;
    elements.topologyEmpty.hidden = visible !== 0;
    topologyGraph?.setFilter({ query, status: activeStatus });
    elements.resetFilters.hidden = !filtersActive;
  }

  function clearFilters() {
    elements.searchInput.value = '';
    activeStatus = 'all';
    for (const button of elements.statusFilters.querySelectorAll('[data-status-filter]')) {
      button.setAttribute('aria-pressed', String(button.dataset.statusFilter === 'all'));
    }
    applyFilters();
  }

  function renderRecords() {
    elements.recordsList.replaceChildren();
    if (data.records.length === 0) {
      const empty = document.createElement('li');
      empty.className = 'empty-state';
      empty.textContent = '还没有学习记录。';
      elements.recordsList.appendChild(empty);
      return;
    }
    const records = showAllRecords ? data.records : data.records.slice(0, 6);
    for (const record of records) {
      const item = document.createElement('li');
      item.className = 'record';
      const time = document.createElement('time');
      time.dateTime = record.at;
      time.textContent = formatDate(record.at);
      const markerWrap = document.createElement('span');
      markerWrap.className = 'record-marker-wrap';
      const marker = createStatusDot(record.toStatus);
      marker.classList.add('record-marker');
      markerWrap.appendChild(marker);
      const content = document.createElement('div');
      content.className = 'record-content';
      const meta = document.createElement('div');
      meta.className = 'record-meta';
      const title = document.createElement('button');
      title.type = 'button';
      title.textContent = record.nodeTitle;
      title.addEventListener('click', () => showNode(record.nodeId, { locate: true }));
      const transition = document.createElement('span');
      transition.textContent = `${statusMeta[record.fromStatus]?.shortLabel ?? record.fromStatus} → ${statusMeta[record.toStatus].shortLabel}`;
      transition.className = `record-transition ${record.toStatus}`;
      meta.append(title, transition);
      content.appendChild(meta);
      const detailText = record.evidence || record.note;
      if (detailText) {
        const detail = document.createElement('p');
        detail.textContent = detailText;
        content.appendChild(detail);
      }
      item.append(time, markerWrap, content);
      elements.recordsList.appendChild(item);
    }

    elements.recordsToggle.hidden = data.records.length <= 6;
    elements.recordsToggle.textContent = showAllRecords
      ? '收起记录'
      : `展开全部 ${data.records.length} 条`;
  }

  renderOverview();
  renderStageOverview();
  renderMap();
  renderTopology();
  renderRecords();
  showNode(selectedNodeId);
  applyFilters();
  setMapView(activeMapView);

  elements.searchInput.addEventListener('input', applyFilters);
  elements.statusFilters.addEventListener('click', (event) => {
    const button = event.target.closest('[data-status-filter]');
    if (button) setStatusFilter(button.dataset.statusFilter);
  });
  for (const button of document.querySelectorAll('[data-quick-filter]')) {
    button.addEventListener('click', () => {
      setStatusFilter(button.dataset.quickFilter);
      elements.mapPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  elements.resetFilters.addEventListener('click', clearFilters);
  elements.filterEmpty.querySelector('button').addEventListener('click', clearFilters);
  elements.topologyEmpty.querySelector('button').addEventListener('click', clearFilters);
  elements.topologyFullscreen.addEventListener('click', toggleTopologyFullscreen);
  document.addEventListener('fullscreenchange', syncFullscreenState);
  elements.mapViewSwitch.addEventListener('click', (event) => {
    const button = event.target.closest('[data-map-view]');
    if (button) setMapView(button.dataset.mapView);
  });
  elements.mapViewSwitch.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...elements.mapViewSwitch.querySelectorAll('[data-map-view]')];
    const currentIndex = buttons.indexOf(document.activeElement);
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const nextButton = buttons[(currentIndex + direction + buttons.length) % buttons.length];
    setMapView(nextButton.dataset.mapView);
    nextButton.focus();
  });
  elements.topologyFit.addEventListener('click', () => topologyGraph?.fit());
  elements.topologyCurrent.disabled = !nodeIndex.has(data.progress.currentNodeId);
  elements.topologyCurrent.addEventListener('click', () => {
    const currentId = data.progress.currentNodeId;
    if (!nodeIndex.has(currentId)) return;
    clearFilters();
    showNode(currentId);
    topologyGraph?.focusNode(currentId, { moveFocus: false });
  });
  elements.topologyZoomIn.addEventListener('click', () => topologyGraph?.zoomIn());
  elements.topologyZoomOut.addEventListener('click', () => topologyGraph?.zoomOut());
  elements.toggleStages.addEventListener('click', () => {
    if (expandedStages.size === data.stages.length) {
      expandedStages.clear();
      for (const stage of data.stages) {
        if (stage.nodes.some((node) => node.progress.status !== 'pending')) expandedStages.add(stage.id);
      }
    } else {
      for (const stage of data.stages) expandedStages.add(stage.id);
    }
    syncStageExpansion(Boolean(elements.searchInput.value) || activeStatus !== 'all');
  });
  elements.currentDetailButton.addEventListener('click', () => {
    if (!data.progress.currentNodeId) return;
    showNode(data.progress.currentNodeId);
    document.querySelector('.detail-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  elements.currentStageButton.addEventListener('click', () => {
    const current = nodeIndex.get(data.progress.currentNodeId);
    if (current) revealStage(current.stageId);
  });
  elements.locateNode.addEventListener('click', locateSelectedNode);
  elements.recordsToggle.addEventListener('click', () => {
    showAllRecords = !showAllRecords;
    renderRecords();
  });
  window.addEventListener('hashchange', () => {
    const nodeId = decodeURIComponent(window.location.hash.slice(1));
    if (nodeIndex.has(nodeId) && nodeId !== selectedNodeId) showNode(nodeId);
  });
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    const isTyping = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
    if (event.key === 'Escape' && pseudoFullscreen) {
      event.preventDefault();
      pseudoFullscreen = false;
      syncFullscreenState();
      elements.topologyFullscreen.focus();
      return;
    }
    if (event.key === '/' && !isTyping) {
      event.preventDefault();
      elements.searchInput.focus();
    }
    if (event.key === 'Escape' && document.activeElement === elements.searchInput && elements.searchInput.value) {
      elements.searchInput.value = '';
      applyFilters();
    }
  });
})();
