(() => {
  const data = window.LEARNING_DATA;
  if (!data) {
    document.body.textContent = '学习数据未生成，请先运行 npm run learning:build';
    return;
  }

  const statusMeta = {
    mastered: { label: '已掌握' },
    current: { label: '当前学习' },
    verify: { label: '待验证' },
    relearn: { label: '待系统重学' },
    pending: { label: '未开始' }
  };

  const allNodes = data.stages.flatMap((stage) =>
    stage.nodes.map((node) => ({ ...node, stageId: stage.id, stageTitle: stage.title }))
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
    updatedAt: document.getElementById('updated-at'),
    masteryCount: document.getElementById('mastery-count'),
    progressFill: document.getElementById('progress-fill'),
    progressTrack: document.querySelector('.progress-track'),
    currentNode: document.getElementById('current-node'),
    currentStage: document.getElementById('current-stage'),
    reviewCount: document.getElementById('review-count'),
    searchInput: document.getElementById('search-input'),
    statusFilter: document.getElementById('status-filter'),
    legend: document.getElementById('legend'),
    map: document.getElementById('learning-map'),
    visibleCount: document.getElementById('visible-count'),
    detailTitle: document.getElementById('detail-title'),
    detailDot: document.getElementById('detail-dot'),
    detailStatus: document.getElementById('detail-status'),
    detailSummary: document.getElementById('detail-summary'),
    detailPrerequisites: document.getElementById('detail-prerequisites'),
    detailDependents: document.getElementById('detail-dependents'),
    detailCriteria: document.getElementById('detail-criteria'),
    detailEvidence: document.getElementById('detail-evidence'),
    recordsList: document.getElementById('records-list')
  };

  let selectedNodeId = data.progress.currentNodeId || allNodes[0]?.id;
  const nodeButtons = new Map();

  function formatDate(value) {
    if (!value) return '尚未更新';
    return new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hour12: false
    }).format(new Date(value));
  }

  function makeStatusDot(status) {
    const dot = document.createElement('span');
    dot.className = `status-dot ${status}`;
    dot.setAttribute('aria-hidden', 'true');
    return dot;
  }

  function showNode(nodeId) {
    const node = nodeIndex.get(nodeId);
    if (!node) return;
    selectedNodeId = nodeId;
    for (const [id, button] of nodeButtons) {
      button.setAttribute('aria-pressed', String(id === nodeId));
    }

    const status = node.progress.status;
    elements.detailTitle.textContent = node.title;
    elements.detailDot.className = `status-dot ${status}`;
    elements.detailStatus.textContent = `${statusMeta[status].label} · ${node.stageTitle}`;
    elements.detailStatus.className = `status-label ${status}`;
    elements.detailSummary.textContent = node.summary;

    renderDependencies(elements.detailPrerequisites, node.prerequisites, '无直接先修知识点');
    renderDependencies(elements.detailDependents, dependents.get(node.id) ?? [], '暂无直接后续知识点');

    elements.detailCriteria.replaceChildren();
    for (const criterion of node.masteryCriteria) {
      const item = document.createElement('li');
      item.textContent = criterion;
      elements.detailCriteria.appendChild(item);
    }

    elements.detailEvidence.replaceChildren();
    const evidence = node.progress.evidence ?? [];
    if (evidence.length === 0) {
      const empty = document.createElement('span');
      empty.className = 'empty-state';
      empty.textContent = '尚无可核验的掌握证据';
      elements.detailEvidence.appendChild(empty);
    } else {
      for (const entry of [...evidence].reverse()) {
        const item = document.createElement('div');
        item.className = 'evidence-item';
        const text = document.createElement('div');
        text.textContent = entry.text;
        const time = document.createElement('time');
        time.dateTime = entry.at;
        time.textContent = formatDate(entry.at);
        item.append(text, time);
        elements.detailEvidence.appendChild(item);
      }
    }
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
      button.textContent = `${dependency.title} · ${statusMeta[dependency.progress.status].label}`;
      button.addEventListener('click', () => showNode(id));
      container.appendChild(button);
    }
  }

  function renderOverview() {
    const counts = data.progress.statusCounts;
    const percentage = data.progress.totalNodes === 0
      ? 0
      : Math.round((counts.mastered / data.progress.totalNodes) * 100);
    elements.goalDescription.textContent = data.goal.description;
    elements.updatedAt.textContent = `数据更新：${formatDate(data.progress.updatedAt || data.generatedAt)}`;
    elements.masteryCount.textContent = `${counts.mastered} / ${data.progress.totalNodes}`;
    elements.progressFill.style.width = `${percentage}%`;
    elements.progressTrack.setAttribute('aria-valuenow', String(percentage));
    elements.reviewCount.textContent = String(counts.verify + counts.relearn);

    const current = nodeIndex.get(data.progress.currentNodeId);
    elements.currentNode.textContent = current?.title ?? '尚未指定';
    elements.currentStage.textContent = current?.stageTitle ?? '通过上报脚本设置当前入口';

    for (const [status, meta] of Object.entries(statusMeta)) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      item.append(makeStatusDot(status), `${meta.label} ${counts[status]}`);
      elements.legend.appendChild(item);
    }
  }

  function renderMap() {
    for (const stage of data.stages) {
      const section = document.createElement('section');
      section.className = 'stage';
      section.dataset.stageId = stage.id;

      const masteredInStage = stage.nodes.filter((node) => node.progress.status === 'mastered').length;
      const header = document.createElement('div');
      header.className = 'stage-header';
      const title = document.createElement('h3');
      title.textContent = stage.title;
      const count = document.createElement('span');
      count.textContent = `${masteredInStage}/${stage.nodes.length} 已掌握`;
      header.append(title, count);

      const description = document.createElement('p');
      description.className = 'stage-description';
      description.textContent = stage.description;

      const grid = document.createElement('div');
      grid.className = 'node-grid';
      for (const node of stage.nodes) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'node-button';
        button.dataset.nodeId = node.id;
        button.dataset.status = node.progress.status;
        button.dataset.search = `${node.title} ${node.summary}`.toLocaleLowerCase('zh-CN');
        button.setAttribute('aria-pressed', String(node.id === selectedNodeId));
        button.appendChild(makeStatusDot(node.progress.status));

        const content = document.createElement('span');
        const nodeTitle = document.createElement('span');
        nodeTitle.className = 'node-title';
        nodeTitle.textContent = node.title;
        const missing = node.prerequisites.filter(
          (id) => nodeIndex.get(id)?.progress.status !== 'mastered'
        ).length;
        const meta = document.createElement('span');
        meta.className = 'node-meta';
        meta.textContent = node.progress.status === 'mastered'
          ? '掌握标准已通过'
          : missing === 0
            ? '先修条件已具备'
            : `${missing} 个先修尚未掌握`;
        content.append(nodeTitle, meta);
        button.appendChild(content);
        button.addEventListener('click', () => showNode(node.id));
        nodeButtons.set(node.id, button);
        grid.appendChild(button);
      }

      section.append(header, description, grid);
      elements.map.appendChild(section);
    }
  }

  function applyFilters() {
    const query = elements.searchInput.value.trim().toLocaleLowerCase('zh-CN');
    const status = elements.statusFilter.value;
    let visible = 0;
    for (const button of nodeButtons.values()) {
      const matchesQuery = !query || button.dataset.search.includes(query);
      const matchesStatus = status === 'all' || button.dataset.status === status;
      const show = matchesQuery && matchesStatus;
      button.classList.toggle('is-hidden', !show);
      if (show) visible += 1;
    }
    for (const section of elements.map.querySelectorAll('.stage')) {
      const hasVisibleNode = [...section.querySelectorAll('.node-button')]
        .some((button) => !button.classList.contains('is-hidden'));
      section.classList.toggle('is-hidden', !hasVisibleNode);
    }
    elements.visibleCount.textContent = `显示 ${visible} / ${allNodes.length}`;
  }

  function renderRecords() {
    elements.recordsList.replaceChildren();
    if (data.records.length === 0) {
      const empty = document.createElement('li');
      empty.className = 'empty-state';
      empty.textContent = '还没有学习记录';
      elements.recordsList.appendChild(empty);
      return;
    }
    for (const record of data.records) {
      const item = document.createElement('li');
      item.className = 'record';
      const time = document.createElement('time');
      time.dateTime = record.at;
      time.textContent = formatDate(record.at);
      const marker = makeStatusDot(record.toStatus);
      marker.classList.add('record-marker');
      const content = document.createElement('div');
      content.className = 'record-content';
      const title = document.createElement('strong');
      title.textContent = `${record.nodeTitle}：${statusMeta[record.fromStatus]?.label ?? record.fromStatus} → ${statusMeta[record.toStatus].label}`;
      content.appendChild(title);
      const detailText = record.evidence || record.note;
      if (detailText) {
        const detail = document.createElement('p');
        detail.textContent = detailText;
        content.appendChild(detail);
      }
      item.append(time, marker, content);
      elements.recordsList.appendChild(item);
    }
  }

  renderOverview();
  renderMap();
  renderRecords();
  showNode(selectedNodeId);
  applyFilters();

  elements.searchInput.addEventListener('input', applyFilters);
  elements.statusFilter.addEventListener('change', applyFilters);
})();
