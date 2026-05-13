const state = {
  config: null,
  scope: "anomaly",
  processingMode: "single",
  activeJobId: null,
  logOffset: 0,
  pollTimer: null,
  pollInFlight: false,
};

const scopeOptions = document.getElementById("scope-options");
const dimensionOptions = document.getElementById("dimension-options");
const form = document.getElementById("analysis-form");
const statusCard = document.getElementById("status-card");
const statusTitle = document.getElementById("status-title");
const statusText = document.getElementById("status-text");
const fillDemoButton = document.getElementById("fill-demo");
const clearLogsButton = document.getElementById("clear-logs");
const submitButton = form.querySelector('button[type="submit"]');
const resultsPanel = document.getElementById("results-panel");
const resultVideo = document.getElementById("result-video");
const finalScoreValue = document.getElementById("final-score-value");
const finalScoreRing = document.getElementById("final-score-ring");
const scoreMeta = document.getElementById("score-meta");
const summaryGrid = document.getElementById("summary-grid");
const dimensionCards = document.getElementById("dimension-cards");
const dimensionTemplate = document.getElementById("dimension-card-template");
const resultPath = document.getElementById("result-path");
const logPath = document.getElementById("log-path");
const artifactRootPath = document.getElementById("artifact-root-path");
const logConsole = document.getElementById("log-console");
const logMeta = document.getElementById("log-meta");
const artifactRibbon = document.getElementById("artifact-ribbon");
const modeOptions = document.getElementById("mode-options");
const singleFields = document.getElementById("single-fields");
const batchFields = document.getElementById("batch-fields");
const batchResultsPanel = document.getElementById("batch-results-panel");
const batchResultDir = document.getElementById("batch-result-dir");
const batchAvgScore = document.getElementById("batch-avg-score");
const batchScoreMeta = document.getElementById("batch-score-meta");
const batchSummaryGrid = document.getElementById("batch-summary-grid");
const batchTableBody = document.getElementById("batch-table-body");
const mllmProviderSelect = document.getElementById("mllm_provider_select");
const mllmModelSelect = document.getElementById("mllm_model_select");
const mllmModelSelectWrap = document.getElementById("mllm_model_select_wrap");
const mllmModelText = document.getElementById("mllm_model_text");
const mllmModelTextWrap = document.getElementById("mllm_model_text_wrap");
const mllmBaseUrl = document.getElementById("mllm_base_url");
const mllmServiceName = document.getElementById("mllm_service_name");
const mllmApiKeyEl = document.querySelector('input[name="mllm_api_key"]');

function setStatus(mode, title, text) {
  statusCard.className = `status-card ${mode}`;
  statusTitle.textContent = title;
  statusText.textContent = text;
}

function resetArtifacts() {
  resultPath.textContent = "-";
  logPath.textContent = "-";
  artifactRootPath.textContent = "-";
  resultPath.title = "";
  logPath.title = "";
  artifactRootPath.title = "";
  artifactRibbon.innerHTML = "";
}

function appendLogLines(lines) {
  if (!lines || lines.length === 0) {
    return;
  }
  const prefix = logConsole.textContent ? "\n" : "";
  logConsole.textContent += `${prefix}${lines.join("\n")}`;
  logConsole.scrollTop = logConsole.scrollHeight;
}

function clearLogConsole() {
  logConsole.textContent = "";
  logMeta.textContent = "日志视图已清空";
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  state.pollInFlight = false;
}

function setSubmitting(isSubmitting) {
  submitButton.disabled = isSubmitting;
  submitButton.textContent = isSubmitting ? "分析进行中..." : "开始分析";
}

function isImmediateResultResponse(data) {
  return Boolean(data) && Array.isArray(data.dimensions) && Object.prototype.hasOwnProperty.call(data, "final_score");
}

function scheduleNextPoll(delay = 800) {
  if (!state.activeJobId) {
    return;
  }
  state.pollTimer = setTimeout(async () => {
    if (state.pollInFlight) {
      scheduleNextPoll(500);
      return;
    }
    state.pollInFlight = true;
    try {
      await runPollingCycle();
      if (state.activeJobId) {
        scheduleNextPoll();
      }
    } catch (error) {
      stopPolling();
      setStatus("error", "轮询失败", error.message || "无法获取任务状态");
      setSubmitting(false);
    } finally {
      state.pollInFlight = false;
    }
  }, delay);
}

function renderScopeOptions() {
  scopeOptions.innerHTML = "";
  state.config.scopes.forEach((scope) => {
    const label = document.createElement("label");
    label.className = "scope-option";
    label.innerHTML = `
      <input type="radio" name="scope" value="${scope.key}" ${scope.key === state.scope ? "checked" : ""}>
      <div>
        <strong>${scope.label}</strong>
        <small>${scope.description}</small>
      </div>
    `;
    label.querySelector("input").addEventListener("change", () => {
      state.scope = scope.key;
      renderDimensionOptions();
    });
    scopeOptions.appendChild(label);
  });
}

function renderDimensionOptions() {
  dimensionOptions.innerHTML = "";
  const scopeConfig = state.config.scopes.find((scope) => scope.key === state.scope);
  const defaultsKey = state.scope === "anomaly" ? "anomaly_types" : "selected_dimensions";
  const selected = new Set(state.config.defaults[defaultsKey]);
  scopeConfig.dimensions.forEach((dimension) => {
    const label = document.createElement("label");
    label.className = "check-option";
    const fieldName = state.scope === "anomaly" ? "anomaly_types" : "selected_dimensions";
    label.innerHTML = `
      <input type="checkbox" name="${fieldName}" value="${dimension.key}" ${selected.has(dimension.key) ? "checked" : ""}>
      <div>
        <strong>${dimension.label}</strong>
        <small>${dimension.description}</small>
      </div>
    `;
    dimensionOptions.appendChild(label);
  });
}

function toggleProcessingMode() {
  const batchMode = state.processingMode === "batch";
  singleFields.classList.toggle("hidden", batchMode);
  batchFields.classList.toggle("hidden", !batchMode);
  if (batchMode) {
    form.elements.video_path.value = "";
    form.elements.video_file.value = "";
  } else {
    form.elements.video_dir.value = "";
    form.elements.file_extensions.value = ".mp4,.avi,.mov,.mkv,.webm";
    form.elements.recursive_scan.checked = false;
  }
}

function bindModeOptions() {
  document.querySelectorAll('input[name="processing_mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.processingMode = radio.value;
      toggleProcessingMode();
    });
  });
  toggleProcessingMode();
}

function populateHuaweiModels() {
  if (!state.config || !state.config.huawei_models) return;
  const models = state.config.huawei_models;
  mllmModelSelect.innerHTML = "";
  models.forEach((model) => {
    const opt = document.createElement("option");
    opt.value = model;
    opt.textContent = model;
    mllmModelSelect.appendChild(opt);
  });
}

function switchMllmModelInput() {
  const provider = mllmProviderSelect.value;
  if (provider === "huawei_custom") {
    mllmModelSelectWrap.classList.remove("hidden");
    mllmModelTextWrap.classList.add("hidden");
  } else {
    mllmModelSelectWrap.classList.add("hidden");
    mllmModelTextWrap.classList.remove("hidden");
  }
}

function getMllmModelValue() {
  if (mllmProviderSelect.value === "huawei_custom") {
    return mllmModelSelect.value;
  }
  return mllmModelText.value;
}

function bindMllmProvider() {
  mllmProviderSelect.addEventListener("change", () => {
    switchMllmModelInput();
    // Auto-fill base_url when switching to huawei_custom
    if (mllmProviderSelect.value === "huawei_custom" && !mllmBaseUrl.value.trim()) {
      mllmBaseUrl.value = "http://aitest-beta.rnd.huawei.com/v1";
      mllmServiceName.value = "simple_client";
    }
  });
  switchMllmModelInput();
}

function fillDemo() {
  if (state.processingMode === "batch") {
    form.elements.video_dir.value = "data/videos";
    form.elements.file_extensions.value = ".mp4,.avi,.mov,.mkv,.webm";
    form.elements.recursive_scan.checked = false;
  } else {
    form.elements.video_path.value = "data/sample.mp4";
  }
  form.elements.device.value = state.config.defaults.device;
  form.elements.au_backend.value = state.config.defaults.au_backend;
  form.elements.au_external_python.value = state.config.defaults.au_external_python;
  form.elements.save_visualizations.checked = Boolean(state.config.defaults.save_visualizations);
  form.elements.visualization_root.value = state.config.defaults.visualization_root;
  form.elements.enable_mllm.checked = Boolean(state.config.defaults.enable_mllm);
  mllmProviderSelect.value = state.config.defaults.mllm_provider;
  mllmBaseUrl.value = state.config.defaults.mllm_base_url || "";
  mllmServiceName.value = state.config.defaults.mllm_service_name || "";
  mllmApiKeyEl.value = state.config.defaults.mllm_api_key || "";
  switchMllmModelInput();
  if (mllmProviderSelect.value === "huawei_custom") {
    if (state.config.defaults.mllm_model) {
      mllmModelSelect.value = state.config.defaults.mllm_model;
    }
  } else {
    mllmModelText.value = state.config.defaults.mllm_model || "";
  }
  form.elements.sample_stride.value = state.config.defaults.sample_stride;
  form.elements.max_frames.value = state.config.defaults.max_frames;
  form.elements.max_side.value = state.config.defaults.max_side;
}

function collectFormData() {
  const payload = new FormData();
  const plainElements = [
    "video_path",
    "device",
    "au_backend",
    "au_external_python",
    "visualization_root",
    "mllm_base_url",
    "mllm_api_key",
    "mllm_service_name",
    "sample_stride",
    "max_frames",
    "max_side",
    "max_workers",
  ];
  plainElements.forEach((name) => {
    const el = form.elements[name];
    if (el && el.value !== "") {
      payload.append(name, el.value);
    }
  });

  // MLLM model: from select or text depending on provider
  const modelValue = getMllmModelValue();
  if (modelValue) {
    payload.append("mllm_model", modelValue);
  }
  // MLLM provider
  if (mllmProviderSelect.value) {
    payload.append("mllm_provider", mllmProviderSelect.value);
  }

  payload.append("scope", state.scope);
  payload.append("parallel", String(form.elements.parallel.checked));
  payload.append("save_visualizations", String(form.elements.save_visualizations.checked));
  payload.append("enable_mllm", String(form.elements.enable_mllm.checked));

  if (state.processingMode === "batch") {
    const videoDir = form.elements.video_dir.value.trim();
    if (videoDir) {
      payload.append("video_dir", videoDir);
    }
    const fileExt = form.elements.file_extensions.value.trim();
    if (fileExt) {
      payload.append("file_extensions", fileExt);
    }
    payload.append("recursive_scan", String(form.elements.recursive_scan.checked));
  }

  const fileInput = form.elements.video_file;
  if (fileInput.files.length > 0) {
    payload.append("video_file", fileInput.files[0]);
  }

  const selectionName = state.scope === "anomaly" ? "anomaly_types" : "selected_dimensions";
  document.querySelectorAll(`input[name="${selectionName}"]:checked`).forEach((checkbox) => {
    payload.append(selectionName, checkbox.value);
  });
  return payload;
}

function renderSummaryItem(label, value) {
  const item = document.createElement("div");
  item.className = "summary-item";
  item.innerHTML = `<span>${label}</span><strong>${value ?? "-"}</strong>`;
  return item;
}

function renderMetric(metric) {
  const item = document.createElement("div");
  item.className = "metric-item";
  let value = metric.value;
  if (typeof value === "number") {
    value = metric.kind === "count" ? String(value) : value.toFixed(3);
  }
  item.innerHTML = `<span>${metric.label}</span><strong>${value ?? "-"}</strong>`;
  return item;
}

function renderEvent(event) {
  const item = document.createElement("div");
  item.className = `event-item ${event.severity || "info"}`;
  item.innerHTML = `<strong>${event.title}</strong><div>${event.detail}</div>`;
  return item;
}

function renderArtifactChip(label, value) {
  const chip = document.createElement("div");
  chip.className = "artifact-chip";
  chip.innerHTML = `<span>${label}</span><code>${value}</code>`;
  return chip;
}

function formatRawOutput(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function renderResults(data) {
  resultsPanel.classList.remove("hidden");
  resultVideo.textContent = `${data.video_name} · ${data.scope === "anomaly" ? "五类异常" : "全量维度"}`;
  const finalScore = typeof data.final_score === "number" ? data.final_score : 0;
  finalScoreValue.textContent = finalScore.toFixed(3);
  finalScoreRing.style.setProperty("--ratio", `${Math.round(finalScore * 360)}deg`);

  resultPath.textContent = data.result_json_path || "-";
  logPath.textContent = data.log_path || "-";
  artifactRootPath.textContent = data.artifact_root || "-";
  resultPath.title = data.result_json_path || "";
  logPath.title = data.log_path || "";
  artifactRootPath.title = data.artifact_root || "";

  artifactRibbon.innerHTML = "";
  if (data.result_json_path) {
    artifactRibbon.appendChild(renderArtifactChip("结果", data.result_json_path));
  }
  if (data.log_path) {
    artifactRibbon.appendChild(renderArtifactChip("日志", data.log_path));
  }
  if (data.artifact_root) {
    artifactRibbon.appendChild(renderArtifactChip("可视化", data.artifact_root));
  }

  scoreMeta.innerHTML = "";
  [
    `耗时 ${data.elapsed_sec.toFixed(2)}s`,
    `设备 ${data.device}`,
    `活跃维度 ${data.active_dimensions.length}/${data.selected_dimensions.length}`,
    `采样步长 ${data.video_processing.sample_stride} / 最大帧 ${data.video_processing.max_frames ?? "不限"}`,
    `并发 ${data.video_processing.parallel ? "开启" : "关闭"} / worker ${data.video_processing.max_workers ?? "auto"}`,
    `AU ${data.video_processing.au_backend} / ${data.video_processing.au_external_python || "-"}`,
    `可视化 ${data.video_processing.save_visualizations ? "开启" : "关闭"} / ${data.video_processing.visualization_root || "-"}`,
    `MLLM ${data.video_processing.enable_mllm ? `${data.video_processing.mllm_provider} / ${data.video_processing.mllm_model}` : "关闭"}`,
  ].forEach((text) => {
    const chip = document.createElement("div");
    chip.className = "meta-chip";
    chip.textContent = text;
    scoreMeta.appendChild(chip);
  });

  summaryGrid.innerHTML = "";
  summaryGrid.appendChild(renderSummaryItem("适用维度", data.summary.applicable_count));
  summaryGrid.appendChild(renderSummaryItem("跳过维度", data.summary.skipped_count));
  summaryGrid.appendChild(renderSummaryItem("最佳维度", data.summary.best_dimension || "-"));
  summaryGrid.appendChild(renderSummaryItem("最弱维度", data.summary.worst_dimension || "-"));
  summaryGrid.appendChild(renderSummaryItem("最佳分数", data.summary.best_score != null ? data.summary.best_score.toFixed(3) : "-"));
  summaryGrid.appendChild(renderSummaryItem("最弱分数", data.summary.worst_score != null ? data.summary.worst_score.toFixed(3) : "-"));
  summaryGrid.appendChild(renderSummaryItem("源视频", data.video_name));
  summaryGrid.appendChild(renderSummaryItem("结果路径", data.result_json_path ? "已生成" : "未写入"));
  summaryGrid.appendChild(renderSummaryItem("可视化产物", Array.isArray(data.artifacts) ? data.artifacts.length : 0));

  dimensionCards.innerHTML = "";
  data.dimensions.forEach((dimension) => {
    const node = dimensionTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".dimension-label").textContent = dimension.label;
    node.querySelector(".dimension-score-text").textContent =
      dimension.applicable && typeof dimension.score === "number"
        ? `${dimension.score.toFixed(3)}`
        : "未适用";
    node.querySelector(".dimension-desc").textContent =
      dimension.applicable
        ? dimension.description
        : `跳过原因: ${dimension.skip_reason || "not applicable"}`;

    const band = node.querySelector(".dimension-band");
    band.className = `dimension-band ${dimension.band}`;
    band.textContent = dimension.band === "na" ? "N/A" : dimension.band;

    const fill = node.querySelector(".progress-fill");
    fill.style.width = `${Math.max(0, Math.min(100, (dimension.score || 0) * 100))}%`;

    const metricsGrid = node.querySelector(".metrics-grid");
    dimension.metrics.forEach((metric) => metricsGrid.appendChild(renderMetric(metric)));

    const highlightsList = node.querySelector(".highlights-list");
    if (dimension.highlights.length === 0) {
      const li = document.createElement("li");
      li.textContent = dimension.applicable ? "未返回额外摘要。" : "该维度未参与本次分析。";
      highlightsList.appendChild(li);
    } else {
      dimension.highlights.forEach((highlight) => {
        const li = document.createElement("li");
        li.textContent = highlight;
        highlightsList.appendChild(li);
      });
    }

    if (dimension.events.length > 0) {
      const eventsBlock = node.querySelector(".events-block");
      eventsBlock.classList.remove("hidden");
      const eventList = node.querySelector(".event-list");
      dimension.events.forEach((event) => eventList.appendChild(renderEvent(event)));
    }

    if (dimension.vlm_raw_output !== null && dimension.vlm_raw_output !== undefined) {
      const rawBlock = node.querySelector(".vlm-raw-block");
      rawBlock.classList.remove("hidden");
      node.querySelector(".raw-output-console").textContent = formatRawOutput(dimension.vlm_raw_output);
    }

    if (Array.isArray(dimension.artifacts) && dimension.artifacts.length > 0) {
      const artifactBlock = node.querySelector(".dimension-artifacts-block");
      artifactBlock.classList.remove("hidden");
      const artifactList = node.querySelector(".dimension-artifact-list");
      dimension.artifacts.forEach((artifact) => {
        artifactList.appendChild(renderArtifactChip(artifact.label, artifact.path));
      });
    }

    dimensionCards.appendChild(node);
  });
}

async function fetchConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("无法加载前端配置");
  }
  state.config = await response.json();
  renderScopeOptions();
  renderDimensionOptions();
  bindModeOptions();
  populateHuaweiModels();
  bindMllmProvider();
  fillDemo();
}

async function pollJobLogs() {
  if (!state.activeJobId) {
    return;
  }
  const response = await fetch(`/api/jobs/${state.activeJobId}/logs?offset=${state.logOffset}`);
  if (!response.ok) {
    throw new Error("日志读取失败");
  }
  const data = await response.json();
  appendLogLines(data.lines);
  state.logOffset = data.next_offset;
  logMeta.textContent = data.completed
    ? `日志已完成，共 ${data.next_offset} 行`
    : `实时更新中，已接收 ${data.next_offset} 行`;
}

async function pollJobStatus() {
  if (!state.activeJobId) {
    return;
  }
  const response = await fetch(`/api/jobs/${state.activeJobId}`);
  if (!response.ok) {
    throw new Error("状态读取失败");
  }
  const data = await response.json();
  if (data.result_json_path) {
    resultPath.textContent = data.result_json_path;
    resultPath.title = data.result_json_path;
  }
  if (data.log_path) {
    logPath.textContent = data.log_path;
    logPath.title = data.log_path;
  }
  if (data.artifact_root) {
    artifactRootPath.textContent = data.artifact_root;
    artifactRootPath.title = data.artifact_root;
  }

  // Show batch progress during running
  if (data.status === "running" && data.result && data.result.batch_progress) {
    renderBatchProgress(
      data.result.batch_progress.current,
      data.result.batch_progress.total,
      data.result.batch_progress.current_video,
    );
  }

  if (data.status === "completed" && data.result) {
    stopPolling();
    await pollJobLogs();
    state.activeJobId = null;
    if (data.result.batch) {
      renderBatchResults(data.result);
      setStatus("success", "批量分析完成", `已完成 ${data.result.total_videos} 个视频的评测，平均分 ${(data.result.aggregate.avg_score || 0).toFixed(3)}。`);
    } else {
      renderResults(data.result);
      setStatus("success", "分析完成", `已完成 ${data.result.video_name} 的评测，综合分 ${data.result.final_score.toFixed(3)}。`);
    }
    setSubmitting(false);
  } else if (data.status === "failed") {
    stopPolling();
    await pollJobLogs();
    state.activeJobId = null;
    setStatus("error", "分析失败", data.error || "任务执行失败");
    setSubmitting(false);
  }
}

async function runPollingCycle() {
  await pollJobLogs();
  await pollJobStatus();
}

function startPolling(jobId) {
  stopPolling();
  state.activeJobId = jobId;
  state.logOffset = 0;
  logConsole.textContent = "";
  logMeta.textContent = "任务已提交，等待首批日志...";
  scheduleNextPoll(0);
}

function renderBatchProgress(current, total, currentVideo) {
  const pct = Math.round((current / total) * 100);
  setStatus(
    "running",
    "批量处理中",
    `正在处理 ${current}/${total}: ${currentVideo}`,
  );
  logMeta.textContent = `批量进度: ${current}/${total} (${pct}%)`;
}

function renderDistribution(data) {
  const dist = { excellent: 0, good: 0, warning: 0, critical: 0 };
  data.video_results.forEach((r) => {
    if (r.status !== "completed" || r.final_score == null) return;
    const s = r.final_score;
    if (s >= 0.85) dist.excellent++;
    else if (s >= 0.7) dist.good++;
    else if (s >= 0.5) dist.warning++;
    else dist.critical++;
  });
  const total = data.completed_videos || 1;
  const bars = [
    { key: "excellent", label: "优秀 ≥ 0.85", color: "var(--olive)" },
    { key: "good", label: "良好 ≥ 0.70", color: "#2f6e41" },
    { key: "warning", label: "警告 ≥ 0.50", color: "var(--warn)" },
    { key: "critical", label: "严重 < 0.50", color: "var(--danger)" },
  ];
  let html = "";
  bars.forEach((b) => {
    const count = dist[b.key];
    const pct = Math.round((count / total) * 100);
    html += `
      <div class="dist-row">
        <span class="dist-label">${b.label}</span>
        <div class="dist-bar-track">
          <div class="dist-bar-fill" style="width:${pct}%;background:${b.color}"></div>
        </div>
        <span class="dist-count">${count}</span>
      </div>`;
  });
  return html;
}

function renderBatchResults(data) {
  resultsPanel.classList.add("hidden");
  batchResultsPanel.classList.remove("hidden");

  const scopeLabel = data.scope === "anomaly" ? "五类异常" : "全量维度";
  batchResultDir.innerHTML = `<code>${data.video_dir || "-"}</code><span class="batch-scope-tag">${scopeLabel}</span>`;

  const avgScore = typeof data.aggregate.avg_score === "number" ? data.aggregate.avg_score : 0;
  batchAvgScore.textContent = avgScore.toFixed(3);
  document.querySelector(".batch-ring").style.setProperty("--ratio", `${Math.round(avgScore * 360)}deg`);

  batchScoreMeta.innerHTML = "";
  [
    { k: "总数", v: `${data.total_videos} 个` },
    { k: "成功", v: data.completed_videos },
    { k: "失败", v: data.failed_videos },
    { k: "耗时", v: formatBatchElapsed(data.elapsed_sec) },
    { k: "设备", v: data.device },
    { k: "MLLM", v: data.video_processing.enable_mllm ? data.video_processing.mllm_provider : "关闭" },
  ].forEach(({ k, v }) => {
    const chip = document.createElement("div");
    chip.className = "meta-chip";
    chip.innerHTML = `<span>${k}</span><strong>${v}</strong>`;
    batchScoreMeta.appendChild(chip);
  });

  const distEl = document.getElementById("batch-distribution");
  distEl.innerHTML = renderDistribution(data);

  batchSummaryGrid.innerHTML = "";
  [
    { l: "最优视频", v: data.aggregate.best_video || "-" },
    { l: "最优分数", v: data.aggregate.best_score != null ? data.aggregate.best_score.toFixed(3) : "-" },
    { l: "最弱视频", v: data.aggregate.worst_video || "-" },
    { l: "最弱分数", v: data.aggregate.worst_score != null ? data.aggregate.worst_score.toFixed(3) : "-" },
    { l: "平均分", v: avgScore.toFixed(3) },
    { l: "并发", v: data.video_processing.parallel ? `开启 × ${data.video_processing.max_workers || "auto"}` : "关闭" },
    { l: "报告目录", v: data.results_dir || "outputs/" },
  ].forEach(({ l, v }) => {
    const div = document.createElement("div");
    div.className = "summary-item";
    div.innerHTML = `<span>${l}</span><strong>${v}</strong>`;
    batchSummaryGrid.appendChild(div);
  });

  // Sort: completed by score desc, then failed
  const sorted = [...data.video_results].sort((a, b) => {
    if (a.status !== b.status) return a.status === "completed" ? -1 : 1;
    return (b.final_score || 0) - (a.final_score || 0);
  });

  batchTableBody.innerHTML = "";
  sorted.forEach((result, idx) => {
    const tr = document.createElement("tr");
    if (result.status === "failed") tr.classList.add("row-failed");
    const scoreClass = result.status === "completed"
      ? (result.final_score >= 0.85 ? "score-high" : result.final_score >= 0.5 ? "score-mid" : "score-low")
      : "";
    const scoreText = result.status === "completed" ? (result.final_score || 0).toFixed(3) : "—";
    const vlmKeys = result.vlm_outputs ? Object.keys(result.vlm_outputs) : [];
    const vlmCell = vlmKeys.length > 0
      ? `<span class="vlm-chip" title="${vlmKeys.join(', ')}">${vlmKeys.length}</span>`
      : "<span class=\"vlm-none\">—</span>";
    tr.innerHTML = `
      <td class="num">${idx + 1}</td>
      <td class="name" title="${result.video_path || ""}">${result.video_name}</td>
      <td class="score-cell ${scoreClass}">${scoreText}</td>
      <td><span class="status-badge ${result.status}">${result.status === "completed" ? "完成" : "失败"}</span>${result.error ? `<span class="err-tip" title="${result.error.replace(/"/g, '&quot;')}">ⓘ</span>` : ""}</td>
      <td>${vlmCell}</td>
      <td class="right">${formatBatchElapsed(result.elapsed_sec || 0)}</td>
      <td class="right dims">${Array.isArray(result.active_dimensions) ? result.active_dimensions.length : "-"}</td>
    `;
    batchTableBody.appendChild(tr);
  });
}

function formatBatchElapsed(sec) {
  if (!sec || sec <= 0) return "-";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const mins = Math.floor(sec / 60);
  const secs = Math.round(sec % 60);
  return `${mins}m ${secs}s`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();
  state.activeJobId = null;
  setSubmitting(true);
  setStatus("running", "分析中", "后端正在抽帧、执行检测并实时写出日志。");
  resultsPanel.classList.add("hidden");
  batchResultsPanel.classList.add("hidden");
  resetArtifacts();
  clearLogConsole();
  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      body: collectFormData(),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "任务提交失败");
    }
    if (isImmediateResultResponse(data)) {
      renderResults(data);
      setStatus("success", "分析完成", `当前后端返回的是同步结果，已完成 ${data.video_name} 的评测。`);
      logMeta.textContent = "当前后端未启用任务日志流";
      setSubmitting(false);
      return;
    }
    if (!data.job_id) {
      throw new Error("后端未返回 job_id。请重启 WebUI 服务并强制刷新页面后重试。");
    }
    setStatus("running", "任务已创建", `任务 ${data.job_id} 已进入队列，正在等待分析结果。`);
    startPolling(data.job_id);
  } catch (error) {
    setStatus("error", "分析失败", error.message || "请求失败");
    setSubmitting(false);
  }
});

fillDemoButton.addEventListener("click", fillDemo);
clearLogsButton.addEventListener("click", clearLogConsole);

fetchConfig()
  .then(() => setStatus("idle", "系统就绪", "可以直接上传视频或填写本地路径开始分析。"))
  .catch((error) => setStatus("error", "初始化失败", error.message || "配置加载失败"));
