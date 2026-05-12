const state = {
  config: null,
  scope: "anomaly",
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

function fillDemo() {
  form.elements.video_path.value = "data/sample.mp4";
  form.elements.device.value = state.config.defaults.device;
  form.elements.au_backend.value = state.config.defaults.au_backend;
  form.elements.au_external_python.value = state.config.defaults.au_external_python;
  form.elements.save_visualizations.checked = Boolean(state.config.defaults.save_visualizations);
  form.elements.visualization_root.value = state.config.defaults.visualization_root;
  form.elements.enable_mllm.checked = Boolean(state.config.defaults.enable_mllm);
  form.elements.mllm_provider.value = state.config.defaults.mllm_provider;
  form.elements.mllm_model.value = state.config.defaults.mllm_model;
  form.elements.mllm_base_url.value = state.config.defaults.mllm_base_url;
  form.elements.mllm_api_key.value = state.config.defaults.mllm_api_key;
  form.elements.mllm_service_name.value = state.config.defaults.mllm_service_name;
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
    "mllm_provider",
    "mllm_model",
    "mllm_base_url",
    "mllm_api_key",
    "mllm_service_name",
    "sample_stride",
    "max_frames",
    "max_side",
    "max_workers",
  ];
  plainElements.forEach((name) => {
    const value = form.elements[name].value;
    if (value !== "") {
      payload.append(name, value);
    }
  });

  payload.append("scope", state.scope);
  payload.append("parallel", String(form.elements.parallel.checked));
  payload.append("save_visualizations", String(form.elements.save_visualizations.checked));
  payload.append("enable_mllm", String(form.elements.enable_mllm.checked));

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

  if (data.status === "completed" && data.result) {
    stopPolling();
    await pollJobLogs();
    state.activeJobId = null;
    renderResults(data.result);
    setStatus("success", "分析完成", `已完成 ${data.result.video_name} 的评测，综合分 ${data.result.final_score.toFixed(3)}。`);
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();
  state.activeJobId = null;
  setSubmitting(true);
  setStatus("running", "分析中", "后端正在抽帧、执行检测并实时写出日志。");
  resultsPanel.classList.add("hidden");
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
