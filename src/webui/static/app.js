const state = {
  config: null,
  scope: "anomaly",
};

const scopeOptions = document.getElementById("scope-options");
const dimensionOptions = document.getElementById("dimension-options");
const form = document.getElementById("analysis-form");
const statusCard = document.getElementById("status-card");
const statusTitle = document.getElementById("status-title");
const statusText = document.getElementById("status-text");
const fillDemoButton = document.getElementById("fill-demo");
const resultsPanel = document.getElementById("results-panel");
const resultVideo = document.getElementById("result-video");
const finalScoreValue = document.getElementById("final-score-value");
const finalScoreRing = document.getElementById("final-score-ring");
const scoreMeta = document.getElementById("score-meta");
const summaryGrid = document.getElementById("summary-grid");
const dimensionCards = document.getElementById("dimension-cards");
const dimensionTemplate = document.getElementById("dimension-card-template");

function setStatus(mode, title, text) {
  statusCard.className = `status-card ${mode}`;
  statusTitle.textContent = title;
  statusText.textContent = text;
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
  form.elements.sample_stride.value = state.config.defaults.sample_stride;
  form.elements.max_frames.value = state.config.defaults.max_frames;
  form.elements.max_side.value = state.config.defaults.max_side;
}

function collectFormData() {
  const payload = new FormData();
  const plainElements = ["video_path", "device", "sample_stride", "max_frames", "max_side", "max_workers"];
  plainElements.forEach((name) => {
    const value = form.elements[name].value;
    if (value !== "") {
      payload.append(name, value);
    }
  });

  payload.append("scope", state.scope);
  payload.append("parallel", String(form.elements.parallel.checked));
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

function renderResults(data) {
  resultsPanel.classList.remove("hidden");
  resultVideo.textContent = `${data.video_name} · ${data.scope === "anomaly" ? "五类异常" : "全量维度"}`;
  const finalScore = typeof data.final_score === "number" ? data.final_score : 0;
  finalScoreValue.textContent = finalScore.toFixed(3);
  finalScoreRing.style.setProperty("--ratio", `${Math.round(finalScore * 360)}deg`);

  scoreMeta.innerHTML = "";
  [
    `耗时 ${data.elapsed_sec.toFixed(2)}s`,
    `设备 ${data.device}`,
    `活跃维度 ${data.active_dimensions.length}/${data.selected_dimensions.length}`,
    `采样步长 ${data.video_processing.sample_stride} / 最大帧 ${data.video_processing.max_frames ?? "不限"}`,
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("running", "分析中", "后端正在抽帧、执行检测并整理可视化结果。");
  resultsPanel.classList.add("hidden");
  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      body: collectFormData(),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "分析失败");
    }
    renderResults(data);
    setStatus("success", "分析完成", `已完成 ${data.video_name} 的评测，综合分 ${data.final_score.toFixed(3)}。`);
  } catch (error) {
    setStatus("error", "分析失败", error.message || "请求失败");
  }
});

fillDemoButton.addEventListener("click", fillDemo);

fetchConfig()
  .then(() => setStatus("idle", "系统就绪", "可以直接上传视频或填写本地路径开始分析。"))
  .catch((error) => setStatus("error", "初始化失败", error.message || "配置加载失败"));
