const AGENT_COLORS = ["#55a7ff", "#ff6b6b", "#f4c430", "#59d98e", "#c77dff"];

const form = document.getElementById("configForm");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusPill = document.getElementById("statusPill");
const frameLabel = document.getElementById("frameLabel");
const eventLog = document.getElementById("eventLog");
const deepseekBtn = document.getElementById("deepseekBtn");
const summaryHeadline = document.getElementById("summaryHeadline");
const diagnosisList = document.getElementById("diagnosisList");
const recommendationList = document.getElementById("recommendationList");
const reportText = document.getElementById("reportText");
const resultTableBody = document.querySelector("#resultTable tbody");
const simCanvas = document.getElementById("simCanvas");
const chartCanvas = document.getElementById("chartCanvas");
const simCtx = simCanvas.getContext("2d");
const chartCtx = chartCanvas.getContext("2d");

const state = {
  status: "idle",
  config: {},
  currentEpisode: 0,
  totalEpisodes: 0,
  coverageRadius: 0.24,
  safeDistance: 0.2,
  worldSize: 2,
  landmarks: [],
  latestFrame: null,
  previousFrame: null,
  frameQueue: [],
  trails: [],
  history: [],
  report: null,
  llmReport: null,
  currentRun: null,
  lastFrameAt: performance.now(),
  step: 0,
};

for (const input of document.querySelectorAll("input[type='range']")) {
  const output = document.getElementById(input.dataset.output);
  const sync = () => (output.textContent = Number(input.value).toFixed(2));
  input.addEventListener("input", sync);
  sync();
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function parseConfig() {
  const data = new FormData(form);
  return {
    experiment_mode: data.get("experiment_mode"),
    algo: data.get("algo"),
    episodes: Number(data.get("episodes")),
    num_agents: Number(data.get("num_agents")),
    actor_lr: Number(data.get("actor_lr")),
    critic_lr: Number(data.get("critic_lr")),
    gamma: Number(data.get("gamma")),
    batch_size: Number(data.get("batch_size")),
    buffer_warmup: Number(data.get("buffer_warmup")),
    seed: Number(data.get("seed")),
    coverage_ratio: Number(data.get("coverage_ratio")),
    safe_ratio: Number(data.get("safe_ratio")),
    frame_stride: Number(data.get("frame_stride")),
    use_weight_scheduling: data.get("use_weight_scheduling") === "on",
    use_wandb: data.get("use_wandb") === "on",
    wandb_project: data.get("wandb_project"),
    wandb_run_name: data.get("wandb_run_name"),
    eval_interval: Number(data.get("eval_interval")),
    eval_episodes: Number(data.get("eval_episodes")),
    noise_final_scale: Number(data.get("noise_final_scale")),
  };
}

async function postJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    startBtn.disabled = true;
    const snapshot = await postJson("/api/train/start", parseConfig());
    applyState(snapshot);
    logEvent("训练已启动");
  } catch (error) {
    logEvent(`启动失败：${error.message}`);
  } finally {
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/train/stop");
    logEvent("停止请求已发送");
  } catch (error) {
    logEvent(`停止失败：${error.message}`);
  }
});

deepseekBtn.addEventListener("click", async () => {
  try {
    deepseekBtn.disabled = true;
    deepseekBtn.textContent = "生成中...";
    const result = await postJson("/api/report/deepseek");
    state.llmReport = result;
    renderLlmReport(result);
    logEvent("DeepSeek 报告已生成");
  } catch (error) {
    logEvent(`DeepSeek 生成失败：${error.message}`);
  } finally {
    deepseekBtn.disabled = false;
    deepseekBtn.textContent = "DeepSeek 生成报告";
  }
});

function applyState(snapshot) {
  state.status = snapshot.status || state.status;
  state.config = snapshot.config || state.config;
  state.currentEpisode = snapshot.episode || state.currentEpisode;
  state.totalEpisodes = state.config.episodes || state.totalEpisodes;
  state.history = snapshot.history || state.history;
  state.currentRun = snapshot.current_run || state.currentRun;
  state.report = snapshot.report || state.report;
  state.llmReport = snapshot.llm_report || state.llmReport;
  updateStatus();
  if (snapshot.metrics) updateKpis(snapshot.metrics);
  if (state.report) renderReport(state.report);
  if (state.llmReport) renderLlmReport(state.llmReport);
  drawChart();
}

function updateStatus() {
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.status}`;
}

function updateKpis(metrics) {
  const isEval = metrics.metric_source === "eval" || metrics.eval_coverage_rate !== undefined;
  const coverage = isEval ? metrics.eval_coverage_rate ?? metrics.coverage_rate : metrics.coverage_rate;
  const collision = isEval ? metrics.eval_collision_count ?? metrics.collision_count : metrics.collision_count;
  const reward = isEval ? metrics.eval_avg_reward ?? metrics.avg_reward : metrics.avg_reward;
  const steps = isEval ? metrics.eval_completion_steps ?? metrics.completion_steps : metrics.completion_steps;
  document.getElementById("episodeKpi").textContent = `${metrics.episode || state.currentEpisode} / ${state.totalEpisodes || 0}`;
  document.getElementById("coverageKpiLabel").textContent = isEval ? "评估覆盖率" : "训练覆盖率";
  document.getElementById("coverageKpi").textContent = `${Math.round((coverage || 0) * 100)}%`;
  document.getElementById("collisionKpi").textContent = Number(collision || 0).toFixed(0);
  document.getElementById("rewardKpi").textContent = Number(reward || 0).toFixed(2);
  document.getElementById("stepsKpi").textContent = Number(steps || 0).toFixed(0);
}

function logEvent(message) {
  const row = document.createElement("div");
  const time = new Date().toLocaleTimeString();
  row.textContent = `${time}  ${message}`;
  eventLog.prepend(row);
  while (eventLog.children.length > 80) eventLog.removeChild(eventLog.lastChild);
}

function connectEvents() {
  const source = new EventSource("/api/events");

  source.addEventListener("state", (event) => applyState(JSON.parse(event.data)));
  source.addEventListener("status", (event) => applyState(JSON.parse(event.data)));
  source.addEventListener("run_start", (event) => {
    const data = JSON.parse(event.data);
    applyState(data);
    if (data.current_run) {
      logEvent(`开始 ${data.current_run.index}/${data.current_run.total}: ${data.current_run.label}`);
    }
  });
  source.addEventListener("run_end", (event) => {
    const data = JSON.parse(event.data);
    if (data.run) logEvent(`${data.run.label} 已完成`);
  });
  source.addEventListener("complete", (event) => {
    applyState(JSON.parse(event.data));
    logEvent("训练结束");
  });
  source.addEventListener("error", (event) => {
    if (event.data) {
      applyState(JSON.parse(event.data));
      logEvent("训练异常");
    }
  });
  source.addEventListener("episode_start", (event) => {
    const data = JSON.parse(event.data);
    state.currentEpisode = data.episode;
    state.totalEpisodes = data.total_episodes;
    state.coverageRadius = data.coverage_radius;
    state.safeDistance = data.safe_distance;
    state.worldSize = data.world_size || 2;
    state.landmarks = data.landmark_positions || [];
    state.frameQueue = [];
    state.trails = [];
    state.latestFrame = null;
    state.previousFrame = null;
    frameLabel.textContent = `Episode ${data.episode} 开始`;
    logEvent(`Episode ${data.episode} 开始`);
  });
  source.addEventListener("frame", (event) => {
    const frame = JSON.parse(event.data);
    state.frameQueue.push(frame);
    if (state.frameQueue.length > 240) state.frameQueue.splice(0, state.frameQueue.length - 240);
  });
  source.addEventListener("episode_end", (event) => {
    const data = JSON.parse(event.data);
    state.currentEpisode = data.episode;
    state.totalEpisodes = data.total_episodes;
    state.history.push(data.metrics);
    if (state.history.length > 1000) state.history.shift();
    updateKpis(data.metrics);
    drawChart();
    const sourceLabel = data.metrics.metric_source === "eval" ? "评估覆盖率" : "训练覆盖率";
    logEvent(`Episode ${data.episode} 完成，${sourceLabel} ${Math.round((data.metrics.coverage_rate || 0) * 100)}%`);
  });
  source.addEventListener("best", (event) => {
    const data = JSON.parse(event.data);
    const score = Number(data.metrics?.score || 0).toFixed(1);
    const coverage = Math.round(Number(data.metrics?.coverage_rate || 0) * 100);
    logEvent(`刷新最佳模型：Episode ${data.metrics?.episode || "-"}，评分 ${score}，覆盖率 ${coverage}%`);
  });
  source.addEventListener("report", (event) => {
    const data = JSON.parse(event.data);
    state.report = data;
    renderReport(data);
    logEvent("结构化结果总结已生成");
  });
  source.addEventListener("llm_report", (event) => {
    const data = JSON.parse(event.data);
    state.llmReport = data;
    renderLlmReport(data);
    logEvent("DeepSeek 报告已返回");
  });
  source.addEventListener("wandb", (event) => {
    const data = JSON.parse(event.data);
    if (data.status === "active") {
      logEvent(`W&B 已连接：${data.project} / ${data.run_name}`);
      if (data.url) logEvent(`W&B 链接：${data.url}`);
    } else {
      logEvent(`W&B 启动失败：${data.message || "unknown error"}`);
    }
  });
}

function renderReport(report) {
  summaryHeadline.textContent = report.headline || "已生成实验总结";

  diagnosisList.innerHTML = "";
  for (const item of report.diagnosis || []) {
    const pill = document.createElement("span");
    pill.textContent = item;
    diagnosisList.appendChild(pill);
  }

  recommendationList.innerHTML = "";
  for (const item of report.recommendations || []) {
    const row = document.createElement("div");
    row.textContent = item;
    recommendationList.appendChild(row);
  }

  resultTableBody.innerHTML = "";
  for (const run of report.runs || []) {
    const last = run.last_window || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(run.label || "")}</td>
      <td>${escapeHtml(run.algo || "")}</td>
      <td>${Number(run.score || 0).toFixed(1)}</td>
      <td>${formatPercent(last.coverage_rate?.mean)}</td>
      <td>${formatNumber(last.collision_count?.mean)}</td>
      <td>${formatPercent(last.redundancy_rate?.mean)}</td>
      <td>${formatNumber(last.completion_steps?.mean)}</td>
    `;
    resultTableBody.appendChild(row);
  }

  reportText.textContent = report.report_text || "暂无报告文本。";
}

function renderLlmReport(result) {
  if (result?.text) reportText.textContent = result.text;
}

function formatPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function worldToCanvas(pos, canvas) {
  const margin = 42 * (window.devicePixelRatio || 1);
  const width = canvas.width - margin * 2;
  const height = canvas.height - margin * 2;
  const x = margin + ((pos[0] + 1.2) / 2.4) * width;
  const y = margin + (1 - (pos[1] + 1.2) / 2.4) * height;
  return [x, y];
}

function radiusToCanvas(r, canvas) {
  const margin = 42 * (window.devicePixelRatio || 1);
  return (r / 2.4) * Math.min(canvas.width - margin * 2, canvas.height - margin * 2);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function interpolatePositions(prev, next, alpha) {
  if (!prev || !next) return next ? next.agent_positions : [];
  return next.agent_positions.map((pos, i) => [
    lerp(prev.agent_positions[i]?.[0] ?? pos[0], pos[0], alpha),
    lerp(prev.agent_positions[i]?.[1] ?? pos[1], pos[1], alpha),
  ]);
}

function drawSimulation(timestamp) {
  resizeCanvas(simCanvas);
  const ctx = simCtx;
  const w = simCanvas.width;
  const h = simCanvas.height;

  if (state.frameQueue.length) {
    state.previousFrame = state.latestFrame || state.frameQueue[0];
    state.latestFrame = state.frameQueue.shift();
    state.lastFrameAt = timestamp;
    state.step = state.latestFrame.step || state.step;
    const positions = state.latestFrame.agent_positions || [];
    positions.forEach((pos, i) => {
      state.trails[i] = state.trails[i] || [];
      state.trails[i].push(pos);
      if (state.trails[i].length > 90) state.trails[i].shift();
    });
  }

  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, "#081018");
  bg.addColorStop(1, "#0f1419");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  drawGrid(ctx, w, h);

  const alpha = Math.min((timestamp - state.lastFrameAt) / 90, 1);
  const positions = interpolatePositions(state.previousFrame, state.latestFrame, alpha);

  drawLandmarks(ctx);
  drawTrails(ctx);
  drawAgents(ctx, positions);

  frameLabel.textContent = state.latestFrame
    ? `Episode ${state.latestFrame.episode} / Step ${state.step}`
    : "等待训练数据";

  requestAnimationFrame(drawSimulation);
}

function drawGrid(ctx, w, h) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth = 1;
  const step = Math.max(42, Math.min(w, h) / 10);
  for (let x = 0; x <= w; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y <= h; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawLandmarks(ctx) {
  const coveragePx = radiusToCanvas(state.coverageRadius, simCanvas);
  for (const landmark of state.landmarks) {
    const [x, y] = worldToCanvas(landmark, simCanvas);
    ctx.save();
    ctx.strokeStyle = "rgba(89,217,142,0.55)";
    ctx.setLineDash([8, 7]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, coveragePx, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f4c430";
    ctx.beginPath();
    ctx.moveTo(x, y - 10);
    ctx.lineTo(x + 10, y);
    ctx.lineTo(x, y + 10);
    ctx.lineTo(x - 10, y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

function drawTrails(ctx) {
  state.trails.forEach((trail, i) => {
    if (!trail || trail.length < 2) return;
    ctx.save();
    for (let k = 1; k < trail.length; k += 1) {
      const [x1, y1] = worldToCanvas(trail[k - 1], simCanvas);
      const [x2, y2] = worldToCanvas(trail[k], simCanvas);
      const alpha = k / trail.length;
      ctx.strokeStyle = hexToRgba(AGENT_COLORS[i % AGENT_COLORS.length], alpha * 0.65);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawAgents(ctx, positions) {
  const safePx = radiusToCanvas(state.safeDistance, simCanvas);
  positions.forEach((pos, i) => {
    const [x, y] = worldToCanvas(pos, simCanvas);
    const color = AGENT_COLORS[i % AGENT_COLORS.length];
    ctx.save();
    ctx.strokeStyle = hexToRgba(color, 0.35);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, safePx, 0, Math.PI * 2);
    ctx.stroke();

    ctx.shadowColor = color;
    ctx.shadowBlur = 16;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 10 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#eef4f8";
    ctx.font = `${12 * (window.devicePixelRatio || 1)}px system-ui`;
    ctx.fillText(`UAV-${i + 1}`, x + 13, y - 12);
    ctx.restore();
  });
}

function drawChart() {
  resizeCanvas(chartCanvas);
  const ctx = chartCtx;
  const w = chartCanvas.width;
  const h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#070b10";
  ctx.fillRect(0, 0, w, h);

  const data = state.history.slice(-120);
  if (!data.length) {
    ctx.fillStyle = "#8fa0ad";
    ctx.font = `${14 * (window.devicePixelRatio || 1)}px system-ui`;
    ctx.fillText("等待 episode 指标", 24, 38);
    return;
  }

  const pad = 34 * (window.devicePixelRatio || 1);
  drawChartLine(ctx, data.map((d) => Number(d.train_coverage_rate ?? d.coverage_rate ?? 0) * 100), "#55a7ff", pad, w, h, 0, 100);
  const evalData = data.map((d) => d.eval_coverage_rate === undefined ? null : Number(d.eval_coverage_rate) * 100);
  drawSparseChartLine(ctx, evalData, "#2ec4b6", pad, w, h, 0, 100);
  const rewards = data.map((d) => d.avg_reward || 0);
  const minReward = Math.min(...rewards);
  const maxReward = Math.max(...rewards);
  drawChartLine(ctx, rewards, "#f4c430", pad, w, h, minReward, maxReward);

  ctx.fillStyle = "#8fa0ad";
  ctx.font = `${11 * (window.devicePixelRatio || 1)}px system-ui`;
  ctx.fillText("训练覆盖", pad, 18 * (window.devicePixelRatio || 1));
  ctx.fillStyle = "#2ec4b6";
  ctx.fillText("评估覆盖", pad + 72 * (window.devicePixelRatio || 1), 18 * (window.devicePixelRatio || 1));
  ctx.fillStyle = "#f4c430";
  ctx.fillText("奖励", pad + 146 * (window.devicePixelRatio || 1), 18 * (window.devicePixelRatio || 1));
}

function drawChartLine(ctx, values, color, pad, w, h, min, max) {
  const span = Math.max(max - min, 1e-6);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad + (i / Math.max(values.length - 1, 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawSparseChartLine(ctx, values, color, pad, w, h, min, max) {
  const span = Math.max(max - min, 1e-6);
  const points = [];
  values.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) return;
    const x = pad + (i / Math.max(values.length - 1, 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    points.push([x, y]);
  });
  if (!points.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.8 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = color;
  points.forEach(([x, y]) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 2.8 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

window.addEventListener("resize", () => {
  drawChart();
});

connectEvents();
requestAnimationFrame(drawSimulation);
fetch("/api/state").then((r) => r.json()).then(applyState).catch(() => {});
