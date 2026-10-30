"use strict";

// 旧版单次图的逐节点学习页面，入口为 /legacy。
// SSE只接收后端进度；请求、审批、澄清分别调用HTTP接口。
// 页面保存可回看的状态快照，不把历史快照重新提交为执行命令。

const $ = (id) => document.getElementById(id);
const pretty = (value) => JSON.stringify(value, null, 2) ?? "（无）";
const titles = { clarify: "意图澄清", agent: "选择下一步", guardrail: "安全检查", tools: "执行工具", result_judge: "双轨判定", hitl: "人工审批", end: "结束" };
const statusNames = { idle: "等待开始", running: "正在执行", step_paused: "单步暂停", hitl_pending: "等待人工审批", awaiting_clarification: "等待补充", complete: "执行完成", rejected: "人工拒绝", error: "执行出错" };
const scenarioNames = { pass: "正常通过", regression: "回归故障", danger: "危险请求", review: "低置信度复核" };
let guides = {};
let taskId = null;
let stream = null;
let steps = [];
let selected = -1;
let status = "idle";
let busy = false;
let seen = new Set();
let reviewPayload = null;

// 动态内容只通过 textContent 写入，包括模型响应和工具输出。
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : pretty(data.detail));
  return data;
}

function updateControls() {
  $("status").textContent = statusNames[status];
  $("status").classList.toggle("warning", status === "hitl_pending" || status === "error");
  $("next").disabled = busy || status !== "step_paused";
  $("start").disabled = busy || ["running", "step_paused", "hitl_pending", "awaiting_clarification"].includes(status);
  $("approve").disabled = busy || status !== "hitl_pending";
  $("reject").disabled = busy || status !== "hitl_pending";
  $("review").hidden = status !== "hitl_pending";
  $("clarification").hidden = status !== "awaiting_clarification";
  $("latest").disabled = selected === steps.length - 1;
  const hints = { idle: "先开始一个实验", running: "节点正在执行…", step_paused: "真实执行下一节点，不会提前运行工具", hitl_pending: "请先在审批面板中作出决定", awaiting_clarification: "请先补充信息", complete: "已结束，可以回看或开始新实验", rejected: "已拒绝，可以回看或开始新实验", error: "运行出错，可以回看或重新开始" };
  $("next-hint").textContent = hints[status];
}

function setStatus(value) {
  status = value;
  updateControls();
}

function actionCard(container, title, description, value, failed = false) {
  const card = element("section", `action-card${failed ? " error" : ""}`);
  card.append(element("h4", "", title));
  if (description) card.append(element("p", "", description));
  if (value !== undefined) card.append(element("pre", "", typeof value === "string" ? value : pretty(value)));
  container.append(card);
}

function parseContent(content) {
  if (typeof content !== "string") return content;
  try { return JSON.parse(content); } catch { return content; }
}

function renderActions(step) {
  const container = $("actions");
  container.replaceChildren();
  const messages = step.update.messages || [];
  for (const message of messages) {
    for (const call of message.tool_calls || []) {
      actionCard(container, `请求工具 · ${call.name}`, `call_id: ${call.id}。这里只提出动作，尚未执行。`, call.args);
    }
    if (message.type === "tool") {
      const rejected = step.node === "hitl" && step.state.hitl_rejected;
      const label = rejected ? "拒绝反馈" : "工具观察";
      const explanation = rejected ? "人工拒绝后补齐协议消息；该工具没有执行。" : "对应前一轮请求。";
      actionCard(container, `${label} · ${message.name}`, `tool_call_id: ${message.tool_call_id} · ${message.status}。${explanation}`, parseContent(message.content), message.status === "error");
    } else if (message.content) {
      actionCard(container, "模型响应", "这是模型返回的可见消息。最终业务结论仍由判定节点确认。", parseContent(message.content));
    }
  }
  if (step.node === "clarify") {
    actionCard(container, "意图检查", step.state.needs_clarification ? "需要补充测试目标。" : "测试目标明确，允许进入执行循环。", { needs_clarification: step.state.needs_clarification, clarification_question: step.state.clarification_question });
  }
  if (step.node === "guardrail") {
    const last = step.input.messages.at(-1);
    actionCard(container, "安全检查结果", step.state.hitl_required ? "危险关键词触发人工审批，工具尚未执行。" : "未命中危险关键词，允许进入工具节点。", { hitl_required: step.state.hitl_required, pending_tool_calls: last?.tool_calls || [] });
  }
  if (step.node === "result_judge") {
    actionCard(container, "业务规则 + 模型归纳", "观察 stat_verdict、verdict 与 issues：硬规则失败不能被模型覆盖。", parseContent(step.state.final_result));
  }
  if (step.node === "hitl") {
    actionCard(container, "审批决定已写回", "恢复的是同一个 Checkpoint。审批历史保留在 review_history 中。", step.state.review_history.at(-1));
    if (step.state.final_result) actionCard(container, "保留的报告", "批准不会把低置信度改为高置信度。", parseContent(step.state.final_result));
  }
}

function renderDiff(step) {
  const container = $("diff");
  container.replaceChildren();
  const keys = [...new Set([...Object.keys(step.input), ...Object.keys(step.state)])];
  for (const key of keys) {
    if (pretty(step.input[key]) === pretty(step.state[key])) continue;
    const field = element("section", "diff-field");
    field.append(element("h4", "", key));
    const values = element("div", "diff-values");
    for (const [label, value] of [["执行前", step.input[key]], ["执行后", step.state[key]]]) {
      const column = element("div");
      column.append(element("small", "", label), element("pre", "", pretty(value)));
      values.append(column);
    }
    field.append(values);
    container.append(field);
  }
  if (!container.children.length) container.append(element("p", "view-note", "本步没有改变共享状态。"));
}

function selectStep(index) {
  if (!steps[index]) return;
  selected = index;
  const step = steps[index];
  $("empty").hidden = true;
  $("detail").hidden = false;
  $("detail-index").textContent = `STEP ${String(index + 1).padStart(2, "0")} / ${step.node}`;
  $("detail-title").textContent = titles[step.node] || step.node;
  $("detail-time").textContent = new Date(step.timestamp).toLocaleTimeString("zh-CN", { hour12: false });
  $("explanation").textContent = step.explanation;
  $("route").textContent = `${step.node} → ${step.next_node} · ${titles[step.next_node] || step.next_node}`;
  $("source").textContent = `对照源码：${guides[step.node]?.source || "nodes.py / graph.py"}`;
  $("raw-update").textContent = pretty(step.update);
  $("input-json").textContent = pretty(step.input);
  $("state-json").textContent = pretty(step.state);
  $("tool-count").textContent = step.state.tool_history?.length || 0;
  $("confidence").textContent = Number(step.state.confidence_score).toFixed(2);
  $("errors").textContent = step.state.error_count;
  document.querySelectorAll("[data-node]").forEach((node) => node.classList.toggle("active", node.dataset.node === step.node));
  document.querySelectorAll(".timeline-step").forEach((node, position) => {
    node.classList.toggle("selected", position === index);
    node.setAttribute("aria-pressed", String(position === index));
  });
  renderActions(step);
  renderDiff(step);
  updateControls();
}

function addStep(step) {
  const follow = selected === steps.length - 1;
  const index = steps.length;
  steps.push(step);
  if (index === 0) $("timeline").replaceChildren();
  const button = element("button", "timeline-step");
  button.type = "button";
  const content = element("span");
  content.append(element("span", "step-title", titles[step.node] || step.node));
  const call = step.update.messages?.flatMap((message) => message.tool_calls || [])[0];
  content.append(element("span", "step-subtitle", call ? call.name : step.node));
  button.append(element("span", "step-number", String(index + 1).padStart(2, "0")), content);
  button.addEventListener("click", () => selectStep(index));
  $("timeline").append(button);
  $("step-count").textContent = String(steps.length).padStart(2, "0");
  if (follow) {
    selectStep(index);
    $("timeline").scrollTop = $("timeline").scrollHeight;
  }
}

function showResult(event) {
  $("result").hidden = false;
  $("verdict").textContent = event.verdict;
  $("root-cause").textContent = event.root_cause || (event.verdict === "REJECTED" ? "人工拒绝继续执行。" : "流程结束。");
  $("issues").textContent = (event.issues || []).join("；") || `规则结论：${event.stat_verdict || "—"} · 模型自报置信度：${event.confidence_score ?? "—"}`;
  $("result").classList.toggle("failed", event.verdict !== "PASS");
  setStatus(event.verdict === "REJECTED" ? "rejected" : "complete");
  notice("运行已结束。可以点击任意历史步骤查看输入、输出和状态变化，也可以开始另一个场景。");
}

function handleEvent(event) {
  if (event.event === "connected") return;
  if (seen.has(event.sequence)) return;
  seen.add(event.sequence);
  switch (event.event) {
    case "node_update": addStep(event); break;
    case "step_paused":
      setStatus("step_paused");
      notice(`已在节点后暂停。下一步：${event.next_nodes.map((name) => `${name}（${titles[name] || name}）`).join("、")}。`);
      break;
    case "hitl_required":
      reviewPayload = event.review;
      $("review-reason").textContent = event.review.reason;
      $("review-payload").textContent = pretty(event.review);
      setStatus("hitl_pending");
      document.querySelectorAll("[data-node]").forEach((node) => node.classList.toggle("active", node.dataset.node === "hitl"));
      notice(`hitl 已触发 interrupt：${event.review.reason}。普通“下一步”不能绕过人工审批。`);
      break;
    case "clarification_required":
      $("question").textContent = event.question;
      setStatus("awaiting_clarification");
      break;
    case "complete": showResult(event); break;
    case "error": setStatus("error"); notice(`执行失败：${event.message}`, true); break;
    case "stream_end": stream?.close(); break;
  }
}

function connect(id) {
  stream?.close();
  stream = new EventSource(`/stream/${id}`);
  stream.onmessage = (message) => handleEvent(JSON.parse(message.data));
  stream.onerror = async () => {
    if (taskId !== id) return;
    notice("事件连接中断，正在自动重连；已收到的步骤会保留。", true);
    try {
      await request(`/result/${id}`);
    } catch (error) {
      if (taskId !== id) return;
      // 服务重启后，内存中的任务已不存在，不再无限重连。
      if (error.message === "任务不存在") {
        stream.close();
        setStatus("error");
        notice("任务已随服务重启失效，请开始一个新实验。", true);
        localStorage.removeItem("agent-demo-task");
      }
    }
  };
}

async function perform(operation) {
  busy = true;
  updateControls();
  try { await operation(); } catch (error) { notice(error.message, true); }
  finally { busy = false; updateControls(); }
}

async function advance(path, body, message) {
  const previous = status;
  setStatus("running");
  notice(message);
  try { await request(path, body); }
  catch (error) { setStatus(previous); throw error; }
}

$("run-form").addEventListener("submit", (event) => {
  event.preventDefault();
  perform(async () => {
    const query = $("query").value.trim();
    if (!query) throw new Error("请填写具体的测试请求。");
    const scenario = document.querySelector('input[name="scenario"]:checked').value;
    const stepMode = $("mode").value === "step";
    const response = await request("/demo/run", { query, scenario, step_mode: stepMode });
    stream?.close();
    taskId = response.task_id;
    steps = [];
    seen = new Set();
    selected = -1;
    reviewPayload = null;
    $("feedback").value = "";
    $("result").hidden = true;
    $("empty").hidden = false;
    $("detail").hidden = true;
    $("timeline").replaceChildren();
    $("step-count").textContent = "00";
    $("tool-count").textContent = "0";
    $("confidence").textContent = "—";
    $("errors").textContent = "0";
    $("task-label").textContent = `TASK ${taskId.slice(0, 12)} · ${scenarioNames[scenario]}`;
    document.querySelectorAll("[data-node]").forEach((node) => node.classList.remove("active"));
    localStorage.setItem("agent-demo-task", JSON.stringify({ taskId, scenario, query, stepMode }));
    setStatus("running");
    notice("实验已启动，正在等待第一个节点。刷新页面可以恢复当前实验轨迹。");
    connect(taskId);
  });
});

$("next").addEventListener("click", () => perform(async () => {
  await advance(`/step/${taskId}`, {}, "正在执行下一个节点…");
}));

for (const [id, approved] of [["approve", true], ["reject", false]]) {
  $(id).addEventListener("click", () => perform(async () => {
    await advance(`/resume/${taskId}`, { approved, feedback: $("feedback").value }, approved ? "审批决定已提交，正在恢复原来的断点。" : "拒绝决定已提交，正在结束本次流程。");
  }));
}

$("clarify-form").addEventListener("submit", (event) => {
  event.preventDefault();
  perform(async () => {
    await advance(`/clarify/${taskId}`, { answer: $("answer").value }, "正在追加补充信息…");
  });
});

$("latest").addEventListener("click", () => {
  selectStep(steps.length - 1);
  $("timeline").scrollTop = $("timeline").scrollHeight;
  if (reviewPayload) document.querySelector('[data-node="hitl"]').classList.add("active");
});

document.querySelectorAll("[data-tab]").forEach((button, index, buttons) => {
  button.addEventListener("click", () => {
    buttons.forEach((tab) => {
      const active = tab === button;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      $(`view-${tab.dataset.tab}`).hidden = !active;
    });
  });
  button.tabIndex = index === 0 ? 0 : -1;
  button.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let position = event.key === "ArrowRight" ? (index + 1) % buttons.length : (index - 1 + buttons.length) % buttons.length;
    if (event.key === "Home") position = 0;
    if (event.key === "End") position = buttons.length - 1;
    buttons[position].focus();
    buttons[position].click();
  });
});

async function initialize() {
  try {
    guides = (await request("/demo/config")).nodes;
    const saved = JSON.parse(localStorage.getItem("agent-demo-task") || "null");
    const requestedScenario = new URLSearchParams(location.search).get("scenario");
    if (saved && (!requestedScenario || requestedScenario === saved.scenario)) {
      const result = await request(`/result/${saved.taskId}`);
      taskId = saved.taskId;
      $("query").value = saved.query;
      $("mode").value = saved.stepMode ? "step" : "auto";
      const radio = [...document.querySelectorAll('[name="scenario"]')].find((node) => node.value === saved.scenario);
      if (radio) radio.checked = true;
      $("task-label").textContent = `TASK ${taskId.slice(0, 12)} · ${scenarioNames[saved.scenario]}`;
      setStatus(result.status);
      notice("正在恢复上一次实验的全部节点轨迹…");
      connect(taskId);
    } else if (requestedScenario && scenarioNames[requestedScenario]) {
      document.querySelector(`input[value="${requestedScenario}"]`).checked = true;
    }
  } catch (error) {
    localStorage.removeItem("agent-demo-task");
    notice(error.message === "任务不存在" ? "上次任务已随服务重启失效，可以开始新实验。" : `初始化失败：${error.message}`, true);
  }
}

updateControls();
initialize();
