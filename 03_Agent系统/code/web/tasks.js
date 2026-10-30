"use strict";

// 本页面只提交用户消息、读取事实与展示，不在浏览器里推断实验通过或原因确认。
// 对应后端：task_api.py → TaskService → experiments.py。
const $ = (id) => document.getElementById(id);
let taskId = sessionStorage.getItem("investigation-task") || null;
let conversation = sessionStorage.getItem("investigation-chat") || crypto.randomUUID();
let timer;
sessionStorage.setItem("investigation-chat", conversation);

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail);
  }
  return data;
}

function renderExperiment(experiment) {
  const article = document.createElement("article");
  const title = document.createElement("h3");
  title.textContent = experiment.plan.question;
  const detail = document.createElement("p");
  const condition = experiment.plan.conditions;
  detail.textContent = `${condition.version} · ${condition.source_cell}→${condition.target_cell} · ${condition.load} · ${condition.handovers} 次切换 · ${experiment.status}`;
  const result = document.createElement("p");
  result.textContent = experiment.interpretation || "正在执行；返回会保存在本组中。";
  const raw = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "参数、运行和原始记录";
  const pre = document.createElement("pre");
  // 日志、模型输出与用户文字都作为文本展示，不用innerHTML解释它们。
  pre.textContent = JSON.stringify(experiment, null, 2);
  raw.append(summary, pre);
  article.append(title, detail, result, raw);
  return article;
}

async function refresh() {
  if (!taskId) return;
  try {
    const requestedTask = taskId;
    const task = await request(`/tasks/${encodeURIComponent(requestedTask)}`);
    // 用户可能在fetch期间切换了对话，旧任务的迟到回应不能覆盖新页面。
    if (requestedTask !== taskId) return;
    $("identity").textContent = `${task.task_id} · ${task.task_type}`;
    $("state").textContent = `${task.status} · 已使用及预留 ${task.reserved_runs}/${task.max_runs} 次运行`;
    $("question").textContent = task.question || task.stop_reason || "";
    $("report").textContent = Object.keys(task.report).length
      ? JSON.stringify(task.report, null, 2)
      : "尚未结束，可查询已有进度或查看实验记录。";
    $("experiments").replaceChildren(...task.experiments.map(renderExperiment));
    clearTimeout(timer);
    // 轮询只是GET。WAITING/PAUSED等状态停止轮询，不偷偷安排实验。
    const active = ["READY", "RUNNING", "PAUSING", "CANCELLING"];
    if (active.includes(task.status) && $("auto").checked) {
      timer = setTimeout(refresh, 800);
    }
  } catch (error) {
    $("notice").textContent = error.message;
  }
}

async function sendMessage() {
  $("send").disabled = true;
  const submittedConversation = conversation;
  try {
    const response = await request("/tasks/messages", {
      // 同一次消息重传应沿用ID；再次点击发送是一条新用户消息。
      message_id: crypto.randomUUID(),
      conversation_id: conversation,
      text: $("message").value,
      target_task_id: taskId,
      auto_run: $("auto").checked,
    });
    if (submittedConversation !== conversation) return;
    $("notice").textContent = response.message;
    if (response.previous_report) {
      $("previous").hidden = false;
      $("previous-report").textContent = JSON.stringify(response.previous_report, null, 2);
    }
    if (response.task_id) {
      taskId = response.task_id;
      sessionStorage.setItem("investigation-task", taskId);
    }
    await refresh();
  } catch (error) {
    $("notice").textContent = error.message;
  } finally {
    $("send").disabled = false;
  }
}

function startConversation() {
  // 换会话只改变页面关联。取消后台任务必须明确发送取消消息。
  clearTimeout(timer);
  taskId = null;
  conversation = crypto.randomUUID();
  sessionStorage.removeItem("investigation-task");
  sessionStorage.setItem("investigation-chat", conversation);
  $("identity").textContent = "新对话；原任务仍保留，未被取消。";
  $("state").textContent = "";
  $("question").textContent = "";
  $("experiments").replaceChildren();
  $("report").textContent = "暂无报告";
  $("previous").hidden = true;
  $("previous-report").textContent = "";
}

$("send").onclick = sendMessage;
$("new").onclick = startConversation;
$("refresh").onclick = refresh;
$("step").onclick = async () => {
  if (!taskId) return;
  try {
    // 单步是完成下一组实验，不是一次模型token或一条底层切换动作。
    await request(`/tasks/${encodeURIComponent(taskId)}/step`, {});
    await refresh();
  } catch (error) {
    $("notice").textContent = error.message;
  }
};
document.querySelectorAll("[data-example]").forEach((button) => {
  button.onclick = () => { $("message").value = button.dataset.example; };
});
refresh();
