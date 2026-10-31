"""【学习定位：统一HTTP适配】
路由只接消息与读结果；后台驱动TaskService。查询和报告不触发新实验，进程内_worker不是分布式队列。
对应11、14；服务装配、消息处理、调度、读取依次阅读。

统一单机教学接口。无生产认证，仅建议绑定本机地址运行。
"""

import asyncio
import os

from fastapi import APIRouter, HTTPException

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from task_models import MessageRequest
from task_service import TaskService
from task_store import ConflictError, TaskStore

router = APIRouter(prefix="/tasks", tags=["统一测试与调查"])
_service = None
_workers = {}
_automatic = {}


def create_service(*, mode=None, path=None):
    """装配依赖：online仅替换意图与计划器，执行器仍为教学沙盒。"""
    store = TaskStore(path or os.getenv("AGENT_TASK_DB", ":memory:"))
    options = {"store": store}
    if (mode or os.getenv("AGENT_TASK_MODE", "offline")) == "online":
        from langchain_openai import ChatOpenAI

        from experiments import ModelPlanner
        from intent import ModelIntent

        if not DEEPSEEK_API_KEY:
            raise ValueError("在线模式需要配置 DEEPSEEK_API_KEY")
        model = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0,
            timeout=60,
            max_retries=2,
        )
        options.update(interpreter=ModelIntent(model), planner=ModelPlanner(model))
    service = TaskService(**options)
    service.recover()
    return service


def service():
    """延迟创建进程内单例，导入路由时不连接在线模型或打开业务文件。"""
    global _service
    if _service is None:
        _service = create_service()
    return _service


def active_count():
    """统计真正处于执行或收尾阶段的任务，供首页健康接口观察。"""
    if _service is None:
        return 0
    with _service.store.lock:
        return sum(
            t.status in {"RUNNING", "PAUSING", "CANCELLING"}
            for t in _service.store.list()
        )


class Submission(MessageRequest):
    auto_run: bool = True


def _error(exc):
    """把业务异常翻译为 HTTP 状态；冲突与字段错误需要客户端区别处理。"""
    if isinstance(exc, KeyError):
        return HTTPException(404, "任务不存在或不属于当前对话")
    if isinstance(exc, ConflictError):
        return HTTPException(409, str(exc))
    return HTTPException(422, str(exc))


def _schedule(task_id):
    """一个后台驱动器按实验边界推进；查询接口不会调用本函数。"""
    instance = service()
    _automatic[task_id] = True
    if task_id in _workers and not _workers[task_id].done():

        def continue_if_ready(_):
            """旧驱动器退出时再看最新状态，接住执行期间到达的继续请求。"""
            if (
                _service is instance
                and _automatic.get(task_id)
                and instance.get(task_id).status == "READY"
            ):
                _schedule(task_id)

        _workers[task_id].add_done_callback(continue_if_ready)
        return

    async def drive():
        """阻塞实验放在线程中执行，让事件循环仍能接收查询与暂停消息。"""
        while _automatic.get(task_id):
            result = await asyncio.to_thread(instance.step, task_id)
            if result.status != "READY":
                return result
            await asyncio.sleep(0.02)

    _workers[task_id] = asyncio.create_task(drive())


@router.post("/messages")
async def message(req: Submission):
    """消息语义先落库，再依据run_requested决定是否调度。"""
    try:
        request = MessageRequest.model_validate(req.model_dump(exclude={"auto_run"}))
        response = await asyncio.to_thread(service().message, request)
    except (ValueError, KeyError) as exc:
        raise _error(exc) from exc
    if req.auto_run and response["run_requested"]:
        _schedule(response["task_id"])
    elif response["run_requested"]:
        _automatic[response["task_id"]] = False
    return response


@router.get("/{task_id}")
async def task(task_id: str):
    """只读取业务记录；不会推进图，也不因页面轮询而消耗实验次数。"""
    try:
        return service().get(task_id).model_dump()
    except KeyError as exc:
        raise _error(exc) from exc


@router.post("/{task_id}/step")
async def step(task_id: str):
    """切换到手动模式并推进一组实验；一组可包含多次 Run。"""
    try:
        _automatic[task_id] = False
        return (await asyncio.to_thread(service().step, task_id)).model_dump()
    except (ValueError, KeyError) as exc:
        raise _error(exc) from exc


@router.get("/{task_id}/events")
async def events(task_id: str):
    """返回已落库事件列表，使用 JSON 读取接口。"""
    return (await task(task_id))["events"]


@router.get("/{task_id}/experiments")
async def experiments(task_id: str):
    """展示完整实验与运行记录，供页面把摘要追到实际证据。"""
    return (await task(task_id))["experiments"]


@router.get("/{task_id}/report")
async def report(task_id: str):
    """阶段报告基于读取副本生成，GET操作不能顺便改变任务状态。"""
    try:
        snapshot = service().get(task_id)
        if not snapshot.report:
            service()._report(snapshot)  # 只对副本生成阶段视图，不改写运行状态。
        return snapshot.report
    except KeyError as exc:
        raise _error(exc) from exc
