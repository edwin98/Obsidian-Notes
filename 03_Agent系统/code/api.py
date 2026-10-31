"""本机教学 API：页面和 HTTP 接口共用 TaskService。

请求处理、后台执行和只读查询分别由 task_api 管理。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from task_api import active_count
from task_api import router as task_router

app = FastAPI(title="测试与排查 Agent")
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
app.include_router(task_router)


@app.get("/", include_in_schema=False)
async def learning_ui():
    return FileResponse(WEB_DIR / "tasks.html")


@app.get("/health")
async def health():
    return {"status": "ok", "active_tasks": active_count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
