"""FastAPI 应用定义与组件初始化。

FastAPI 作为智能路由中心，底座原生构建于 asyncio 事件循环之上，
适合 RAG 这种并发长链接和网络 I/O 密集型操作。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api.dependencies import init_components, shutdown_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="RAG Demo System",
    description="基于三级渐进式混合检索的 RAG 系统 Demo",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """启动时初始化所有组件连接。"""
    init_components()


@app.on_event("shutdown")
async def shutdown():
    """关闭时清理连接。"""
    shutdown_components()


# 注册路由
from api.routes import router  # noqa: E402

app.include_router(router)
