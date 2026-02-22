"""FastAPI 应用定义与组件初始化。

FastAPI 作为智能路由中心，底座原生构建于 asyncio 事件循环之上，
适合 RAG 这种并发长链接和网络 I/O 密集型操作。
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

# 静态文件（挂载在路由注册之后，作为兜底）
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
