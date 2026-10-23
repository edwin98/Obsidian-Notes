"""RAG Demo 系统一键启动脚本。

流程：
1. 初始化所有组件（Redis/ES/Milvus/Kafka/Celery）
2. 加载 8 篇示例文档执行 Ingestion
3. 运行端到端查询演示
4. 启动 FastAPI 服务（http://localhost:8000/docs）
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def load_sample_data():
    """加载示例文档到系统中。"""
    from api.dependencies import get_components
    from data.sample_documents import SAMPLE_DOCUMENTS

    comp = get_components()

    logger.info("=" * 60)
    logger.info("开始加载 %d 篇示例文档...", len(SAMPLE_DOCUMENTS))
    logger.info("=" * 60)

    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        try:
            chunks = comp.ingestion_pipeline.ingest_document(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                raw_content=doc["content"],
            )
        except Exception:
            logger.warning("Kafka 链路不可用，使用直接模式")
            chunks = comp.ingestion_pipeline.ingest_document_direct(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                raw_content=doc["content"],
            )
        total_chunks += len(chunks)
        logger.info("  [OK] %s -> %d chunks", doc["doc_name"], len(chunks))

    # 刷新索引
    try:
        comp.bm25_engine.refresh()
    except Exception:
        pass
    try:
        comp.vector_engine.flush()
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info(
        "数据加载完毕: %d 篇文档, %d 个 chunk", len(SAMPLE_DOCUMENTS), total_chunks
    )
    logger.info("=" * 60)


async def run_demo_queries():
    """运行端到端查询演示。"""
    from api.dependencies import get_components
    from data.sample_documents import TEST_QUERIES

    comp = get_components()

    logger.info("\n" + "=" * 60)
    logger.info("开始端到端查询演示...")
    logger.info("=" * 60)

    for i, test in enumerate(TEST_QUERIES[:4]):  # 演示前 4 个查询
        query = test["query"]
        logger.info("\n--- 查询 %d: %s ---", i + 1, query)
        logger.info("[期望命中] %s (%s)", test["expected_doc"], test["description"])

        # 查询改写
        rewritten = comp.rewriter.rewrite(query)
        logger.info("[改写结果] %s", rewritten)

        # 三级检索
        results = await comp.retriever.retrieve(query, rewritten, top_k=5)
        logger.info("[检索结果] %d 个候选:", len(results))
        for r in results:
            logger.info(
                "  - %s (score=%.4f) [%s] %s",
                r.chunk.chunk_id,
                r.score,
                r.chunk.metadata.doc_id,
                r.chunk.metadata.heading_path[:50],
            )

        # LLM 生成
        chunks = [r.chunk for r in results]
        answer_parts = []
        async for token in comp.generator.generate_stream(query, chunks):
            answer_parts.append(token)
        answer = "".join(answer_parts)
        logger.info(
            "[生成答案] %s", answer[:200] + ("..." if len(answer) > 200 else "")
        )

        # 缓存写入
        try:
            comp.redis_cache.set_exact_cache(query, answer)
            logger.info("[缓存] 已写入精确缓存")
        except Exception as e:
            logger.warning("[缓存] 写入失败: %s", e)

    # 缓存命中测试
    logger.info("\n--- 缓存命中测试 ---")
    try:
        cached = comp.redis_cache.get_exact_cache(TEST_QUERIES[0]["query"])
        if cached:
            logger.info("[Cache HIT] 精确缓存命中，直接返回（< 50ms）")
        else:
            logger.info("[Cache MISS] 缓存未命中")
    except Exception as e:
        logger.warning("[Cache] 测试失败: %s", e)

    logger.info("\n" + "=" * 60)
    logger.info("端到端演示完毕!")
    logger.info("=" * 60)


def main():
    # 切换到项目目录以便模块导入
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    # 1. 初始化组件
    from api.dependencies import init_components

    logger.info("初始化组件...")
    init_components()

    # 2. 加载示例数据
    load_sample_data()

    # 3. 运行演示查询
    asyncio.run(run_demo_queries())

    # 4. 启动 FastAPI 服务
    logger.info("\n" + "=" * 60)
    logger.info("启动 FastAPI 服务: http://localhost:8000/docs")
    logger.info("=" * 60)

    from api.app import app

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
