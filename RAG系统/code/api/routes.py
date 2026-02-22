"""API 路由：在线问答、文档摄入、健康检查。

完整在线链路：
缓存检查 -> 查询改写 -> 三级检索 -> LLM 流式生成 -> 异步写缓存 + 触发摘要检查
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import Components, get_components
from models.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from tasks.summarize import summarize_history

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check(comp: Components = Depends(get_components)):
    """健康检查。"""
    return {
        "status": "ok",
        "chunks_indexed": len(comp.chunk_store),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, comp: Components = Depends(get_components)):
    """非流式问答接口（返回完整 JSON）。"""
    # 1. Pydantic 参数强校验已由 FastAPI 自动完成

    # 2. 缓存检查
    try:
        cached = comp.redis_cache.get_exact_cache(request.query)
        if cached:
            return ChatResponse(answer=cached, source="exact_cache")

        query_vec = comp.embedder.embed_384(request.query)
        semantic_cached = comp.redis_cache.get_semantic_cache(query_vec)
        if semantic_cached:
            return ChatResponse(answer=semantic_cached, source="semantic_cache")
    except Exception as e:
        logger.warning("[Cache] 缓存查询失败（继续 RAG 链路）: %s", e)

    # 3. 获取会话历史
    try:
        history = comp.redis_cache.get_session_messages(
            request.user_id, request.session_id
        )
    except Exception:
        history = []

    # 4. 查询改写（指代消解 + 问题扩展）
    rewritten = comp.rewriter.rewrite(request.query, history)

    # 5. 三级检索
    results = await comp.retriever.retrieve(
        request.query, rewritten, top_k=request.top_k
    )

    # 6. LLM 生成
    chunks = [r.chunk for r in results]
    answer_parts = []
    async for token in comp.generator.generate_stream(request.query, chunks, history):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    citations = [r.chunk.chunk_id for r in results]

    # 7. 异步写缓存 + 更新会话
    try:
        comp.redis_cache.set_exact_cache(request.query, answer)
        comp.redis_cache.set_semantic_cache(request.query, query_vec, answer)
        comp.redis_cache.push_message(
            request.user_id,
            request.session_id,
            {"role": "user", "content": request.query},
        )
        comp.redis_cache.push_message(
            request.user_id,
            request.session_id,
            {"role": "assistant", "content": answer},
        )
    except Exception as e:
        logger.warning("[Cache] 缓存写入失败: %s", e)

    # 8. 触发 Celery 异步摘要检查
    try:
        summarize_history.delay(request.user_id, request.session_id)
    except Exception as e:
        logger.warning("[Celery] 摘要任务触发失败: %s", e)

    return ChatResponse(
        answer=answer,
        citations=citations,
        rewritten_queries=rewritten,
        source="rag",
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, comp: Components = Depends(get_components)):
    """SSE 流式问答接口（Server-Sent Events 打字机效果）。"""
    # 1. 缓存检查
    try:
        cached = comp.redis_cache.get_exact_cache(request.query)
        if cached:

            async def cached_stream():
                yield f"data: {cached}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(cached_stream(), media_type="text/event-stream")
    except Exception:
        pass

    # 2. 获取历史 + 改写 + 检索
    try:
        history = comp.redis_cache.get_session_messages(
            request.user_id, request.session_id
        )
    except Exception:
        history = []

    rewritten = comp.rewriter.rewrite(request.query, history)
    results = await comp.retriever.retrieve(
        request.query, rewritten, top_k=request.top_k
    )
    chunks = [r.chunk for r in results]

    # 3. SSE 流式生成
    async def sse_generator():
        full_answer: list[str] = []
        async for token in comp.generator.generate_stream(
            request.query, chunks, history
        ):
            full_answer.append(token)
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

        # 流结束后异步写缓存
        answer = "".join(full_answer)
        try:
            query_vec = comp.embedder.embed_384(request.query)
            comp.redis_cache.set_exact_cache(request.query, answer)
            comp.redis_cache.set_semantic_cache(request.query, query_vec, answer)
            comp.redis_cache.push_message(
                request.user_id,
                request.session_id,
                {"role": "user", "content": request.query},
            )
            comp.redis_cache.push_message(
                request.user_id,
                request.session_id,
                {"role": "assistant", "content": answer},
            )
        except Exception as e:
            logger.warning("[Cache] 流式缓存写入失败: %s", e)

        try:
            summarize_history.delay(request.user_id, request.session_id)
        except Exception:
            pass

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest, comp: Components = Depends(get_components)
):
    """文档摄入接口：解析 -> 清洗 -> 切分 -> Kafka -> Embedding -> 索引。"""
    try:
        chunks = comp.ingestion_pipeline.ingest_document(
            doc_id=request.doc_id,
            doc_name=request.doc_name,
            raw_content=request.content,
            file_type="markdown",
        )
    except Exception:
        logger.warning("[Ingest] Kafka 链路失败，使用直接模式")
        chunks = comp.ingestion_pipeline.ingest_document_direct(
            doc_id=request.doc_id,
            doc_name=request.doc_name,
            raw_content=request.content,
            file_type="markdown",
        )

    # 刷新 ES 索引使文档可搜索
    try:
        comp.bm25_engine.refresh()
    except Exception:
        pass

    # 刷新 Milvus
    try:
        comp.vector_engine.flush()
    except Exception:
        pass

    return IngestResponse(status="success", chunks_created=len(chunks))
