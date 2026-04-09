"""
文本相似度匹配 API 服务

基于 FastAPI + 外部 Embedding 服务实现，支持多模型切换和批量推理。
"""

import logging
import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== 配置 ==============

# 模型到服务地址的映射
MODEL_SERVICE_URLS: Dict[str, str] = {
    "bge-large-zh": os.getenv("BGE_SERVICE_URL", "http://localhost:8801/embeddings"),
    "qwen3-embedding-0.6b": os.getenv("QWEN_SERVICE_URL", "http://localhost:8800/embeddings"),
}

# 默认模型名称
DEFAULT_MODEL_NAME = "qwen3-embedding-0.6b"

# HTTP客户端超时设置
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))


# ============== Pydantic 数据模型 ==============

class MatchRequest(BaseModel):
    """相似度匹配请求模型"""
    source_text: str = Field(
        ...,
        description="需要对比的新文章/文本",
        min_length=1
    )
    compare_list: List[str] = Field(
        ...,
        description="被对比的历史文章列表",
        min_length=0
    )
    model_name: str = Field(
        DEFAULT_MODEL_NAME,
        description="使用的Embedding模型名，支持: 'bge-large-zh', 'qwen3-embedding-0.6B'",
    )
    threshold: float = Field(
        0.85,
        description="相似度判定阈值，默认0.85",
        ge=0.0,
        le=1.0
    )


class MatchResponse(BaseModel):
    """相似度匹配响应模型"""
    matched_index: int = Field(
        ...,
        description="最高相似度文章在 compare_list 中的索引。若均低于阈值，则返回 -1"
    )
    max_score: float = Field(
        ...,
        description="最高相似度的得分，保留4位小数"
    )
    model_used: str = Field(
        ...,
        description="实际使用的模型名称"
    )


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    available_models: List[str]


# ============== Embedding 服务客户端 ==============

class EmbeddingServiceClient:
    """
    Embedding 服务客户端
    
    负责通过 HTTP 接口调用外部 Embedding 服务获取文本向量。
    """
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    def get_service_url(self, model_name: str) -> str:
        """获取模型对应的服务地址"""
        model_key = model_name.lower().strip()
        if model_key not in MODEL_SERVICE_URLS:
            raise ValueError(f"不支持的模型: {model_name}，支持的模型: {list(MODEL_SERVICE_URLS.keys())}")
        return MODEL_SERVICE_URLS[model_key]
    
    async def get_embeddings(
        self,
        texts: List[str],
        model_name: str
    ) -> np.ndarray:
        """
        批量获取文本的 Embedding 向量
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            
        Returns:
            np.ndarray: 文本向量矩阵 (N x D)
        """
        if not texts:
            return np.array([])
        
        service_url = self.get_service_url(model_name)
        
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.post(
                service_url,
                json={
                    "input": texts,
                    "model": model_name
                }
            )
            response.raise_for_status()
            result = response.json()
            
            embeddings = []
            for data in result.get("data", []):
                embeddings.append(data.get("embedding", []))
            
            return np.array(embeddings)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding 服务返回错误: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Embedding 服务错误: {e.response.status_code}")
        except Exception as e:
            logger.error(f"调用 Embedding 服务失败: {str(e)}")
            raise RuntimeError(f"调用 Embedding 服务失败: {str(e)}")


# 全局 Embedding 客户端
_embedding_client: Optional[EmbeddingServiceClient] = None


async def get_embedding_client() -> EmbeddingServiceClient:
    """获取 Embedding 客户端实例"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingServiceClient(timeout=HTTP_TIMEOUT)
    return _embedding_client


# ============== 核心业务逻辑 ==============

class SimilarityMatcher:
    """
    文本相似度匹配器
    
    负责计算文本之间的余弦相似度，找出最相似的文本。
    """
    
    @staticmethod
    async def compute_embeddings(
        texts: List[str],
        model_name: str,
        client: EmbeddingServiceClient
    ) -> np.ndarray:
        """
        批量计算文本的 Embedding 向量
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            client: Embedding 服务客户端
            
        Returns:
            np.ndarray: 文本向量矩阵 (N x D)
        """
        if not texts:
            return np.array([])
        
        embeddings = await client.get_embeddings(texts, model_name)
        return embeddings
    
    @staticmethod
    def find_most_similar(
        source_embedding: np.ndarray,
        compare_embeddings: np.ndarray
    ) -> tuple[int, float]:
        """
        找出与源文本最相似的文本
        
        Args:
            source_embedding: 源文本向量 (1 x D) 或 (D,)
            compare_embeddings: 对比文本向量矩阵 (N x D)
            
        Returns:
            tuple: (最相似文本的索引, 相似度得分)
        """
        if source_embedding.ndim == 1:
            source_embedding = source_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(source_embedding, compare_embeddings)
        
        max_score = float(np.max(similarities))
        max_index = int(np.argmax(similarities))
        
        return max_index, max_score
    
    @classmethod
    async def match(
        cls,
        source_text: str,
        compare_list: List[str],
        model_name: str,
        threshold: float,
        client: EmbeddingServiceClient
    ) -> MatchResponse:
        """
        执行相似度匹配
        
        Args:
            source_text: 源文本
            compare_list: 对比文本列表
            model_name: 模型名称
            threshold: 相似度阈值
            client: Embedding 服务客户端
            
        Returns:
            MatchResponse: 匹配结果
        """
        if not compare_list:
            return MatchResponse(
                matched_index=-1,
                max_score=0.0,
                model_used=model_name
            )
        
        all_texts = [source_text] + compare_list
        embeddings = await cls.compute_embeddings(all_texts, model_name, client)
        
        source_embedding = embeddings[0:1]
        compare_embeddings = embeddings[1:]
        
        max_index, max_score = cls.find_most_similar(source_embedding, compare_embeddings)
        
        if max_score >= threshold:
            matched_index = max_index
        else:
            matched_index = -1
        
        return MatchResponse(
            matched_index=matched_index,
            max_score=round(max_score, 4),
            model_used=model_name
        )


# ============== FastAPI 应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    logger.info("应用启动中...")
    
    global _embedding_client
    _embedding_client = EmbeddingServiceClient(timeout=HTTP_TIMEOUT)
    await _embedding_client.__aenter__()
    
    logger.info("应用启动成功")
    
    yield
    
    if _embedding_client:
        await _embedding_client.__aexit__(None, None, None)
    
    logger.info("应用关闭")


app = FastAPI(
    title="文本相似度匹配 API",
    description="基于 Embedding 模型的文本相似度匹配服务",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        available_models=list(MODEL_SERVICE_URLS.keys())
    )


@app.post("/api/v1/similarity/match", response_model=MatchResponse)
async def similarity_match(request: MatchRequest):
    """
    文本相似度匹配接口
    
    将源文本与历史文章列表进行比对，返回最相似的文章索引。
    
    - **source_text**: 需要对比的新文章/文本
    - **compare_list**: 被对比的历史文章列表
    - **model_name**: 使用的 Embedding 模型名
    - **threshold**: 相似度判定阈值
    """
    try:
        client = await get_embedding_client()
        
        result = await SimilarityMatcher.match(
            source_text=request.source_text,
            compare_list=request.compare_list,
            model_name=request.model_name,
            threshold=request.threshold,
            client=client
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"请求参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Embedding 服务错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"内部服务器错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理请求时发生错误: {str(e)}"
        )


# ============== 主入口 ==============

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"启动服务: {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
