"""
文本相似度匹配 API 服务

基于 FastAPI + sentence-transformers 实现，支持多模型切换和批量推理。
"""

import logging
import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== 模型配置 ==============

# 支持的模型映射表
SUPPORTED_MODELS: Dict[str, str] = {
    "bge-large-zh": "BAAI/bge-large-zh-v1.5",
    "qwen3-embedding-0.6B": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # 备选方案
}

# 默认模型名称
DEFAULT_MODEL_NAME = "bge-large-zh"


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
        description="使用的Embedding模型名，支持: 'bge-large-zh', 'qwen3-embedding-0.6B'"
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
    loaded_models: List[str]


# ============== 模型管理器（单例模式） ==============

class ModelManager:
    """
    Embedding 模型管理器（单例模式）
    
    负责模型的懒加载和缓存，确保每个模型只加载一次，避免重复加载造成的性能损耗。
    """
    _instance: Optional["ModelManager"] = None
    _models: Dict[str, SentenceTransformer] = {}
    _lock: bool = False
    
    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_model_key(self, model_name: str) -> str:
        """获取模型缓存键"""
        return model_name.lower().strip()
    
    def _get_model_path(self, model_name: str) -> str:
        """
        获取模型路径
        
        支持传入简称或完整路径，自动映射到 HuggingFace 模型名
        """
        model_key = self._get_model_key(model_name)
        
        # 如果是支持的简称，映射到完整路径
        if model_key in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_key]
        
        # 否则直接返回传入的路径（支持自定义模型）
        return model_name
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        加载指定模型（懒加载）
        
        Args:
            model_name: 模型名称（支持简称或完整路径）
            
        Returns:
            SentenceTransformer: 加载好的模型实例
            
        Raises:
            ValueError: 模型名称不支持时
            RuntimeError: 模型加载失败时
        """
        model_key = self._get_model_key(model_name)
        
        # 检查缓存
        if model_key in self._models:
            logger.info(f"使用缓存的模型: {model_name}")
            return self._models[model_key]
        
        # 防止并发加载
        if self._lock:
            raise RuntimeError("模型正在加载中，请稍后重试")
        
        try:
            self._lock = True
            model_path = self._get_model_path(model_name)
            
            logger.info(f"正在加载模型: {model_name} (路径: {model_path})")
            
            # 加载模型
            model = SentenceTransformer(model_path)
            
            # 缓存模型
            self._models[model_key] = model
            
            logger.info(f"模型加载成功: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {model_name}, 错误: {str(e)}")
            raise RuntimeError(f"无法加载模型 '{model_name}': {str(e)}")
        finally:
            self._lock = False
    
    def get_model(self, model_name: str) -> SentenceTransformer:
        """
        获取已加载的模型，如未加载则自动加载
        
        Args:
            model_name: 模型名称
            
        Returns:
            SentenceTransformer: 模型实例
        """
        model_key = self._get_model_key(model_name)
        
        if model_key not in self._models:
            return self.load_model(model_name)
        
        return self._models[model_key]
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self._models.keys())
    
    def preload_models(self, model_names: List[str]) -> None:
        """
        预加载多个模型
        
        Args:
            model_names: 需要预加载的模型名称列表
        """
        for name in model_names:
            try:
                self.load_model(name)
            except Exception as e:
                logger.warning(f"预加载模型失败: {name}, 错误: {str(e)}")
    
    def match(
        self,
        source_text: str,
        compare_list: List[str],
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        执行相似度匹配
        
        Args:
            source_text: 源文本
            compare_list: 对比文本列表
            model_name: 模型名称
            threshold: 相似度阈值
            
        Returns:
            Dict: 匹配结果
        """
        model = self.get_model(model_name)
        result = SimilarityMatcher.match(
            source_text=source_text,
            compare_list=compare_list,
            model=model,
            threshold=threshold
        )
        return {
            "matched_index": result.matched_index,
            "max_score": result.max_score,
            "model_used": model_name
        }


# 全局模型管理器实例
model_manager = ModelManager()


# ============== 核心业务逻辑 ==============

class SimilarityMatcher:
    """
    文本相似度匹配器
    
    负责计算文本之间的余弦相似度，找出最相似的文本。
    """
    
    @staticmethod
    def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
        """
        批量计算文本的 Embedding 向量
        
        Args:
            texts: 文本列表
            model: SentenceTransformer 模型
            
        Returns:
            np.ndarray: 文本向量矩阵 (N x D)
        """
        # 使用模型批量编码，提高效率
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
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
        # 确保维度正确
        if source_embedding.ndim == 1:
            source_embedding = source_embedding.reshape(1, -1)
        
        # 计算余弦相似度
        # source: (1 x D), compare: (N x D) -> similarities: (1 x N)
        similarities = cosine_similarity(source_embedding, compare_embeddings)
        
        # 获取最高相似度及其索引
        max_score = float(np.max(similarities))
        max_index = int(np.argmax(similarities))
        
        return max_index, max_score
    
    @classmethod
    def match(
        cls,
        source_text: str,
        compare_list: List[str],
        model: SentenceTransformer,
        threshold: float
    ) -> MatchResponse:
        """
        执行相似度匹配
        
        Args:
            source_text: 源文本
            compare_list: 对比文本列表
            model: Embedding 模型
            threshold: 相似度阈值
            
        Returns:
            MatchResponse: 匹配结果
        """
        # 边界情况处理：空列表
        if not compare_list:
            return MatchResponse(
                matched_index=-1,
                max_score=0.0,
                model_used="unknown"
            )
        
        # 批量计算 Embedding
        all_texts = [source_text] + compare_list
        embeddings = cls.compute_embeddings(all_texts, model)
        
        # 分离源文本和对比文本的向量
        source_embedding = embeddings[0:1]  # (1, D)
        compare_embeddings = embeddings[1:]  # (N, D)
        
        # 找出最相似的文本
        max_index, max_score = cls.find_most_similar(source_embedding, compare_embeddings)
        
        # 阈值判定
        if max_score >= threshold:
            matched_index = max_index
        else:
            matched_index = -1
        
        return MatchResponse(
            matched_index=matched_index,
            max_score=round(max_score, 4),
            model_used="unknown"
        )


# ============== FastAPI 应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时预加载默认模型，确保第一次请求时模型已就绪。
    """
    logger.info("应用启动中...")
    
    # 预加载默认模型
    try:
        model_manager.load_model(DEFAULT_MODEL_NAME)
        logger.info(f"默认模型 '{DEFAULT_MODEL_NAME}' 预加载成功")
    except Exception as e:
        logger.warning(f"默认模型预加载失败: {str(e)}")
    
    yield
    
    # 清理资源（如有需要）
    logger.info("应用关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="文本相似度匹配 API",
    description="基于 Embedding 模型的文本相似度匹配服务",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        loaded_models=model_manager.get_loaded_models()
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
        # 获取或加载模型
        model = model_manager.get_model(request.model_name)
        
        # 执行匹配
        result = SimilarityMatcher.match(
            source_text=request.source_text,
            compare_list=request.compare_list,
            model=model,
            threshold=request.threshold
        )
        
        # 设置实际使用的模型名称
        result.model_used = request.model_name
        
        return result
        
    except ValueError as e:
        logger.error(f"请求参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"模型加载错误: {str(e)}")
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
    # 从环境变量获取配置
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
