# 文本相似度匹配 API 服务

基于 FastAPI + sentence-transformers 实现的高性能文本相似度匹配服务，支持多模型切换和批量推理。

## 功能特性

- 高性能异步 API：基于 FastAPI 构建
- 多模型支持：支持 BGE、Qwen 等多种 Embedding 模型
- 单例模式加载：模型启动时预加载，避免重复加载
- 批量推理：一次性处理多个文本，提高效率
- 矩阵运算：使用 scikit-learn 计算余弦相似度
- 阈值判定：支持自定义相似度阈值
- 容器化部署：提供 Dockerfile 和 docker-compose 配置

## 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **数据验证**: Pydantic v2
- **向量计算**: scikit-learn + numpy
- **模型加载**: sentence-transformers
- **容器化**: Docker + Docker Compose

## 支持的模型

| 模型简称 | HuggingFace 模型名 |
|---------|-------------------|
| bge-large-zh | BAAI/bge-large-zh-v1.5 |
| qwen3-embedding-0.6B | Alibaba-NLP/gte-Qwen2-1.5B-instruct |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/similarity-match.git
cd similarity-match
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn 直接启动：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问 http://localhost:8000/docs 查看 API 文档。

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
docker-compose up -d
```

### 使用 Docker

```bash
docker build -t similarity-match .
docker run -d -p 8000:8000 --name similarity-match similarity-match
```

## API 使用

### 健康检查

```bash
curl http://localhost:8000/health
```

### 相似度匹配

**接口**: `POST /api/v1/similarity/match`

**请求参数**:

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|-----|-------|-----|
| source_text | string | 是 | - | 需要对比的新文章/文本 |
| compare_list | array | 是 | - | 被对比的历史文章列表 |
| model_name | string | 否 | bge-large-zh | 使用的 Embedding 模型名 |
| threshold | float | 否 | 0.85 | 相似度判定阈值 (0-1) |

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/similarity/match" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "这是一篇关于人工智能的文章",
    "compare_list": [
      "机器学习是人工智能的一个分支",
      "深度学习在图像识别中应用广泛",
      "这是一篇关于人工智能的文章，讨论了AI的发展"
    ],
    "model_name": "bge-large-zh",
    "threshold": 0.85
  }'
```

**响应示例**:

```json
{
  "matched_index": 2,
  "max_score": 0.9234,
  "model_used": "bge-large-zh"
}
```

**响应说明**:

- `matched_index`: 最高相似度文章在 compare_list 中的索引。若均低于阈值，则返回 -1
- `max_score`: 最高相似度的得分，保留4位小数
- `model_used`: 实际使用的模型名称

## 环境变量

| 变量名 | 默认值 | 说明 |
|-------|-------|-----|
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| RELOAD | false | 是否开启热重载（开发环境） |

## 项目结构

```
similarity-match/
├── main.py              # 主程序
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 构建文件
├── docker-compose.yml   # Docker Compose 配置
├── README.md           # 项目说明
├── .gitignore          # Git 忽略文件
└── .github/
    └── workflows/
        └── ci.yml      # CI/CD 工作流
```

## 性能优化建议

1. **模型缓存**: 服务启动时会自动预加载默认模型，避免首次请求延迟
2. **批量推理**: 将多个文本一次性送入模型，比单条处理效率更高
3. **GPU 加速**: 如果有 GPU，sentence-transformers 会自动使用
4. **模型选择**: 根据场景选择合适的模型，轻量级模型推理更快

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
