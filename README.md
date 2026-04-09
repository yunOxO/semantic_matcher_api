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

| 模型简称 | 说明 |
|---------|------|
| bge-large-zh | 本地 BGE 模型 (端口 8801) |
| qwen3-embedding-0.6b | 本地 Qwen3 模型 (端口 8800) |
| text-embedding-v4 | 阿里云官方 Embedding API |  

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
    "model_name": "text-embedding-v4",
    "threshold": 0.85
  }'
```

**响应示例**:

```json
{
  "matched_index": 2,
  "max_score": 0.9234,
  "model_used": "text-embedding-v4"
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
| BGE_SERVICE_URL | http://localhost:8801/embeddings | BGE 模型服务地址 |
| QWEN_SERVICE_URL | http://localhost:8800/embeddings | Qwen 模型服务地址 |
| ALIYUN_EMB_SERVICE_URL | https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding | 阿里云 Embedding API 地址 |
| ALIYUN_API_KEY | sk-bc98c76b621047d2a73a56609c460ac7 | 阿里云 API Key |
| HTTP_TIMEOUT | 60.0 | HTTP 请求超时时间（秒） |

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

## 阿里云 Embedding API 使用说明

本项目支持使用阿里云官方的 text-embedding-v4 模型。

### 配置 API Key

使用环境变量设置阿里云 API Key：

```bash
# Linux/macOS
export ALIYUN_API_KEY="your-api-key-here"

# Windows PowerShell
$env:ALIYUN_API_KEY="your-api-key-here"
```

默认 API Key 已内置，建议使用您自己的 Key。

### 调用示例

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
    "model_name": "text-embedding-v4",
    "threshold": 0.85
  }'
```

### 自定义 API 地址

如果需要使用其他阿里云 API 网关地址，可以通过环境变量配置：

```bash
export ALIYUN_EMB_SERVICE_URL="https://your-custom-gateway.com/api/v1/services/embeddings/text-embedding/text-embedding"
```

## Flash Attention 2 配置（Qwen3-06B 模型专用）

Qwen3-06B 模型支持使用 Flash Attention 2 进行加速推理。请根据您的环境选择合适的版本安装：

### 前置条件检查

首先确认您的环境版本：

```bash
# 检查 Python 版本
python --version

# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 检查 CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

### 下载并安装 Flash Attention 2

访问 [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) 查找与您环境匹配的版本，然后使用以下命令下载和安装：

#### Linux x86_64 示例

```bash
# 示例：Python 3.10 + PyTorch 2.7 + CUDA 12.4
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu124torch2.7-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu124torch2.7-cp310-cp310-linux_x86_64.whl
```

#### Windows x86_64 示例

```bash
# 示例：Python 3.11 + PyTorch 2.11 + CUDA 12.8
# 使用 PowerShell 下载
Invoke-WebRequest -Uri "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.6/flash_attn-2.8.3+cu128torch2.11-cp311-cp311-win_amd64.whl" -OutFile "flash_attn-2.8.3+cu128torch2.11-cp311-cp311-win_amd64.whl"
pip install flash_attn-2.8.3+cu128torch2.11-cp311-cp311-win_amd64.whl
```

### Wheel 文件命名规则

文件名格式：`flash_attn-{flash_attn_version}+cu{cuda_version}torch{torch_version}-cp{python_version}-cp{python_version}-{platform}.whl`

例如：`flash_attn-2.8.3+cu124torch2.7-cp310-cp310-linux_x86_64.whl`

- Flash Attention 版本：2.8.3
- CUDA 版本：12.4
- PyTorch 版本：2.7
- Python 版本：3.10
- 平台：linux_x86_64

### 启动 Qwen3-06B 模型服务

安装 Flash Attention 2 后，启动 Qwen3-06B 模型服务：

```bash
python tool_emb_qwen3_06b.py
```

服务将在 `http://0.0.0.0:8800` 启动，并自动使用 Flash Attention 2 加速。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
