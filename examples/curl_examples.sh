#!/bin/bash

# 文本相似度匹配 API 测试脚本
# 使用 curl 命令测试各种场景

BASE_URL="http://localhost:8000"

echo "=========================================="
echo "文本相似度匹配 API 测试 (curl)"
echo "=========================================="
echo ""

# 1. 健康检查
echo "1. 健康检查"
curl -s "$BASE_URL/health" | python -m json.tool
echo ""
echo ""

# 2. 正常匹配场景
echo "2. 相似度匹配 - 正常场景"
curl -s -X POST "$BASE_URL/api/v1/similarity/match" \
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
  }' | python -m json.tool
echo ""
echo ""

# 3. 无匹配场景
echo "3. 相似度匹配 - 无匹配场景（低于阈值）"
curl -s -X POST "$BASE_URL/api/v1/similarity/match" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "今天天气真好，适合去公园散步",
    "compare_list": [
      "机器学习是人工智能的一个分支",
      "深度学习在图像识别中应用广泛"
    ],
    "model_name": "bge-large-zh",
    "threshold": 0.85
  }' | python -m json.tool
echo ""
echo ""

# 4. 空列表场景
echo "4. 相似度匹配 - 空列表场景"
curl -s -X POST "$BASE_URL/api/v1/similarity/match" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "这是一篇测试文章",
    "compare_list": [],
    "model_name": "bge-large-zh",
    "threshold": 0.85
  }' | python -m json.tool
echo ""
echo ""

# 5. 不同模型测试
echo "5. 相似度匹配 - 使用不同模型"
curl -s -X POST "$BASE_URL/api/v1/similarity/match" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "Python 是一种流行的编程语言",
    "compare_list": [
      "Java 是企业级开发的首选语言",
      "Python 在数据科学领域非常流行"
    ],
    "model_name": "bge-large-zh",
    "threshold": 0.80
  }' | python -m json.tool
echo ""
echo ""

echo "=========================================="
echo "所有测试完成!"
echo "=========================================="
