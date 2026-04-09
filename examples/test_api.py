"""
API 测试脚本示例

用于测试相似度匹配 API 的各种场景。
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    response = requests.get(f"{BASE_URL}/health")
    print("健康检查:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_similarity_match():
    """测试相似度匹配接口 - 正常场景"""
    payload = {
        "source_text": "这是一篇关于人工智能的文章",
        "compare_list": [
            "机器学习是人工智能的一个分支",
            "深度学习在图像识别中应用广泛",
            "这是一篇关于人工智能的文章，讨论了AI的发展"
        ],
        "model_name": "bge-large-zh",
        "threshold": 0.85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/similarity/match",
        json=payload
    )
    print("相似度匹配 - 正常场景:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_similarity_no_match():
    """测试相似度匹配接口 - 无匹配场景"""
    payload = {
        "source_text": "今天天气真好，适合去公园散步",
        "compare_list": [
            "机器学习是人工智能的一个分支",
            "深度学习在图像识别中应用广泛",
            "自然语言处理是AI的重要方向"
        ],
        "model_name": "bge-large-zh",
        "threshold": 0.85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/similarity/match",
        json=payload
    )
    print("相似度匹配 - 无匹配场景:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_similarity_empty_list():
    """测试相似度匹配接口 - 空列表场景"""
    payload = {
        "source_text": "这是一篇测试文章",
        "compare_list": [],
        "model_name": "bge-large-zh",
        "threshold": 0.85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/similarity/match",
        json=payload
    )
    print("相似度匹配 - 空列表场景:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_different_threshold():
    """测试不同阈值"""
    payload = {
        "source_text": "Python 是一种流行的编程语言",
        "compare_list": [
            "Java 是企业级开发的首选语言",
            "Python 在数据科学领域非常流行",
            "Go 语言适合构建高性能服务"
        ],
        "model_name": "bge-large-zh",
        "threshold": 0.70
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/similarity/match",
        json=payload
    )
    print("相似度匹配 - 较低阈值 (0.70):")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("文本相似度匹配 API 测试")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_similarity_match()
        test_similarity_no_match()
        test_similarity_empty_list()
        test_different_threshold()
        
        print("=" * 50)
        print("所有测试完成!")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务，请确保服务已启动 (python main.py)")
    except Exception as e:
        print(f"测试出错: {str(e)}")
