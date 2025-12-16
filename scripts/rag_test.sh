#!/usr/bin/env bash

# 阿里云百炼 API Key（以 sk- 开头）
# export DASHSCOPE_API_KEY="Your-dashscope-api-key"

# Tavily 搜索 API Key
# export TAVILY_API_KEY="Your-tavily-api-key"

python scripts/benchmark_rag.py

# 只测试 RAG 检索性能（不调用 Tavily）
python scripts/benchmark_rag.py --skip-tavily

# 使用持久化
python scripts/benchmark_rag.py --persist-path ./test/rag_cache.json

# 调整相似度阈值
python scripts/benchmark_rag.py --threshold 0.6