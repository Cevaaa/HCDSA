#!/usr/bin/env bash
set -euo pipefail

# ========== 配置区域（请按需修改） ==========

export DASHSCOPE_API_KEY="Your-dashscope-api-key"
export DASHSCOPE_API_KEY_2="Your-dashscope-api-key"  # Optional: For API Pool

# Tavily 搜索 API Key
export TAVILY_API_KEY="Your-tavily-api-key"
export TAVILY_API_KEY_2="Your-tavily-api-key"      # Optional: For API Pool


export AGENT_OPERATION_DIR="${AGENT_OPERATION_DIR:-$(pwd)/results}"

echo "AGENT_OPERATION_DIR=${AGENT_OPERATION_DIR}"

# ========== 运行区域 ==========
# 如需使用缓存，去掉 --no-cache
# python scripts/mult_query_benchmark_w_APIpool.py --queries-file "../inputs/queries_1.txt"
python scripts/mult_query_benchmark_w_APIpool.py --queries-file "./inputs/queries_1.txt" --no-cache
