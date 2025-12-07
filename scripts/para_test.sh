#!/usr/bin/env bash
set -euo pipefail

# ========== 配置区域（请按需修改） ==========
# 阿里云百炼 API Key（以 sk- 开头）
export DASHSCOPE_API_KEY="Your-dashscope-api-key"
# Tavily 搜索 API Key
export TAVILY_API_KEY="Your-tavily-api-key"
# 任务运行的工作目录（用于缓存/中间产物/日志等）
export AGENT_OPERATION_DIR="${AGENT_OPERATION_DIR:-$(pwd)/results}"

echo "AGENT_OPERATION_DIR=${AGENT_OPERATION_DIR}"

# ========== 运行区域 ==========
# 如需使用缓存，去掉 --no-cache
# python scripts/multi_query_benchmark.py --queries-file "../inputs/queries_1.txt"
python scripts/multi_query_benchmark.py --queries-file "../inputs/queries_1.txt" --no-cache