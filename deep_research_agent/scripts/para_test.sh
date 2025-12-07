#!/usr/bin/env bash
set -euo pipefail

# ========== 环境变量检查 ==========
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    export DASHSCOPE_API_KEY="sk-4a8067ffa0404dac93f0e23d910e8f46"
fi

if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    export TAVILY_API_KEY="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"
fi

# 任务运行的工作目录（用于缓存/中间产物/日志等），可通过环境变量覆盖
: "${AGENT_OPERATION_DIR:=$(pwd)/results}"
export AGENT_OPERATION_DIR

echo "AGENT_OPERATION_DIR=${AGENT_OPERATION_DIR}"

#python scripts/multi_query_benchmark.py --queries-file "../inputs/queries_1.txt"
python scripts/multi_query_benchmark.py --queries-file "../inputs/queries_1.txt" --no-cache