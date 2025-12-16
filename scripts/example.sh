#!/usr/bin/env bash
set -euo pipefail

# ========== 配置区域（请按需修改） ==========
# 阿里云百炼 API Key（以 sk- 开头）
# export DASHSCOPE_API_KEY="Your-dashscope-api-key"
# Tavily 搜索 API Key
export TAVILY_API_KEY="Your-tavily-api-key"
# 任务运行的工作目录（用于缓存/中间产物/日志等）
export AGENT_OPERATION_DIR="Your-agent-operation-dir"

# 查询问题（也可以用命令行参数覆盖）
DEFAULT_QUERY="Find the 2023 GDP growth rate and inflation rate for China, USA, and Germany. Summarize briefly."
QUERY="${1:-$DEFAULT_QUERY}"

# ==========================================

# 运行
python -m deep_research_agent.runners.main --query "${QUERY}"