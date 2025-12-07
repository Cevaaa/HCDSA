#!/usr/bin/env bash
set -euo pipefail

# ========== 配置区域（请按需修改） ==========
# 阿里云百炼 API Key（以 sk- 开头）
export DASHSCOPE_API_KEY="sk-4a8067ffa0404dac93f0e23d910e8f46"
# Tavily 搜索 API Key
export TAVILY_API_KEY="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"
# 任务运行的工作目录（用于缓存/中间产物/日志等）
export AGENT_OPERATION_DIR="/home/mmlab/gejunchen/Work/2025-11/Projects/HCDSA/results"
# 查询问题（也可以用命令行参数覆盖）

# ==========================================

# 运行
#python -m deep_research_agent.runners.main --query "${QUERY}"
python -m deep_research_agent.runners.main --query "Find the 2023 GDP growth rate and inflation rate for China, USA, and Germany. Summarize briefly."