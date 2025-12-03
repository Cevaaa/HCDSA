#!/usr/bin/env bash
set -euo pipefail

# ========== 配置区域（请按需修改） ==========
# 阿里云百炼 API Key（以 sk- 开头）
export DASHSCOPE_API_KEY="sk-37ef0c517cca46d5b8f4a8a20f6ba63a"
# Tavily 搜索 API Key
export TAVILY_API_KEY="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"
# 任务运行的工作目录（用于缓存/中间产物/日志等）
export AGENT_OPERATION_DIR="/home/mmlab/gejunchen/Work/2025-11/Projects/HCDSA/results"
# 查询问题（也可以用命令行参数覆盖）
DEFAULT_QUERY="If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."
QUERY="${1:-$DEFAULT_QUERY}"
# ==========================================

# 运行
python -m deep_research_agent.runners.main --query "${QUERY}"