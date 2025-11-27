# Deep Research Agent

A runnable example of a Deep Research Agent using Agentscope and an MCP Tavily client.

Quickstart:
1) Prepare environment variables (see .env.example) or export via shell.
2) Install dependencies:
   pip install -e .
3) Run demo:
   python -m deep_research_agent.runners.main --query "Your question here"

Requirements:
- Python 3.10+
- Environment variables:
  - DASHSCOPE_API_KEY
  - TAVILY_API_KEY