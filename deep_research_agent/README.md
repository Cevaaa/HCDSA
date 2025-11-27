# Deep Research Agent

A runnable example of a Deep Research Agent using Agentscope and an MCP Tavily client.

Quickstart:
1) Prepare environment variables (see ./deep_research_agent/scripts/example.sh) or export via shell.
2) Run demo:
```bash
# Make sure the running path is HCDSA/deep_research_agent/
bash scripts/example.sh
```

Requirements:
- Python 3.10+
- Environment variables:
  - DASHSCOPE_API_KEY
  - TAVILY_API_KEY
  - AGENT_OPERATION_DIR