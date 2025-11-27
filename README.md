# HCDSA
High-Concurrency Deep Search Agent

## 文件说明
包含两个文件目录，分别是`./deep_research_agent`和`./high_concurrency_deep_research_agent`，前者是一个输入单个Query的最小运行示例，后者则为本项目所需搭建的高并发深度搜索智能体。

## 环境安装
1) 在自己的环境目录下安装 [AgentScope](https://github.com/agentscope-ai/agentscope)，注意不是该项目目录，下面四种方式选择其一即可。
```bash
# Make sure the running path is path/to/your/envs/, instead of this project
# From source
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
pip install -e .

# Using uv (recommended for faster installs)
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
uv pip install -e .

# From PyPi
pip install agentscope

# Or with uv
uv pip install agentscope
```

2) 为了启动MCP服务需要安装npm。

```bash
apt-get install npm
# 确保以下指令能够正常运行
npx -y tavily-mcp@latest
# Tavily MCP server running on stdio
```

3) 安装该项目的环境依赖，在该项目的目录下运行。（暂时无需这一步）
```bash
# Make sure the running path is HCDSA/
   pip install -e .
```

## 运行示例
运行`./deep_research_agent`参考[README](./deep_research_agent/README.md)。

注意要先准备[千问](https://www.aliyun.com/product/bailian)和[tavily](https://www.tavily.com/)的API，并且在`./deep_research_agent`下运行。

运行后会在指定的`AGENT_OPERATION_DIR`目录下生成过程文件和输出报告。