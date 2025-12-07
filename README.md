# HCDSA

High-Concurrency Deep Search Agent

## 项目说明

大模型搜索和检索增强（RAG)是一项非常重要的工具，能够帮助大模型不需要训练和微调就能实现规范化可信的输出。

经过调研，当前的 大语言模型搜索基本都是采用串行策略（如GPT、Gemini等)，因为串行策略，大模型能够基于上一个回答动态地修改下一次的搜索prompt细节。然而，很多搜索是可以并行的，尤其是小领域微调模型，比如：

```
调研2022、2023、2024，中、美、英、法国的经济增长趋势和通货膨胀率的关系
子任务-->
1. 2022 中国的 经济增长趋势 
2. 2022 中国的 通货膨胀率
…… 
```

这里，至少能拆分出3乘4乘2=24个可以并行的子任务。我们可以先绘制概念DAG图，对于同层的问题展开并行。（目前已经完成)

同时，在小领域专用大语言模型，用户经常会提问相似的问题，比如：

```
用户A：调研2022、2023、2024，中、美、英、法国的经济增长趋势和通货膨胀率的关系
用于B：比较2022、2023、2024，中、美、英的经济GDP增长率
用户C：2024年，查询通货膨胀率是如何影响到经济增长率的
```

这些deep resarch问题，模型都需要先填写一样的知识空白，在互联网中查询末年末国具体的数据，因此多查询请求是可以使用缓存策略并行的（完成部分，但是出现有些查询失败导致实验使用缓存没有明显的差距)。

GrapthRAG（Microsoft 2024.12)是当前被关注的技术，其实我们在概念图上的缓存和检索本质上也是一种GrapthRAG，后续可以尝试实现（Todo)

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

可以参考 `./deep_research_agent`的[README](./deep_research_agent/README.md)。

```
cd deep_reasrch_agent
bash scripts/example.sh
```

注意要先准备[千问](https://www.aliyun.com/product/bailian)和[tavily](https://www.tavily.com/)的API，目前是来自gjc的API，如果没有反应可能是欠费了

运行后会在指定的 `AGENT_OPERATION_DIR`目录下生成过程文件和输出报告。

## 单次调研逻辑图建立和子问题并行

```
cd deep_reasrch_agent
bash scripts/example.sh
```

会输出子问题DAG：

```
==================== Semantic Parallel Plan (Experimental) ====================
- Subtask #1: Retrieve 2023 GDP growth rate for China from official or reputable economic data sources.
  depends_on: []
- Subtask #2: Retrieve 2023 GDP growth rate for the USA from official or reputable economic data sources.
  depends_on: []
- Subtask #3: Retrieve 2023 GDP growth rate for Germany from official or reputable economic data sources.
……
- Subtask #7: Research and summarize the historical GDP and inflation data for China from 2013 to 2023 using official or reputable sources.
  depends_on: [1, 4]
- Subtask #8: Research and summarize the historical GDP and inflation data for the USA from 2013 to 2023 using official or reputable sources.
  depends_on: [2, 5]
……
```

会展示分层并行和串行的时间差：

```
[Semantic Execution] Layer 1 - 6 subtasks (parallel)
  - #1: Retrieve 2023 GDP growth rate for China from official or reputable economic data sources.
  - #2: Retrieve 2023 GDP growth rate for the USA from official or reputable economic data sources.
  - #3: Retrieve 2023 GDP growth rate for Germany from official or reputable economic data sources.
  - #4: Retrieve 2023 inflation rate for China from official or reputable economic data sources.
  - #5: Retrieve 2023 inflation rate for the USA from official or reputable economic data sources.
  - #6: Retrieve 2023 inflation rate for Germany from official or reputable economic data sources.
[Semantic Execution] Layer 1 parallel time : 2.49 s
[Semantic Execution] Layer 1 serial estimate: 11.09 s (speedup ~ 4.45x)
```

### 多问题缓存策略

```
cd deep_reasrch_agent
bash scripts/para_test.sh
```

现在可以实现，247个子问题命中49个，命中率20%
