# -*- coding: utf-8 -*-
"""The main entry point of the Deep Research agent example."""
import asyncio
import os
import argparse
from typing import List

from agentscope import logger
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg

from deep_research_agent.agent.deep_research_agent import DeepResearchAgent
from deep_research_agent.core.mcp_client import get_tavily_client
from deep_research_agent.config.defaults import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OPERATION_DIR,
)
from deep_research_agent.runners.utils import *


async def run_once(user_querys: List) -> None:
    logger.setLevel(DEFAULT_LOG_LEVEL)

    tavily_search_client = await get_tavily_client()

    agent_working_dir = os.getenv("AGENT_OPERATION_DIR", DEFAULT_OPERATION_DIR)
    os.makedirs(agent_working_dir, exist_ok=True)

    try:
        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are a helpful assistant named Friday.",
            model=DashScopeChatModel(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                model_name=DEFAULT_MODEL_NAME,
                enable_thinking=False,
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            search_mcp_client=tavily_search_client,
            tmp_file_storage_dir=agent_working_dir,
        )

        msgs = []
        for i in range(len(user_querys)):
            user_name = "Bob" + str(i+1)
            msg = Msg(
                user_name,
                content=user_querys[i],
                role="user",
            )
            msgs.append(msg)
        result = await agent(msgs)
        logger.info(result)

    except Exception as err:
        logger.exception(err)
    finally:
        await tavily_search_client.close()


def cli_entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, default="path/to/your/dataset", help="Input directory.")
    args = parser.parse_args()

    queries = read_queries_from_file(args.input_dir)
    
    asyncio.run(run_once(queries))


if __name__ == "__main__":
    cli_entry()