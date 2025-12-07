# -*- coding: utf-8 -*-
"""The main entry point of the Deep Research agent example."""
import asyncio
import os
import argparse

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


async def run_once(user_query: str) -> None:
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
        user_name = "Bob"
        msg = Msg(
            user_name,
            content=user_query,
            role="user",
        )
        print("Success init")
        result = await agent(msg)
        logger.info(result)

    except Exception as err:
        logger.exception(err)
    finally:
        await tavily_search_client.close()


def cli_entry():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        default=(
            "If Eliud Kipchoge could maintain his record-making "
            "marathon pace indefinitely, how many thousand hours "
            "would it take him to run the distance between the "
            "Earth and the Moon its closest approach? Please use "
            "the minimum perigee value on the Wikipedia page for "
            "the Moon when carrying out your calculation. Round "
            "your result to the nearest 1000 hours and do not use "
            "any comma separators if necessary."
        ),
        help="User query to research.",
    )
    args = parser.parse_args()
    asyncio.run(run_once(args.query))


if __name__ == "__main__":
    cli_entry()