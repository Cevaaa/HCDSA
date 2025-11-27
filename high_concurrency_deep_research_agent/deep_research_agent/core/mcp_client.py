#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
from agentscope.mcp import StdIOStatefulClient


async def get_tavily_client() -> StdIOStatefulClient:
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        raise RuntimeError("Missing TAVILY_API_KEY, please export it before running.")

    client = StdIOStatefulClient(
        name="tavily_mcp",
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={"TAVILY_API_KEY": tavily_key},
    )
    await client.connect()
    return client


async def main():
    client = await get_tavily_client()
    caps = await client.list_tools()
    print("Tavily MCP tools:", caps)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())