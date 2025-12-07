import asyncio
import os
import time
import sys
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure loggers to be quiet
logging.basicConfig(level=logging.ERROR)
from agentscope import logger as agentscope_logger
agentscope_logger.setLevel("ERROR")

from deep_research_agent.agent.deep_research_agent import DeepResearchAgent
from deep_research_agent.core.mcp_client import get_tavily_client
from deep_research_agent.config.defaults import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OPERATION_DIR,
)

# Check for keys
if not os.environ.get("TAVILY_API_KEY"):
    print("ERROR: TAVILY_API_KEY environment variable is not set.")
    print("Please run: export TAVILY_API_KEY='your_key'")
    sys.exit(1)

if not os.environ.get("DASHSCOPE_API_KEY"):
    print("ERROR: DASHSCOPE_API_KEY environment variable is not set.")
    print("Please run: export DASHSCOPE_API_KEY='your_key'")
    sys.exit(1)

async def run_benchmark(mode_name: str, parallel: bool, query: str):
    print(f"\n[{mode_name}] Initializing...")
    
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
            enable_parallel=parallel
        )
        
        msg = Msg("Bob", content=query, role="user")
        
        print(f"[{mode_name}] Starting execution...")
        start_time = time.time()
        
        # Run the agent
        # We wrap in a timeout to prevent infinite hanging if something goes wrong
        await asyncio.wait_for(agent(msg), timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"[{mode_name}] Finished. Time taken: {duration:.2f} seconds")
        return duration

    except Exception as err:
        print(f"[{mode_name}] Failed with error: {err}")
        # Return 0 to indicate failure
        return 0
    finally:
        await tavily_search_client.close()

from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg

async def main():
    # A slightly simpler query to avoid overwhelming the context even with truncation
    # But still complex enough to benefit from parallel search
    query = (
        "Find the 2023 GDP growth rate and inflation rate for: "
        "China, USA, and Germany. Summarize briefly."
    )
    
    print(f"Benchmark Query: {query}")
    print("-" * 60)

    # Run Serial
    serial_time = await run_benchmark("SERIAL", False, query)
    
    # Pause to reset state/API limits
    print("\nWaiting 5 seconds...")
    await asyncio.sleep(5)
    
    # Run Parallel
    parallel_time = await run_benchmark("PARALLEL", True, query)

    print("\n" + "=" * 30)
    print("BENCHMARK RESULTS")
    print("=" * 30)
    
    if serial_time > 0:
        print(f"Serial Execution Time  : {serial_time:.2f} s")
    else:
        print("Serial Execution Time  : FAILED")

    if parallel_time > 0:
        print(f"Parallel Execution Time: {parallel_time:.2f} s")
    else:
        print("Parallel Execution Time: FAILED")

    if serial_time > 0 and parallel_time > 0:
        improvement = (serial_time - parallel_time) / serial_time * 100
        ratio = serial_time / parallel_time
        print(f"Speedup Improvement    : {improvement:.2f}%")
        print(f"Ratio (Serial/Parallel): {ratio:.2f}x")
    print("=" * 30)

if __name__ == "__main__":
    asyncio.run(main())
