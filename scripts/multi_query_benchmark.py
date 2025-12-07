import asyncio
import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.ERROR)
from agentscope import logger as agentscope_logger

agentscope_logger.setLevel("ERROR")

from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg

from deep_research_agent.agent.deep_research_agent import DeepResearchAgent
from deep_research_agent.core.mcp_client import get_tavily_client
from deep_research_agent.config.defaults import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OPERATION_DIR,
)
from deep_research_agent.core.cache import get_global_tavily_cache_stats


def _load_queries_from_file(path: str) -> List[str]:
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                queries.append(q)
    return queries


def _default_queries() -> List[str]:
    """一组默认的、经济与宏观数据相关的问题，用于制造子任务重叠。"""
    return [
        "Find the 2023 GDP growth rate and inflation rate for China, USA, and Germany. Summarize briefly.",
        "Compare the 2023 inflation rates of China, USA, and the Euro area, and explain main drivers.",
        "Summarize the 2023 GDP growth and CPI inflation for major G7 economies.",
        "For 2023, what are the GDP growth and inflation rates of Japan, UK, and France? Provide a short comparison.",
        "How did global inflation and GDP growth in 2023 differ between advanced economies and emerging markets?",
    ]


async def _run_single_query(
    idx: int,
    query: str,
    tavily_client,
    agent_working_dir: str,
    timeout: int = 600,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """运行单个 DeepResearchAgent，返回耗时等统计信息。"""
    agent = DeepResearchAgent(
        name=f"Friday-{idx}",
        sys_prompt="You are a helpful assistant named Friday.",
        model=DashScopeChatModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model_name=DEFAULT_MODEL_NAME,
            enable_thinking=False,
            stream=True,
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
        search_mcp_client=tavily_client,
        tmp_file_storage_dir=agent_working_dir,
        enable_parallel=True,
        enable_search_cache=use_cache,
    )

    msg = Msg(f"User-{idx}", content=query, role="user")

    start = time.time()
    try:
        await asyncio.wait_for(agent(msg), timeout=timeout)
        ok = True
        err_msg = ""
    except Exception as err:  # pragma: no cover - benchmark 辅助代码
        ok = False
        err_msg = str(err)
    end = time.time()

    return {
        "index": idx,
        "ok": ok,
        "error": err_msg,
        "elapsed": end - start,
        "query": query,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Run multiple DeepResearchAgent queries concurrently, with or without Tavily cache.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="",
        help="Path to a text file containing one query per line.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Limit concurrent tasks (0 means no explicit limit).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Tavily search/extract cache for this run.",
    )
    args = parser.parse_args()

    if not os.environ.get("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY environment variable is not set.")
        print("Please run: export TAVILY_API_KEY='your_key'")
        sys.exit(1)

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY environment variable is not set.")
        print("Please run: export DASHSCOPE_API_KEY='your_key'")
        sys.exit(1)

    if args.queries_file:
        if not os.path.exists(args.queries_file):
            print(f"ERROR: queries file not found: {args.queries_file}")
            sys.exit(1)
        queries = _load_queries_from_file(args.queries_file)
    else:
        queries = _default_queries()

    if not queries:
        print("No queries to run.")
        return

    print("Multi-Query Benchmark")
    print("-" * 60)
    print(f"Total queries: {len(queries)}")
    print(f"Cache enabled: {not args.no_cache}")

    tavily_search_client = await get_tavily_client()
    agent_working_dir = os.getenv("AGENT_OPERATION_DIR", DEFAULT_OPERATION_DIR)
    os.makedirs(agent_working_dir, exist_ok=True)

    try:
        start_batch = time.time()

        tasks = []
        for idx, q in enumerate(queries, start=1):
            tasks.append(
                _run_single_query(
                    idx=idx,
                    query=q,
                    tavily_client=tavily_search_client,
                    agent_working_dir=agent_working_dir,
                    use_cache=not args.no_cache,
                ),
            )

        if args.max_concurrency and args.max_concurrency > 0:
            # 简单的并发控制：分批次 gather
            results: List[Dict[str, Any]] = []
            for i in range(0, len(tasks), args.max_concurrency):
                batch = tasks[i : i + args.max_concurrency]
                batch_res = await asyncio.gather(*batch, return_exceptions=False)
                results.extend(batch_res)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=False)

        end_batch = time.time()

        print("\nPer-query results:")
        for r in results:
            status = "OK" if r["ok"] else f"FAIL ({r['error']})"
            print(
                f"  [{r['index']:02d}] {status} - {r['elapsed']:.2f}s "
                f"- {r['query']}",
            )

        print("\n" + "=" * 30)
        print("MULTI-QUERY BENCHMARK SUMMARY")
        print("=" * 30)
        total_elapsed = end_batch - start_batch
        print(f"Total wall-clock time : {total_elapsed:.2f} s")

        ok_runs = [r for r in results if r["ok"]]
        if ok_runs:
            avg_time = sum(r["elapsed"] for r in ok_runs) / len(ok_runs)
            print(f"Avg per-query time    : {avg_time:.2f} s (successful runs)")
        else:
            print("Avg per-query time    : N/A (all failed)")

        stats = get_global_tavily_cache_stats()
        print("\nTavily cache stats:")
        print(f"  hits      : {stats.get('hits', 0)}")
        print(f"  misses    : {stats.get('misses', 0)}")
        print(f"  co_waiters: {stats.get('co_waiters', 0)}")
        print("=" * 30)

    finally:
        await tavily_search_client.close()


if __name__ == "__main__":
    asyncio.run(main())


