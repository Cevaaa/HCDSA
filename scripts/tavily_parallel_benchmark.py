import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

API_URL = "https://api.tavily.com/search"


def tavily_search(query: str, api_key: str, max_results: int = 3) -> dict:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        # 关闭不必要的信息，减小响应体
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_serial(queries: List[str], api_key: str) -> float:
    start = time.perf_counter()
    for q in queries:
        tavily_search(q, api_key)
    end = time.perf_counter()
    return end - start


def run_parallel(queries: List[str], api_key: str, max_workers: int = 5) -> float:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(tavily_search, q, api_key) for q in queries]
        for f in as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"Parallel query failed: {e}")
    end = time.perf_counter()
    return end - start


def main() -> None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("ERROR: TAVILY_API_KEY not set.")
        print("Please export TAVILY_API_KEY before running this script.")
        return

    # 6 个子查询：3 国 x 2 指标
    queries = [
        "2023 GDP growth rate China",
        "2023 GDP growth rate United States",
        "2023 GDP growth rate Germany",
        "2023 inflation rate China",
        "2023 inflation rate United States",
        "2023 inflation rate Germany",
    ]

    print("Queries to test:")
    for q in queries:
        print(" -", q)
    print("-" * 60)

    print("Running serial Tavily search...")
    t_serial = run_serial(queries, api_key)
    print(f"Serial time   : {t_serial:.2f} s")

    print("\nRunning parallel Tavily search...")
    t_parallel = run_parallel(queries, api_key, max_workers=len(queries))
    print(f"Parallel time : {t_parallel:.2f} s")

    if t_parallel > 0:
        speedup = t_serial / t_parallel
        print("\nSpeedup (Serial / Parallel): {:.2f}x".format(speedup))


if __name__ == "__main__":
    main()
