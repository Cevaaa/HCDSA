#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Benchmark Script - 独立运行，测试 RAG 模块效果

测试内容：
1. RAG 存储和检索性能
2. 模拟多查询场景下的 RAG 命中率
3. 对比有/无 RAG 的实际 Tavily 搜索次数

使用方法：
    export TAVILY_API_KEY='your_key'
    python scripts/rag_benchmark.py [--persist-path path] [--threshold 0.5]
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Tuple

import requests

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入已有的 RAG 模块
from deep_research_agent.core.rag import SimpleRAGStore

# ============================================================================
# Tavily 搜索（模拟实际调用）
# ============================================================================

TAVILY_API_URL = "https://api.tavily.com/search"


def tavily_search(query: str, api_key: str, max_results: int = 3) -> Dict[str, Any]:
    """调用 Tavily API 进行搜索"""
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = requests.post(TAVILY_API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ============================================================================
# Benchmark 逻辑
# ============================================================================

def get_test_queries() -> List[str]:
    """
    测试查询集 - 设计有重叠的查询来测试 RAG 命中
    模拟用户问相似问题的场景
    """
    return [
        # 第一批：基础查询
        "2023 GDP growth rate China",
        "2023 GDP growth rate United States",
        "2023 GDP growth rate Germany",
        "2023 inflation rate China",
        "2023 inflation rate United States",
        "2023 inflation rate Germany",
        
        # 第二批：相似查询（应该命中 RAG）
        "China GDP growth 2023",
        "US GDP growth rate 2023",
        "Germany economic growth 2023",
        "China inflation 2023",
        "United States inflation rate 2023",
        "German inflation 2023",
        
        # 第三批：更多变体
        "What is China's GDP growth in 2023",
        "2023 US economic growth rate",
        "inflation rate in Germany 2023",
    ]


def run_without_rag(
    queries: List[str],
    api_key: str,
) -> Tuple[float, int]:
    """
    无 RAG 模式：每个查询都实际调用 Tavily
    返回：(总耗时, 实际搜索次数)
    """
    search_count = 0
    start = time.perf_counter()
    
    for query in queries:
        try:
            tavily_search(query, api_key)
            search_count += 1
        except Exception as e:
            print(f"  [FAIL] {query[:40]}... - {e}")
    
    end = time.perf_counter()
    return end - start, search_count


def run_with_rag(
    queries: List[str],
    api_key: str,
    rag_store: SimpleRAGStore,
    similarity_threshold: float = 0.5,
) -> Tuple[float, int, int, int]:
    """
    有 RAG 模式：先查 RAG，没命中再调用 Tavily，然后存入 RAG
    返回：(总耗时, 实际搜索次数, RAG命中次数, RAG未命中次数)
    """
    search_count = 0
    rag_hits = 0
    rag_misses = 0
    
    start = time.perf_counter()
    
    for query in queries:
        # 1. 先查 RAG
        cached_result = rag_store.get_or_search(query, threshold=similarity_threshold)
        
        if cached_result:
            # RAG 命中，跳过实际搜索
            rag_hits += 1
            print(f"  [RAG HIT]  {query[:50]}...")
        else:
            # RAG 未命中，执行实际搜索
            rag_misses += 1
            try:
                result = tavily_search(query, api_key)
                search_count += 1
                
                # 将结果存入 RAG
                result_text = str(result.get("results", []))
                rag_store.add(
                    content=result_text,
                    metadata={"query": query, "source": "tavily"},
                )
                print(f"  [SEARCH]   {query[:50]}...")
                
            except Exception as e:
                print(f"  [FAIL]     {query[:40]}... - {e}")
    
    end = time.perf_counter()
    return end - start, search_count, rag_hits, rag_misses


def benchmark_rag_retrieval(
    rag_store: SimpleRAGStore,
    test_queries: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    单独测试 RAG 检索性能（不涉及 Tavily）
    """
    results = {
        "total_queries": len(test_queries),
        "hits": 0,
        "misses": 0,
        "avg_score": 0.0,
        "retrieval_times": [],
    }
    
    scores = []
    
    for query in test_queries:
        start = time.perf_counter()
        search_results = rag_store.search(query, top_k=1)
        end = time.perf_counter()
        
        results["retrieval_times"].append(end - start)
        
        if search_results and search_results[0]["score"] >= threshold:
            results["hits"] += 1
            scores.append(search_results[0]["score"])
        else:
            results["misses"] += 1
    
    if scores:
        results["avg_score"] = sum(scores) / len(scores)
    
    results["avg_retrieval_time_ms"] = (
        sum(results["retrieval_times"]) / len(results["retrieval_times"]) * 1000
        if results["retrieval_times"] else 0
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark - 测试 RAG 模块对搜索效率的提升",
    )
    parser.add_argument(
        "--persist-path",
        type=str,
        default="",
        help="RAG 持久化文件路径（默认使用内存）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="RAG 相似度阈值 (0.0-1.0)",
    )
    parser.add_argument(
        "--skip-tavily",
        action="store_true",
        help="跳过实际 Tavily 调用，只测试 RAG 检索",
    )
    args = parser.parse_args()

    # 检查 API Key
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key and not args.skip_tavily:
        print("ERROR: TAVILY_API_KEY not set.")
        print("Please run: export TAVILY_API_KEY='your_key'")
        print("Or use --skip-tavily to only test RAG retrieval.")
        sys.exit(1)

    queries = get_test_queries()
    
    print("=" * 60)
    print("RAG BENCHMARK")
    print("=" * 60)
    print(f"Total queries     : {len(queries)}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Persist path      : {args.persist_path or '(memory only)'}")
    print("-" * 60)

    # ========== 测试 1: 无 RAG 模式 ==========
    if not args.skip_tavily:
        print("\n[Test 1] Running WITHOUT RAG (all queries hit Tavily)...")
        time_no_rag, searches_no_rag = run_without_rag(queries, api_key)
        print(f"\n  Total time      : {time_no_rag:.2f} s")
        print(f"  Tavily searches : {searches_no_rag}")
    else:
        time_no_rag, searches_no_rag = 0, len(queries)
        print("\n[Test 1] Skipped (--skip-tavily)")

    # ========== 测试 2: 有 RAG 模式 ==========
    print("\n" + "-" * 60)
    print("\n[Test 2] Running WITH RAG (check RAG first, then Tavily)...")
    
    # 创建新的 RAG store
    rag_store = SimpleRAGStore(
        persist_path=args.persist_path if args.persist_path else None,
        similarity_threshold=args.threshold,
    )
    
    if not args.skip_tavily:
        time_with_rag, searches_with_rag, hits, misses = run_with_rag(
            queries, api_key, rag_store, args.threshold
        )
        print(f"\n  Total time      : {time_with_rag:.2f} s")
        print(f"  Tavily searches : {searches_with_rag}")
        print(f"  RAG hits        : {hits}")
        print(f"  RAG misses      : {misses}")
        
        hit_rate = hits / len(queries) * 100 if queries else 0
        print(f"  Hit rate        : {hit_rate:.1f}%")
    else:
        # 只测试 RAG 检索
        # 先填充一些数据
        print("  Populating RAG with sample data...")
        sample_data = [
            ("2023 GDP growth rate China was 5.2%", {"query": "2023 GDP growth rate China"}),
            ("2023 GDP growth rate United States was 2.5%", {"query": "2023 GDP growth rate United States"}),
            ("2023 GDP growth rate Germany was -0.3%", {"query": "2023 GDP growth rate Germany"}),
            ("2023 inflation rate China was 0.2%", {"query": "2023 inflation rate China"}),
            ("2023 inflation rate United States was 3.4%", {"query": "2023 inflation rate United States"}),
            ("2023 inflation rate Germany was 5.9%", {"query": "2023 inflation rate Germany"}),
        ]
        for content, metadata in sample_data:
            rag_store.add(content, metadata)
        
        print(f"  RAG populated with {len(sample_data)} documents")

    # ========== 测试 3: RAG 检索性能 ==========
    print("\n" + "-" * 60)
    print("\n[Test 3] RAG Retrieval Performance...")
    
    retrieval_stats = benchmark_rag_retrieval(rag_store, queries, args.threshold)
    print(f"  Total queries        : {retrieval_stats['total_queries']}")
    print(f"  Hits                 : {retrieval_stats['hits']}")
    print(f"  Misses               : {retrieval_stats['misses']}")
    print(f"  Avg similarity score : {retrieval_stats['avg_score']:.3f}")
    print(f"  Avg retrieval time   : {retrieval_stats['avg_retrieval_time_ms']:.3f} ms")

    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if not args.skip_tavily and searches_no_rag > 0:
        search_reduction = (1 - searches_with_rag / searches_no_rag) * 100
        time_reduction = (1 - time_with_rag / time_no_rag) * 100 if time_no_rag > 0 else 0
        
        print(f"Without RAG:")
        print(f"  - Time           : {time_no_rag:.2f} s")
        print(f"  - Tavily calls   : {searches_no_rag}")
        print(f"\nWith RAG:")
        print(f"  - Time           : {time_with_rag:.2f} s")
        print(f"  - Tavily calls   : {searches_with_rag}")
        print(f"\nImprovement:")
        print(f"  - Search reduction : {search_reduction:.1f}%")
        print(f"  - Time reduction   : {time_reduction:.1f}%")
        
        if time_no_rag > time_with_rag:
            speedup = time_no_rag / time_with_rag
            print(f"  - Speedup          : {speedup:.2f}x")
    
    print(f"\nRAG Store Stats:")
    print(f"  - Documents stored : {len(rag_store.documents)}")
    print(f"  - Vocab size       : {len(rag_store.vocab)}")
    
    if args.persist_path:
        print(f"  - Persisted to     : {args.persist_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()