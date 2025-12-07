#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async task cache utilities for Tavily search/extract results.

设计目标：
- 进程内异步任务缓存：对相同 key 的协程只执行一次，其它协程 await 复用结果；
- 支持简单的 TTL、命中统计，便于并发实验观测；
- 尽量保持与业务逻辑解耦，只暴露通用 `AsyncTaskCache` 与 key 生成工具。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional


JsonDict = Dict[str, Any]
CoroutineFactory = Callable[[], Awaitable[Any]]


@dataclass
class _CacheEntry:
    status: str  # "RUNNING" | "COMPLETED"
    event: asyncio.Event
    created_at: float
    result: Any = None
    waiters: int = 0


@dataclass
class CacheStats:
    """简单统计信息，便于 benchmark 打印。"""

    hits: int = 0
    misses: int = 0
    # 统计有多少次“已经在 RUNNING 状态，后来又有协程加入等待”
    co_waiters: int = 0

    def to_dict(self) -> JsonDict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "co_waiters": self.co_waiters,
        }


class AsyncTaskCache:
    """基于 asyncio.Event 的简单异步任务缓存。

    语义：
    - 第一个 `get_or_await(key, coro_factory)` 的调用负责实际执行 `coro_factory()`；
    - 期间若有其它协程以同一 key 调用，会等待同一个 `event`，共享执行结果；
    - 可选 TTL，过期后视为未命中并重新执行。
    """

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    async def get_or_await(
        self,
        key: str,
        coro_factory: CoroutineFactory,
        ttl: Optional[float] = None,
    ) -> Any:
        """获取或等待同 key 的任务结果。"""
        now = time.time()
        ttl_val = self._default_ttl if ttl is None else ttl

        async with self._lock:
            entry = self._entries.get(key)

            # 命中且未过期
            if entry and entry.status == "COMPLETED":
                if ttl_val is None or now - entry.created_at <= ttl_val:
                    self._stats.hits += 1
                    return entry.result
                # 过期，丢弃旧 entry
                self._entries.pop(key, None)
                entry = None

            # 任务已经在运行中：加入等待队列
            if entry and entry.status == "RUNNING":
                entry.waiters += 1
                self._stats.co_waiters += 1
                event = entry.event
            else:
                # 创建新的 entry，由当前协程负责执行
                event = asyncio.Event()
                entry = _CacheEntry(
                    status="RUNNING",
                    event=event,
                    created_at=now,
                    result=None,
                )
                self._entries[key] = entry
                self._stats.misses += 1

                async def _runner() -> None:
                    try:
                        result = await coro_factory()
                    except Exception:
                        # 失败时，为避免所有等待者永久挂起，仍需 set event，并移除缓存
                        async with self._lock:
                            self._entries.pop(key, None)
                            event.set()
                        raise

                    async with self._lock:
                        cur = self._entries.get(key)
                        if cur is not None:
                            cur.status = "COMPLETED"
                            cur.result = result
                            cur.created_at = time.time()
                            event.set()

                # 使用后台任务执行，当前协程统一通过等待 event 获取结果
                asyncio.create_task(_runner())

        # 等待任务完成
        await event.wait()
        async with self._lock:
            final_entry = self._entries.get(key)
            return final_entry.result if final_entry is not None else None


def _normalize_input(input_dict: Optional[JsonDict]) -> JsonDict:
    """对工具输入参数做稳定序列化：排序 key，仅保留与语义强相关字段。

    这里选择保留全部字段，但使用 sort_keys=True 保证 hash 稳定；
    若后续实验希望忽略某些参数（如 `max_results`），可在这里做过滤。
    """
    if not isinstance(input_dict, dict):
        return {}
    return input_dict


def make_search_key(
    tool_name: str,
    input_dict: Optional[JsonDict],
    user_scope: Optional[str] = None,
) -> str:
    """为 Tavily 搜索/抽取生成稳定指纹 key。"""
    payload: JsonDict = {
        "tool": tool_name,
        "input": _normalize_input(input_dict),
    }
    if user_scope:
        payload["user_scope"] = user_scope

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest


# ========== 全局 Tavily 结果缓存 ==========

def _init_global_cache() -> AsyncTaskCache:
    """根据环境变量初始化全局缓存 TTL。"""
    ttl_env = os.getenv("DEEP_RESEARCH_SEARCH_TTL", "").strip()
    ttl_val: Optional[float]
    if ttl_env:
        try:
            ttl_val = float(ttl_env)
        except ValueError:
            ttl_val = None
    else:
        # 默认 10 分钟
        ttl_val = 600.0
    return AsyncTaskCache(default_ttl=ttl_val)


GLOBAL_TAVILY_CACHE: AsyncTaskCache = _init_global_cache()


def get_global_tavily_cache_stats() -> JsonDict:
    """获取全局 Tavily 缓存统计。"""
    return GLOBAL_TAVILY_CACHE.stats.to_dict()


