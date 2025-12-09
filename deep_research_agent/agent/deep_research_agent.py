# -*- coding: utf-8 -*-
"""Deep Research Agent"""
# pylint: disable=too-many-lines, no-name-in-module
import os
import json
import asyncio
import random
import time

from typing import Type, Optional, Any, Tuple
from datetime import datetime
from copy import deepcopy
import shortuuid
from pydantic import BaseModel

from deep_research_agent.agent.prompt.promptmodule import (
    SubtasksDecomposition,
    WebExtraction,
    FollowupJudge,
    ReflectFailure,
    SemanticPlan,
)
from deep_research_agent.core.utils import (
    truncate_search_result,
    load_prompt_dict,
    get_dynamic_tool_call_json,
    get_structure_output,
)
from deep_research_agent.core.cache import GLOBAL_TAVILY_CACHE, make_search_key
from deep_research_agent.core.rag import SimpleRAGStore

from agentscope import logger
from agentscope.mcp import StatefulClientBase
from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase
from agentscope.tool import (
    ToolResponse,
    view_text_file,
    write_text_file,
)
from agentscope.message import (
    Msg,
    ToolUseBlock,
    TextBlock,
    ToolResultBlock,
)


_DEEP_RESEARCH_AGENT_DEFAULT_SYS_PROMPT = "You're a helpful assistant."


class SubTaskItem(BaseModel):
    """Subtask item of deep research agent."""

    objective: str
    working_plan: Optional[str] = None
    knowledge_gaps: Optional[str] = None


class DeepResearchAgent(ReActAgent):
    """
    Deep Research Agent for sophisticated research tasks.
    """

    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        memory: MemoryBase,
        search_mcp_client: StatefulClientBase,
        sys_prompt: str = _DEEP_RESEARCH_AGENT_DEFAULT_SYS_PROMPT,
        max_iters: int = 30,
        max_depth: int = 3,
        tmp_file_storage_dir: str = "tmp",
        enable_parallel: bool = True,
        enable_search_cache: bool = True,
        rag_store: Optional["SimpleRAGStore"] = None, 
        enable_rag: bool = False, 

    ) -> None:
        """Initialize the Deep Research Agent."""
        self.enable_parallel = enable_parallel
        self.enable_search_cache = enable_search_cache
        # initialization of prompts
        self.prompt_dict = load_prompt_dict()

        # Enhance the system prompt for deep research agent
        add_note = self.prompt_dict["add_note"].format_map(
            {"finish_function_name": f"`{self.finish_function_name}`"},
        )
        tool_use_rule = self.prompt_dict["tool_use_rule"].format_map(
            {"tmp_file_storage_dir": tmp_file_storage_dir},
        )
        sys_prompt = f"{sys_prompt}\n{add_note}\n{tool_use_rule}"

        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            memory=memory,
            max_iters=max_iters,
        )
        self.max_depth = max_depth
        self.memory = memory
        self.tmp_file_storage_dir = tmp_file_storage_dir
        self.current_subtask = []

        # register all necessary tools for deep research agent
        self.toolkit.register_tool_function(view_text_file)
        self.toolkit.register_tool_function(write_text_file)
        asyncio.get_running_loop().create_task(
            self.toolkit.register_mcp_client(search_mcp_client),
        )

        self.search_function = "tavily-search"
        self.extract_function = "tavily-extract"
        self.read_file_function = "view_text_file"
        self.write_file_function = "write_text_file"
        self.summarize_function = "summarize_intermediate_results"

        self.intermediate_memory = []
        self.report_path_based = f"{self.name}_{datetime.now():%Y%m%d%H%M%S}"
        self.report_index = 1
        self._required_structured_model = None
        self.user_query = None

        # add functions into toolkit
        self.toolkit.register_tool_function(self.reflect_failure)
        self.toolkit.register_tool_function(
            self.summarize_intermediate_results,
        )

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """The reply method of the agent."""
        # Maintain the subtask list
        self.user_query = msg.get_text_content()
        self.current_subtask.append(
            SubTaskItem(objective=self.user_query),
        )

        # Identify the expected output and generate a plan
        await self.decompose_and_expand_subtask()
        # Experimental: analyze semantic dependency structure for potential parallel subtasks
        try:
            await self._analyze_semantic_parallel_plan()
        except Exception as exc:  # pragma: no cover - debug helper
            logger.warning("Semantic parallel analysis failed: %s", exc)
            # 追加预期输出信息，注意 current_subtask 可能在异常路径中被清空
            root_subtask = self.current_subtask[0] if self.current_subtask else None
            expected = root_subtask.knowledge_gaps if root_subtask else ""
            msg.content += f"\nExpected Output:\n{expected}"

        # Add user query message to memory
        await self.memory.add(msg)  # type: ignore

        # Record structured output model if provided
        if structured_model:
            self._required_structured_model = structured_model
            self.toolkit.set_extended_model(
                self.finish_function_name,
                structured_model,
            )

        for _ in range(self.max_iters):
            # Generate the working plan first
            if not self.current_subtask[-1].working_plan:
                await self.decompose_and_expand_subtask()

            # Write the instruction for reasoning
            cur_plan = self.current_subtask[-1].working_plan
            cur_know_gap = self.current_subtask[-1].knowledge_gaps
            reasoning_prompt = self.prompt_dict["reasoning_prompt"].format_map(
                {
                    "objective": self.current_subtask[-1].objective,
                    "plan": cur_plan
                    if cur_plan
                    else "There is no working plan now.",
                    "knowledge_gap": f"## Knowledge Gaps:\n {cur_know_gap}"
                    if cur_know_gap
                    else "",
                    "depth": len(self.current_subtask),
                },
            )
            reasoning_prompt_msg = Msg(
                "user",
                content=[
                    TextBlock(
                        type="text",
                        text=reasoning_prompt,
                    ),
                ],
                role="user",
            )
            self.intermediate_memory.append(reasoning_prompt_msg)

            # Reasoning to generate tool calls
            backup_memory = deepcopy(self.memory)  # type: ignore
            await self.memory.add(self.intermediate_memory)  # type: ignore
            msg_reasoning = await self._reasoning()
            self.memory = backup_memory

            # Calling the tools
            tool_calls = msg_reasoning.get_content_blocks("tool_use")
            if len(tool_calls) > 0:
                # 1. Log intention for all tool calls first
            #    for tool_call in tool_calls:
            #        self.intermediate_memory.append(
            #            Msg(
            #                self.name,
            #                content=[tool_call],
            #                role="assistant",
            #            ),
            #        )
                self.intermediate_memory.append(
                        Msg(
                            self.name,
                            content=tool_calls,
                            role="assistant",
                        ),
                    )
                
                results = []
                if self.enable_parallel:
                    # 2a. Execute tool calls in parallel
                    logger.info(f"--- Executing {len(tool_calls)} tool calls in PARALLEL ---")
                    tasks = [self._acting(tool_call) for tool_call in tool_calls]
                    results = await asyncio.gather(*tasks, return_exceptions=True) 
                else:
                    # 2b. Execute tool calls serially
                    logger.info(f"--- Executing {len(tool_calls)} tool calls SERIALLY ---")
                    for tool_call in tool_calls:
                        res = await self._acting(tool_call)
                        results.append(res)
                        # In serial mode, if one action returns a finish signal, we might stop early,
                        # but to keep comparison fair with parallel (which runs all), we act all or logic might need adjustment.
                        # For DeepResearch, usually subtasks are independent searches, so running all is fine.
                        if res: 
                            # If a tool returns a Msg (like finish), we return immediately in serial mode
                            # matching original behavior
                            await self.memory.add(res)
                            self.current_subtask = []
                            return res

                # 3. Process results (mostly for Parallel mode, or Serial if no early return happened)
                for msg_response in results:
                    if msg_response:
                        await self.memory.add(msg_response)
                        self.current_subtask = []
                        return msg_response

        # When the maximum iterations are reached, summarize all the findings
        return await self._summarizing()

    async def _analyze_semantic_parallel_plan(self) -> None:
        """Experimental: ask the model to output a semantic dependency DAG of subtasks.

        This does NOT affect core logic; it only prints and (optionally) executes
        a suggested parallelization structure based on the current working plan /
        knowledge gaps.
        """
        if not self.current_subtask:
            return
        cur = self.current_subtask[-1]
        if not cur.working_plan:
            return

        sys_prompt = (
            "You are an expert task planner. Given a deep research working plan "
            "and knowledge gaps, you must rewrite them as a set of ATOMIC subtasks "
            "with explicit dependencies, suitable for parallel execution scheduling. "
            "Subtasks that have no dependencies and only require external web "
            "search can be executed fully in parallel."
        )
        user_inst = (
            "## Working Plan\n"
            f"{cur.working_plan}\n\n"
            "## Knowledge Gaps\n"
            f"{cur.knowledge_gaps}\n\n"
            "Now: break this into atomic subtasks and fill the SemanticPlan "
            "schema. Make sure ids are 1..N and dependencies form a DAG."
        )

        result = await self.get_model_output(
            msgs=[
                Msg("system", sys_prompt, "system"),
                Msg("user", user_inst, "user"),
            ],
            format_template=SemanticPlan,
            stream=self.model.stream,
        )

        subtasks = result.get("subtasks", []) if isinstance(result, dict) else []
        if not subtasks:
            logger.info("No semantic subtasks returned for parallel analysis.")
            return

        print("\n==================== Semantic Parallel Plan (Experimental) ====================")
        for node in subtasks:
            node_id = node.get("id")
            desc = node.get("description")
            deps = node.get("depends_on", [])
            print(f"- Subtask #{node_id}: {desc}")
            print(f"  depends_on: {deps}")
        print("===============================================================================\n")

        # Experimental: actually execute the first few layers of independent subtasks
        try:
            await self._execute_semantic_plan(result)
        except Exception as exc:  # pragma: no cover - debug helper
            logger.warning("Semantic parallel execution failed: %s", exc)

    async def _execute_semantic_plan(self, plan: dict) -> None:
        """Experimental: execute SemanticPlan layer by layer using parallel Tavily search.

        This function is side-channel only: it does NOT touch self.memory /
        self.current_subtask to avoid interfering with the main ReAct loop.
        It is intended to empirically measure/observe layered parallelism.
        """
        nodes = plan.get("subtasks", []) if isinstance(plan, dict) else []
        if not nodes:
            return

        id_to_node: dict[int, dict] = {}
        for node in nodes:
            try:
                nid = int(node.get("id"))
            except Exception:  # pragma: no cover
                continue
            id_to_node[nid] = node

        completed: set[int] = set()
        layer_idx = 1

        while len(completed) < len(id_to_node):
            ready: list[dict] = []
            for nid, node in id_to_node.items():
                if nid in completed:
                    continue
                deps = node.get("depends_on", []) or []
                if all(int(d) in completed for d in deps):
                    ready.append(node)

            if not ready:
                break  # cyclic or malformed plan

            # 只执行前几层的纯“检索型”子任务，避免无限扩展
            if layer_idx > 3:
                logger.info(
                    "Semantic plan execution: stop after %d layers to avoid over-search.",
                    layer_idx - 1,
                )
                break

            print(f"\n[Semantic Execution] Layer {layer_idx} - {len(ready)} subtasks (parallel)")
            queries: list[tuple[int, str]] = []
            for node in ready:
                nid = int(node.get("id"))
                desc = node.get("description", "")
                if not desc:
                    continue
                # 简单做法：直接用描述作为 Tavily 的查询
                queries.append((nid, desc))
                print(f"  - #{nid}: {desc}")

            if not queries:
                break

            async def _run_one(nid: int, desc: str) -> tuple[int, float]:
                """Run one semantic subtask search and measure its own latency."""
                params = {
                    "query": desc,
                    "max_results": 3,
                    "topic": "general",
                }
                t0 = time.perf_counter()
                try:
                    await self.call_specific_tool(self.search_function, params)
                except Exception as exc:  # pragma: no cover - debug helper
                    logger.warning("Semantic subtask #%d failed: %s", nid, exc)
                t1 = time.perf_counter()
                return nid, t1 - t0

            # 并行执行本层所有子任务，同时记录每个子任务自身耗时
            layer_start = time.perf_counter()
            tasks = [_run_one(nid, desc) for nid, desc in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            layer_end = time.perf_counter()

            per_task_times: list[float] = []
            for res in results:
                if isinstance(res, tuple) and len(res) == 2:
                    _, dt = res
                    per_task_times.append(dt)

            parallel_time = layer_end - layer_start
            serial_estimate = sum(per_task_times) if per_task_times else 0.0

            print(
                f"[Semantic Execution] Layer {layer_idx} parallel time : "
                f"{parallel_time:.2f} s"
            )
            if serial_estimate > 0:
                print(
                    f"[Semantic Execution] Layer {layer_idx} serial estimate: "
                    f"{serial_estimate:.2f} s "
                    f"(speedup ~ {serial_estimate / max(parallel_time, 1e-6):.2f}x)"
                )

            # 标记本层节点为完成（即使部分失败，我们这里只关心结构）
            for node in ready:
                completed.add(int(node.get("id")))

            layer_idx += 1

    async def _acting(self, tool_call: ToolUseBlock) -> Msg | None:
        """
        Execute a tool call and process its response.

        对 Tavily 搜索 / 抽取工具增加进程内缓存，其它工具保持原有行为。
        """
        # Add small random jitter for parallel execution to avoid rate limits
        if getattr(self, "enable_parallel", False):
            await asyncio.sleep(random.uniform(0.5, 2.0))

        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        update_memory = False
        intermediate_report = ""
        chunk = None  # 安全：避免 finally 中引用未定义
        chunks_from_cache: list[Any] | None = None

        try:
            tool_name = tool_call["name"]


            # Check the rag first before actual searching if 
            if (
                self.enable_rag 
                and tool_call["name"] == self.search_function
                and self.rag_store is not None
            ):
                query = tool_call.get("input", {}).get("query", "")
                cached_result = self.rag_store.get_or_search(query, threshold=0.6)
                if cached_result:
                    logger.info(f"RAG hit for query: {query[:50]}...")
                    # 直接返回 RAG 结果，跳过实际搜索
                    tool_res_msg.content[0]["output"] = [
                        {"type": "text", "text": f"[From RAG Cache]\n{cached_result}"}
                    ]
                    self.intermediate_memory.append(tool_res_msg)
                    return None


            use_cache = self.enable_search_cache and tool_name in [
                getattr(self, "search_function", ""),
                getattr(self, "extract_function", ""),
            ]

            if use_cache:
                cache_key = make_search_key(
                    tool_name,
                    tool_call.get("input", {}),
                    None,
                )

                async def _run_and_collect() -> list[Any]:
                    collected: list[Any] = []
                    tool_res_inner = await self.toolkit.call_tool_function(
                        tool_call,
                    )
                    async for inner_chunk in tool_res_inner:
                        collected.append(inner_chunk)
                    return collected

                chunks_from_cache = await GLOBAL_TAVILY_CACHE.get_or_await(
                    cache_key,
                    _run_and_collect,
                )
                iterator = chunks_from_cache or []
                is_async_iter = False
            else:
                tool_res = await self.toolkit.call_tool_function(tool_call)
                iterator = tool_res
                is_async_iter = True

            if is_async_iter:
                async for chunk in iterator:  # type: ignore[union-attr]
                    await self._process_tool_chunk(
                        tool_call,
                        tool_res_msg,
                        chunk,
                    )
                    if (
                        tool_call["name"] == self.finish_function_name
                        and chunk.metadata.get("success", True)
                        and len(self.current_subtask) == 0
                    ):
                        return chunk.metadata.get("response_msg")
            else:
                for chunk in iterator:
                    await self._process_tool_chunk(
                        tool_call,
                        tool_res_msg,
                        chunk,
                    )
                    if (
                        tool_call["name"] == self.finish_function_name
                        and chunk.metadata.get("success", True)
                        and len(self.current_subtask) == 0
                    ):
                        return chunk.metadata.get("response_msg")

            # Update memory flags from the last chunk if needed
            if chunk and isinstance(chunk.metadata, dict) and chunk.metadata.get(
                "update_memory",
            ):
                update_memory = True
                intermediate_report = chunk.metadata.get("intermediate_report")

            return None

        finally:
            
            # record the rag if the query is useful
            if (
                self.enable_rag 
                and tool_call["name"] == self.search_function
                and self.rag_store is not None
                and chunk is not None
            ):
                query = tool_call.get("input", {}).get("query", "")
                result_text = str(chunk.content) if chunk else ""
                if result_text and "error" not in result_text.lower():
                    self.rag_store.add(
                        content=result_text,
                        metadata={"query": query, "source": "tavily-search"},
                    )
                
            # Record the tool result message in the intermediate memory
            if tool_call["name"] != self.summarize_function:
                self.intermediate_memory.append(tool_res_msg)

            # Read more information from the web page if necessary
            if tool_call["name"] == self.search_function:
                if chunk is None and chunks_from_cache:
                    chunk = chunks_from_cache[-1]
                search_results = chunk.content if chunk else []
                extract_res = await self._follow_up(search_results, tool_call)
                if isinstance(extract_res.metadata, dict) and extract_res.metadata.get(
                    "update_memory",
                ):
                    self.intermediate_memory = []

                    # 安全地从 intermediate_report 中取文本，避免空列表越界
                    interm = extract_res.metadata.get("intermediate_report")
                    interm_text = ""
                    if hasattr(interm, "content") and isinstance(
                        getattr(interm, "content"),
                        list,
                    ):
                        content_list = getattr(interm, "content")
                        if content_list:
                            first = content_list[0]
                            if isinstance(first, dict):
                                interm_text = first.get("text", "") or ""
                    if not interm_text:
                        # 退化为字符串形式，避免抛错
                        interm_text = str(getattr(interm, "content", interm))

                    await self.memory.add(
                        Msg(
                            "assistant",
                            content=[
                                TextBlock(
                                    type="text",
                                    text=interm_text,
                                ),
                            ],
                            role="assistant",
                        ),
                    )

            # Update memory with the intermediate report
            if update_memory and intermediate_report:
                self.intermediate_memory = []
                # 同样防止 intermediate_report.content 为空
                interm_text = ""
                if hasattr(intermediate_report, "content") and isinstance(
                    intermediate_report.content,
                    list,
                ):
                    if intermediate_report.content:
                        first = intermediate_report.content[0]
                        if isinstance(first, dict):
                            interm_text = first.get("text", "") or ""
                if not interm_text:
                    interm_text = str(getattr(intermediate_report, "content", intermediate_report))

                await self.memory.add(
                    Msg(
                        "assistant",
                        content=[
                            TextBlock(
                                type="text",
                                text=interm_text,
                            ),
                        ],
                        role="assistant",
                    ),
                )

    async def _process_tool_chunk(
        self,
        tool_call: ToolUseBlock,
        tool_res_msg: Msg,
        chunk: Any,
    ) -> None:
        """处理单个工具返回 chunk 的通用逻辑。"""
        # Turn into a tool result block
        tool_res_msg.content[0]["output"] = chunk.content  # type: ignore[index]

        # Skip the printing of the finish function call
        if (
            tool_call["name"] != self.finish_function_name
            or tool_call["name"] == self.finish_function_name
            and not chunk.metadata.get("success")
        ):
            # For display only: truncate output to avoid flooding the console
            display_msg = deepcopy(tool_res_msg)
            raw_output = display_msg.content[0]["output"]
            if isinstance(raw_output, list) and len(raw_output) > 0:
                text = raw_output[0].get("text", "")
                if len(text) > 100:
                    display_msg.content[0]["output"][0]["text"] = (
                        text[:100] + "... (truncated for display)"
                    )

            await self.print(display_msg, chunk.is_last)

        # Summarize intermediate results into a draft report
        if tool_call["name"] == self.summarize_function:
            self.intermediate_memory = []

            # 安全提取 chunk 内文本，避免 content 为空导致越界
            text = ""
            if isinstance(chunk.content, list) and chunk.content:
                first = chunk.content[0]
                if isinstance(first, dict):
                    text = first.get("text", "") or ""
            if not text:
                text = str(getattr(chunk, "content", chunk))

            await self.memory.add(
                Msg(
                    "assistant",
                    [
                        TextBlock(
                            type="text",
                            text=text,
                        ),
                    ],
                    "assistant",
                ),
            )

        # Truncate the web extract results that exceeds max length
        elif tool_call["name"] in [
            self.search_function,
            self.extract_function,
        ]:
            tool_res_msg.content[0]["output"] = truncate_search_result(
                tool_res_msg.content[0]["output"],
            )

    async def get_model_output(
        self,
        msgs: list,
        format_template: Type[BaseModel] = None,
        stream: bool = True,
    ) -> Any:
        """
        Call the model and get output with or without a structured format.
        """
        blocks = None
        if format_template:
            res = await self.model(
                await self.formatter.format(msgs=msgs),
                tools=get_dynamic_tool_call_json(
                    format_template,
                ),
            )

            if stream:
                async for content_chunk in res:
                    blocks = content_chunk.content
            else:
                blocks = res.content

            return get_structure_output(blocks)
        else:
            res = await self.model(
                await self.formatter.format(msgs=msgs),
            )

            if stream:
                async for content_chunk in res:
                    blocks = content_chunk.content
            else:
                blocks = res.content
            return blocks

    async def call_specific_tool(
        self,
        func_name: str,
        params: dict = None,
    ) -> Tuple[Msg, Msg]:
        """
        Call the specific tool in toolkit.
        """
        tool_call = ToolUseBlock(
            id=shortuuid.uuid(),
            type="tool_use",
            name=func_name,
            input=params,
        )
        tool_call_msg = Msg(
            "assistant",
            [tool_call],
            role="assistant",
        )

        # get tool acting res
        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        # 对 Tavily 搜索 / 抽取在此路径也应用缓存（例如 _follow_up 中的 extract）
        use_cache = self.enable_search_cache and func_name in [
            getattr(self, "search_function", ""),
            getattr(self, "extract_function", ""),
        ]

        if use_cache:
            cache_key = make_search_key(func_name, params or {}, None)

            async def _run_and_collect() -> list[Any]:
                collected: list[Any] = []
                tool_res_inner = await self.toolkit.call_tool_function(
                    tool_call,
                )
                async for inner_chunk in tool_res_inner:
                    collected.append(inner_chunk)
                return collected

            chunks = await GLOBAL_TAVILY_CACHE.get_or_await(
                cache_key,
                _run_and_collect,
            )
            if chunks:
                # 这里只需要最终输出内容，取最后一个 chunk 即可
                tool_res_msg.content[0]["output"] = chunks[-1].content
        else:
            tool_res = await self.toolkit.call_tool_function(
                tool_call,
            )
            async for chunk in tool_res:
                tool_res_msg.content[0]["output"] = chunk.content

        return tool_call_msg, tool_res_msg

    async def decompose_and_expand_subtask(self) -> ToolResponse:
        """Identify knowledge gaps and generate working plan."""
        if len(self.current_subtask) <= self.max_depth:
            decompose_sys_prompt = self.prompt_dict["decompose_sys_prompt"]

            previous_plan = ""
            for i, subtask in enumerate(self.current_subtask):
                previous_plan += f"The {i}-th plan: {subtask.working_plan}\n"
            previous_plan_inst = self.prompt_dict[
                "previous_plan_inst"
            ].format_map(
                {
                    "previous_plan": previous_plan,
                    "objective": self.current_subtask[-1].objective,
                },
            )

            try:
                gaps_and_plan = await self.get_model_output(
                    msgs=[
                        Msg("system", decompose_sys_prompt, "system"),
                        Msg("user", previous_plan_inst, "user"),
                    ],
                    format_template=SubtasksDecomposition,
                    stream=self.model.stream,
                )
                print(f"\n{'='*20} Subtask Decomposition {'='*20}")
                print(f"Current Objective: {self.current_subtask[-1].objective}")
                print(f"Working Plan: {gaps_and_plan.get('working_plan')}")
                print(f"Knowledge Gaps: {gaps_and_plan.get('knowledge_gaps')}")
                print(f"{'='*63}\n")

                response = json.dumps(
                    gaps_and_plan,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:  # noqa: F841
                gaps_and_plan = {}
                response = self.prompt_dict["retry_hint"].format_map(
                    {"state": "decomposing the subtask"},
                )
            self.current_subtask[-1].knowledge_gaps = gaps_and_plan.get(
                "knowledge_gaps",
                None,
            )
            self.current_subtask[-1].working_plan = gaps_and_plan.get(
                "working_plan",
                None,
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=response,
                    ),
                ],
            )
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=self.prompt_dict["max_depth_hint"],
                ),
            ],
        )

    async def _follow_up(
        self,
        search_results: list | str,
        tool_call: ToolUseBlock,
    ) -> ToolResponse:
        """Read the website more intensively to mine more information."""
        if len(self.current_subtask) < self.max_depth:
            # Step#1: query expansion
            expansion_sys_prompt = self.prompt_dict["expansion_sys_prompt"]
            expansion_inst = self.prompt_dict["expansion_inst"].format_map(
                {
                    "objective": tool_call["input"].get("query", ""),
                    "checklist": self.current_subtask[0].knowledge_gaps,
                    "knowledge_gaps": self.current_subtask[-1].working_plan,
                    "search_results": search_results,
                },
            )

            try:
                follow_up_subtask = await self.get_model_output(
                    msgs=[
                        Msg("system", expansion_sys_prompt, "system"),
                        Msg("user", expansion_inst, "user"),
                    ],
                    format_template=WebExtraction,
                    stream=self.model.stream,
                )
            except Exception:  # noqa: F841
                follow_up_subtask = {}

            #  Step #2: extract the url
            if follow_up_subtask.get("need_more_information", False):
                expansion_response_msg = Msg(
                    "assistant",
                    follow_up_subtask.get(
                        "reasoning",
                        "I need more information.",
                    ),
                    role="assistant",
                )
                urls = follow_up_subtask.get("url", None)
                logger.info("Reading %s", urls)

                # call the extract_function
                params = {
                    "urls": urls,
                    "extract_depth": "basic",
                }
                (
                    extract_tool_use_msg,
                    extract_tool_res_msg,
                ) = await self.call_specific_tool(
                    func_name=self.extract_function,
                    params=params,
                )
                self.intermediate_memory.append(extract_tool_use_msg)

                extract_tool_res_msg.content[0]["output"] = truncate_search_result(
                    extract_tool_res_msg.content[0]["output"],
                )
                await self.print(extract_tool_res_msg, True)
                self.intermediate_memory.append(extract_tool_res_msg)

                # Step #4: follow up judge
                try:
                    follow_up_response = await self.get_model_output(
                        msgs=[
                            Msg("user", expansion_inst, "user"),
                            expansion_response_msg,
                            extract_tool_use_msg,
                            extract_tool_res_msg,
                            Msg(
                                "user",
                                self.prompt_dict["follow_up_judge_sys_prompt"],
                                role="user",
                            ),
                        ],
                        format_template=FollowupJudge,
                        stream=self.model.stream,
                    )
                except Exception:  # noqa: F841
                    follow_up_response = {}
                if not follow_up_response.get("is_sufficient", True):
                    subtasks = follow_up_subtask.get("subtask", None)
                    logger.info("Figuring out %s", subtasks)
                    intermediate_report = (
                        await self.summarize_intermediate_results()
                    )
                    self.current_subtask.append(
                        SubTaskItem(objective=subtasks),
                    )
                    return ToolResponse(
                        content=[
                            TextBlock(
                                type="text",
                                text=follow_up_response.get(
                                    "reasoning",
                                    self.prompt_dict["need_deeper_hint"],
                                ),
                            ),
                        ],
                        metadata={
                            "update_memory": True,
                            "intermediate_report": intermediate_report,
                        },
                    )
                else:
                    return ToolResponse(
                        content=[
                            TextBlock(
                                type="text",
                                text=follow_up_response.get(
                                    "reasoning",
                                    self.prompt_dict["sufficient_hint"],
                                ),
                            ),
                        ],
                    )
            else:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=follow_up_subtask.get(
                                "reasoning",
                                self.prompt_dict["sufficient_hint"],
                            ),
                        ),
                    ],
                )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["max_depth_hint"],
                    ),
                ],
            )

    async def summarize_intermediate_results(self) -> ToolResponse:
        """Summarize the intermediate results into a report."""
        if len(self.intermediate_memory) == 0:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["no_result_hint"],
                    ),
                ],
            )
        # agent actively call this tool
        if self.intermediate_memory[-1].name == self.summarize_function:
            blocks = await self.get_model_output(
                msgs=self.intermediate_memory
                + [
                    Msg(
                        "user",
                        self.prompt_dict["summarize_hint"].format_map(
                            {
                                "plan": self.current_subtask[-1].working_plan,
                            },
                        ),
                        role="user",
                    ),
                ],
                stream=self.model.stream,
            )
            # 防御性更新 working_plan，避免 blocks 为空或结构异常
            if isinstance(blocks, list) and blocks:
                first = blocks[0]
                if isinstance(first, dict):
                    new_plan = first.get("text")
                    if isinstance(new_plan, str) and new_plan.strip():
                        self.current_subtask[-1].working_plan = new_plan
        report_prefix = "#" * len(self.current_subtask)
        summarize_sys_prompt = self.prompt_dict[
            "summarize_sys_prompt"
        ].format_map(
            {"report_prefix": report_prefix},
        )
        # get all tool result
        tool_result = ""
        for item in self.intermediate_memory:
            if isinstance(item.content, str):
                tool_result += item.content + "\n"
            elif isinstance(item.content, list):
                for each in item.content:
                    if each["type"] == "tool_result":
                        tool_result += str(each) + "\n"
            else:
                logger.warning(
                    "Unknown content type: %s!",
                    type(item.content),
                )
                continue
        root_subtask = self.current_subtask[0] if self.current_subtask else None
        summarize_instruction = self.prompt_dict["summarize_inst"].format_map(
            {
                "objective": getattr(root_subtask, "objective", "") if root_subtask else "",
                "knowledge_gaps": getattr(root_subtask, "knowledge_gaps", "") if root_subtask else "",
                "working_plan": self.current_subtask[-1].working_plan if self.current_subtask else "",
                "tool_result": tool_result,
            },
        )

        blocks = await self.get_model_output(
            msgs=[
                Msg("system", summarize_sys_prompt, "system"),
                Msg("user", summarize_instruction, "user"),
            ],
            stream=self.model.stream,
        )
        if blocks and len(blocks) > 0:
            intermediate_report = blocks[0]["text"]  # type: ignore[index]
        else:
            logger.warning("Model returned empty blocks in summarize_intermediate_results")
            intermediate_report = "Failed to generate summary due to empty model response."

        # Write the intermediate report
        intermediate_report_path = os.path.join(
            self.tmp_file_storage_dir,
            f"{self.report_path_based}_"
            f"inprocess_report_{self.report_index}.md",
        )
        self.report_index += 1
        params = {
            "file_path": intermediate_report_path,
            "content": intermediate_report,
        }
        await self.call_specific_tool(
            func_name=self.write_file_function,
            params=params,
        )
        logger.info(
            "Storing the intermediate findings: %s",
            intermediate_report,
        )
        if (
            self.intermediate_memory[-1].has_content_blocks("tool_use")
            and self.intermediate_memory[-1].get_content_blocks("tool_use")[0][
                "name"
            ]
            == self.summarize_function
        ):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["update_report_hint"].format_map(
                            {
                                "intermediate_report": intermediate_report,
                                "report_path": intermediate_report_path,
                            },
                        ),
                    ),
                ],
            )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["save_report_hint"].format_map(
                            {
                                "intermediate_report": intermediate_report,
                            },
                        ),
                    ),
                ],
            )

    async def _generate_deepresearch_report(
        self,
        checklist: str,
    ) -> Tuple[Msg, str]:
        """Collect and polish all draft reports into a final report."""
        reporting_sys_prompt_template = self.prompt_dict["reporting_sys_prompt"]
        reporting_sys_prompt = reporting_sys_prompt_template.format_map(
            {
                "original_task": self.user_query,
                "checklist": checklist,
            },
        )

        # Collect all intermediate reports
        if self.report_index > 1:
            inprocess_report = ""
            # 注意：原逻辑从 1 开始，实际需要读取 1..report_index-1
            for index in range(1, self.report_index):
                params = {
                    "file_path": os.path.join(
                        self.tmp_file_storage_dir,
                        f"{self.report_path_based}_"
                        f"inprocess_report_{index}.md",
                    ),
                }
                _, read_draft_tool_res_msg = await self.call_specific_tool(
                    func_name=self.read_file_function,
                    params=params,
                )
                # 防御性读取文件内容，避免 output 或列表为空
                draft_text = ""
                content_list = read_draft_tool_res_msg.content
                if isinstance(content_list, list) and content_list:
                    first_block = content_list[0]
                    if isinstance(first_block, dict):
                        output = first_block.get("output")
                        if isinstance(output, list) and output:
                            first_out = output[0]
                            if isinstance(first_out, dict):
                                draft_text = first_out.get("text", "") or ""
                        elif isinstance(output, str):
                            draft_text = output
                if not draft_text:
                    draft_text = str(read_draft_tool_res_msg.content)

                inprocess_report += draft_text + "\n"

            msgs = [
                Msg(
                    "system",
                    content=reporting_sys_prompt,
                    role="system",
                ),
                Msg(
                    "user",
                    content=f"Draft report:\n{inprocess_report}",
                    role="user",
                ),
            ]
        else:  # Use only intermediate memory to generate report
            msgs = [
                Msg(
                    "system",
                    content=reporting_sys_prompt,
                    role="system",
                ),
            ] + self.intermediate_memory

        blocks = await self.get_model_output(
            msgs=msgs,
            stream=self.model.stream,
        )
        if isinstance(blocks, list) and blocks:
            first = blocks[0]
            if isinstance(first, dict):
                final_report_content = first.get("text") or ""
            else:
                final_report_content = str(first)
        else:
            logger.warning("Model returned empty blocks in _generate_deepresearch_report")
            final_report_content = "Failed to generate final report due to empty model response."
        logger.info(
            "The final Report is generated: %s",
            final_report_content,
        )

        # Write the final report into a file
        detailed_report_path = os.path.join(
            self.tmp_file_storage_dir,
            f"{self.report_path_based}_detailed_report.md",
        )

        params = {
            "file_path": detailed_report_path,
            "content": final_report_content,
        }
        _, write_report_tool_res_msg = await self.call_specific_tool(
            func_name=self.write_file_function,
            params=params,
        )

        return write_report_tool_res_msg, detailed_report_path

    async def _summarizing(self) -> Msg:
        """Generate a report based on the existing findings when max iters hit."""
        checklist_text = (
            self.current_subtask[0].knowledge_gaps
            if self.current_subtask
            else ""
        )
        (
            summarized_content,
            _,
        ) = await self._generate_deepresearch_report(
            checklist=checklist_text,
        )

        # 防御性提取最终报告内容
        final_block = None
        if (
            isinstance(summarized_content.content, list)
            and summarized_content.content
        ):
            first_block = summarized_content.content[0]
            if isinstance(first_block, dict):
                output = first_block.get("output")
                if isinstance(output, list) and output:
                    final_block = output[0]
        if final_block is None:
            final_block = getattr(summarized_content, "content", {})

        return Msg(
            name=self.name,
            role="assistant",
            content=json.dumps(
                final_block,
                indent=2,
                ensure_ascii=False,
            ),
        )

    async def reflect_failure(self) -> ToolResponse:
        """Reflect on the failure of the action and determine adjustments."""
        reflect_sys_prompt = self.prompt_dict["reflect_sys_prompt"]
        conversation_history = ""
        for msg in self.intermediate_memory:
            conversation_history += (
                json.dumps(
                    {"role": "user", "content": msg.content},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        reflect_inst = self.prompt_dict["reflect_instruction"].format_map(
            {
                "conversation_history": conversation_history,
                "plan": self.current_subtask[-1].working_plan,
            },
        )
        try:
            reflection = await self.get_model_output(
                msgs=[
                    Msg("system", reflect_sys_prompt, "system"),
                    Msg("user", reflect_inst, "user"),
                ],
                format_template=ReflectFailure,
                stream=self.model.stream,
            )
            response = json.dumps(
                reflection,
                indent=2,
                ensure_ascii=False,
            )
        except Exception:  # noqa: F841
            reflection = {}
            response = self.prompt_dict["retry_hint"].format_map(
                {"state": "making the reflection"},
            )

        if reflection.get("rephrase_subtask", False) and reflection[
            "rephrase_subtask"
        ].get(
            "need_rephrase",
            False,
        ):  # type: ignore[index]
            self.current_subtask[-1].working_plan = reflection[
                "rephrase_subtask"
            ][
                "rephrased_plan"
            ]  # type: ignore[index]
        elif reflection.get("decompose_subtask", False) and reflection[
            "decompose_subtask"
        ].get(
            "need_decompose",
            False,
        ):  # type: ignore[index]
            if len(self.current_subtask) <= self.max_depth:
                intermediate_report = (
                    await self.summarize_intermediate_results()
                )
                self.current_subtask.append(
                    SubTaskItem(
                        objective=reflection[
                            "decompose_subtask"
                        ].get(  # type: ignore[index]
                            "failed_subtask",
                            None,
                        ),
                    ),
                )
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=response,
                        ),
                    ],
                    metadata={
                        "update_memory": True,
                        "intermediate_report": intermediate_report,
                    },
                )
            else:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=self.prompt_dict["max_depth_hint"],
                        ),
                    ],
                )
        else:
            pass
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=response,
                ),
            ],
        )

    # pylint: disable=invalid-overridden-method, unused-argument
    async def generate_response(  #
        self,
        response: str,
        **_kwargs: Any,
    ) -> ToolResponse:
        """Generate a detailed report as a response."""
        checklist = (
            self.current_subtask[0].knowledge_gaps
            if self.current_subtask
            else ""
        )
        completed_subtask = self.current_subtask.pop()

        if len(self.current_subtask) == 0:
            (
                summarized_content,
                _,
            ) = await self._generate_deepresearch_report(
                checklist=checklist,
            )

            # 同样防御性提取最终报告块
            final_block = None
            if (
                isinstance(summarized_content.content, list)
                and summarized_content.content
            ):
                first_block = summarized_content.content[0]
                if isinstance(first_block, dict):
                    output = first_block.get("output")
                    if isinstance(output, list) and output:
                        final_block = output[0]
            if final_block is None:
                final_block = getattr(summarized_content, "content", {})

            response_msg = Msg(
                name=self.name,
                role="assistant",
                content=json.dumps(
                    final_block,
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Successfully generated detailed report.",
                    ),
                ],
                metadata={
                    "success": True,
                    "response_msg": response_msg,
                },
                is_last=True,
            )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict[
                            "subtask_complete_hint"
                        ].format_map(
                            {
                                "cur_obj": completed_subtask.objective,
                                "next_obj": self.current_subtask[-1].objective,
                            },
                        ),
                    ),
                ],
                metadata={
                    "success": True,
                },
                is_last=True,
            )