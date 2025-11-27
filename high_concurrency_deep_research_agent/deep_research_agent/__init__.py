# -*- coding: utf-8 -*-
"""Package init for deep_research_agent."""

from .agent.deep_research_agent import DeepResearchAgent  # re-export

def __all__():
    return ["DeepResearchAgent"]