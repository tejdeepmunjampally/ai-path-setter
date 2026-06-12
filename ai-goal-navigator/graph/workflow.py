"""LangGraph workflow wiring for the AI Goal Navigator."""

from langgraph.graph import END, StateGraph, START

from agents.goal_analyzer import goal_analyzer
from agents.planner import planner
from agents.resource_finder import resource_agent
from agents.risk_analyzer import risk_agent
from agents.roadmap_generator import roadmap_agent
from state import NavigatorState

workflow = StateGraph(NavigatorState)

workflow.add_node("goal_step", goal_analyzer)
workflow.add_node("planner_step", planner)
workflow.add_node("resources_step", resource_agent)
workflow.add_node("risks_step", risk_agent)
workflow.add_node("roadmap_step", roadmap_agent)

# Set multiple parallel entry points from START
workflow.add_edge(START, "goal_step")
workflow.add_edge(START, "planner_step")
workflow.add_edge(START, "resources_step")
workflow.add_edge(START, "risks_step")

# Combine results of the parallel agents into the roadmap_step
workflow.add_edge("goal_step", "roadmap_step")
workflow.add_edge("planner_step", "roadmap_step")
workflow.add_edge("resources_step", "roadmap_step")
workflow.add_edge("risks_step", "roadmap_step")

workflow.add_edge("roadmap_step", END)

graph = workflow.compile()

