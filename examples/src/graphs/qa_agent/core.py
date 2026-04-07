from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from pydantic import BaseModel
from ...tools import lightrag_tool, ddgs_tool
from ...agent import Agent
from ._prompts import *

TOOLS = [lightrag_tool, ddgs_tool]

class Params(BaseModel):
    location : str
    year : int
    topic : str
    question : str

class GraphState(TypedDict):
    params : Params
    tools : dict
    result : str

def init(state : GraphState):
    return {
        'params': Params(**state['params']),
        'tools': state.get('tools', TOOLS),
    }

def actor(state : GraphState):
    agent = Agent(system_prompt=QA_PROMPT, tools=[lightrag_tool, ddgs_tool])
    params = state['params']
    message = QA_MESSAGE.format(**params.model_dump())

    return {
        'result': agent.invoke(message)
    }

def create_qa_graph(**compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('actor', actor)

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'actor')
    graph.add_edge('actor', END)

    return graph.compile(**compile_kwargs)