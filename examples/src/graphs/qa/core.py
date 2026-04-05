from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from operator import add
from ...agent import Agent
from ._prompts import *
from ...tools import lightrag_tool, ddgs_tool

MAX_ITER = 3
TOOLS = [lightrag_tool, ddgs_tool]

class Params(BaseModel):
    location : str
    year : int
    topic : str
    question : str

class GraphState(TypedDict):
    params : Params
    tools : list
    max_iters : int
    curr_iter : int
    messages : Annotated[list[str], add]
    actor : Agent
    critic : Agent

def init(state : GraphState):
    params = Params(**state['params'])
    tools = state.get('tools', TOOLS)
    max_iters = state.get(MAX_ITER, 3)

    actor = Agent(
        system_prompt=ACTOR_PROMPT.format(**params.model_dump()),
        tools=tools,
        is_checkpointer=True
    )
    critic = Agent(
        system_prompt=CRITIC_PROMPT.format(**params.model_dump()),
        tools=tools,
        is_checkpointer=True
    )

    return {
        'params': params,
        'tools': tools,
        'max_iters': max_iters,
        'curr_iter': 0,
        'messages': [],
        'actor': actor,
        'critic': critic
    }

def get_invoke(agent_name : str):

    def invoke(state : GraphState):
        messages = state['messages']
        agent = state[agent_name]
        res = agent.invoke([m for m in messages[-1:]])
        return {
            'messages': [res]
        }
    
    return invoke

def loop(state : GraphState):
    return {
        'curr_iter': state['curr_iter'] + 1
    }

def loop_conditional_edges(state : GraphState):
    max_iters = state['max_iters']
    curr_iter = state['curr_iter']
    if curr_iter == max_iters:
        return END
    return 'critic'

def create_qa_graph(**compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('actor', get_invoke('actor'))
    graph.add_node('loop', loop)
    graph.add_node('critic', get_invoke('critic'))

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'actor')
    graph.add_edge('actor', 'loop')
    graph.add_conditional_edges('loop', loop_conditional_edges, [END, 'critic'])
    graph.add_edge('critic', 'actor')

    return graph.compile(**compile_kwargs)
