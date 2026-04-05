from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import TypedDict, Annotated
from operator import add
from ...agent import llm
from ._prompts import *
from ...tools import lightrag_tool, ddgs_tool

MAX_TOOLS = 5
MAX_ITERS = 3
TOOLS = [lightrag_tool, ddgs_tool]

class Params(BaseModel):
    location : str
    year : int
    topic : str
    question : str

class GraphState(TypedDict):
    params : Params
    tools : dict
    max_iters : int
    curr_iter : int
    context : Annotated[list[dict], add]
    messages : Annotated[list[str], add]
    result : str

def init(state : GraphState):
    return {
        'params': Params(**state['params']),
        'tools': {t.name: t for t in TOOLS},
        'max_iters': state.get('max_iters', MAX_ITERS),
        'curr_iter': 0,
        'context': [],
        'messages': [],
    }

def loop(state : GraphState):
    return {
        'curr_iter': state['curr_iter'] + 1
    }

def tool(state : GraphState):
    tools = state['tools']
    tools_llm = llm.bind_tools(tools=tools.values(), tool_choice='required')
    tool_calls = tools_llm.invoke(TOOL_MESSAGE.format(
        **state['params'].model_dump(),
        context=str(state['context']),
        messages=str(state['messages'])
    )).tool_calls

    results = []

    for tool_call in tool_calls[:MAX_TOOLS]:
        name = tool_call['name']
        args = tool_call['args']
        tool = tools[name]
        result = tool.invoke(args)
        results.append({
            'tool': name,
            'args': args,
            'result': result
        })

    return {
        'context': results
    }

def actor(state : GraphState):
    res = llm.invoke(ACTOR_MESSAGE.format(
        **state['params'].model_dump(),
        context=str(state['context']),
        messages=str(state['messages'])
    )).content
    return {
        'messages': [{
            'author': 'actor',
            'content': res
        }]
    }

def critic(state : GraphState):
    res = llm.invoke(CRITIC_MESSAGE.format(
        **state['params'].model_dump(),
        context=str(state['context']),
        messages=str(state['messages'])
    )).content
    return {
        'messages': [{
            'author': 'critic',
            'content': res
        }]
    }

def finalizer(state : GraphState):
    res = llm.invoke(FINALIZER_MESSAGE.format(
        **state['params'].model_dump(),
        messages=str(state['messages'])
    )).content
    return {
        'result': res
    }   

def critic_conditional_edges(state : GraphState):
    max_iters = state['max_iters']
    curr_iter = state['curr_iter']
    if curr_iter == max_iters:
        return 'finalizer'
    return 'loop'

def create_qa_graph(**compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('loop', loop)
    graph.add_node('tool', tool)
    graph.add_node('actor', actor)
    graph.add_node('critic', critic)
    graph.add_node('finalizer', finalizer)

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'loop')
    graph.add_edge('loop', 'tool')
    graph.add_edge('tool', 'actor')
    graph.add_edge('actor', 'critic')
    graph.add_conditional_edges('critic', critic_conditional_edges, ['finalizer', 'loop'])
    graph.add_edge('finalizer', END)

    return graph.compile(**compile_kwargs)
