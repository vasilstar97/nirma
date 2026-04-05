from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import TypedDict, Annotated
from operator import add
from ...agent import Agent, llm
from ._prompts import *
from ...tools import lightrag_tool, ddgs_tool

MAX_ITERS = 1
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
    result : str

def init(state : GraphState):
    params = Params(**state['params'])
    tools = state.get('tools', TOOLS)
    max_iters = state.get('max_iters', MAX_ITERS)

    actor = Agent(
        system_prompt=ACTOR_PROMPT.format(**params.model_dump()),
        tools=tools,
        is_checkpointer=True,
        # is_summarization=True,
    )
    critic = Agent(
        system_prompt=CRITIC_PROMPT.format(**params.model_dump()),
        tools=tools,
        is_checkpointer=True,
        # is_summarization=True,
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

def finalize(state : GraphState):
    params = state['params']
    messages = state['messages']
    if len(messages) == 1:
        return {
            'result': messages[0]
        }
    res = llm.invoke(FINALIZER_MESSAGE.format(
        **params.model_dump(),
        messages=messages
    ))
    return {
        'result': res.content
    }   

def critic_conditional_edges(state : GraphState):
    max_iters = state['max_iters']
    curr_iter = state['curr_iter']
    if curr_iter == max_iters:
        return 'finalize'
    return 'loop'

def create_qa_graph(critisize : bool = True, **compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('loop', loop)
    graph.add_node('actor', get_invoke('actor'))
    graph.add_node('finalize', finalize)
    if critisize:
        graph.add_node('critic', get_invoke('critic'))

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'loop')
    graph.add_edge('loop', 'actor')
    if critisize:
        graph.add_edge('actor', 'critic')
        graph.add_conditional_edges('critic', critic_conditional_edges, ['finalize', 'loop'])
    else:
        graph.add_edge('actor', 'finalize')
    graph.add_edge('finalize', END)

    return graph.compile(**compile_kwargs)
