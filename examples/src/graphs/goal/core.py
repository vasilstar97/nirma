from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, Annotated, Any
from operator import add
from ._prompts import *
from ...agent import Agent

MAX_ITERS = 3

class Indicator(BaseModel):
    code : str = Field(description='Код показателя')
    name : str = Field(description='Название показателя')
    target : Literal['Минимизация', 'Максимизация'] = Field(description='Направление целевого значения (минимизация или максимизация)')

class Task(BaseModel):
    description : str = Field(description='Описание задачи')
    indicators : list[Indicator] = Field(description='Целевые показатели')

class Goal(BaseModel):
    description : str = Field(description='Описание цели')
    tasks : list[Task] = Field(description='Задачи пространственного развития')

class ActorResponse(BaseModel):
    external_factors : list[str] = Field(description='Внешние факторы развития')
    internal_factors : list[str] = Field(description='Внутренние факторы развития')
    mission : str = Field(description='Миссия МО')
    goals : list[Goal] = Field(description='Цели пространственного развития')

class GraphState(TypedDict):
    analysis : Any
    max_iters : int
    curr_iter : int
    tools : list
    actor : Agent
    critic : Agent
    messages : Annotated[list[str], add]
    result : ActorResponse

def init(state : GraphState):
    analysis = state['analysis']
    tools = state['tools']
    return {
        'max_iters': state.get('max_iters', MAX_ITERS),
        'curr_iter': 0,
        'actor': Agent(system_prompt=ACTOR_PROMPT.format(analysis=str(analysis)), tools=tools, response_format=ActorResponse, is_checkpointer=True),
        'critic': Agent(system_prompt=CRITIC_PROMPT.format(analysis=str(analysis)), is_checkpointer=True),
        'messages': []
    }

def loop(state : GraphState):
    return {
        'curr_iter': state['curr_iter'] + 1
    }

def actor(state : GraphState):
    actor = state['actor']
    messages = [str(m) for m in state['messages'][-1:]]
    res = actor.invoke(messages)
    return {
        'messages': [res]
    }

def critic(state : GraphState):
    critic = state['critic']
    messages = [str(m) for m in state['messages'][-1:]]
    res = critic.invoke(messages)
    return {
        'messages': [res]
    }

def finalize(state : GraphState):
    return {
        'result': state['messages'][-1]
    }

def actor_conditional_edges(state : GraphState):
    max_iters = state['max_iters']
    curr_iter = state['curr_iter']
    if curr_iter == max_iters:
        return 'finalize'
    return 'critic'

def create_goal_graph(**compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('loop', loop)
    graph.add_node('actor', actor)
    graph.add_node('critic', critic)
    graph.add_node('finalize', finalize)

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'loop')
    graph.add_edge('loop', 'actor')
    graph.add_conditional_edges('actor', actor_conditional_edges, ['finalize', 'critic'])
    graph.add_edge('critic', 'loop')
    graph.add_edge('finalize', END)

    return graph.compile(**compile_kwargs)