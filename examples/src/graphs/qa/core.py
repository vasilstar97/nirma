from typing import Annotated, TypedDict
from operator import add
from pydantic import BaseModel, Field
from ._prompts import *
from ...llms import llm
from ...agent import Agent

class GraphState(TypedDict):
    main_question : str
    _questions : list[str]
    _question : str
    _results : Annotated[list, add]
    final_answer : str
    iterations : int
    _iteration : int
    
class PlannerResponse(BaseModel):
    questions : list[str] = Field(description='Перечень вопросов')

def invoke_master(state : GraphState):
    return {
        '_questions': [],
        '_results': [],
        '_iteration': 1,
        'iterations': state.get('iterations', 3)
    }

def invoke_planner(state : GraphState):
    iterations = state['iterations']
    iteration = state['_iteration']
    if iteration >= iterations:
        return {
            '_questions': []
        }
    
    message = PLANNER_MESSAGE.format(**state)
    structured_llm = llm.with_structured_output(PlannerResponse)
    return {
        '_questions': structured_llm.invoke(message).questions,
        '_iteration': iteration + 1
    }

def invoke_chooser(state : GraphState):
    questions = state['_questions']
    return {
        '_questions': questions[1:],
        '_question': questions[0]
    }

def create_invoke_agent(tools : list):

    agent = Agent(system_prompt=AGENT_PROMPT, tools=tools)

    def invoke_agent(state : GraphState):
        question = state['_question']
        res = agent.invoke(question)
        return {
            '_results': [{
                'question': question,
                'answer': res
            }]
        }
    
    return invoke_agent

def invoke_gate(state : GraphState):
    return {}

def invoke_finalizer(state : GraphState):
    message = FINALIZER_MESSAGE.format(**state)
    return {
        'final_answer': llm.invoke(message)
    }

from langgraph.graph import StateGraph, START, END
from ...tools import ddgs_tool

def planner_condition(state : GraphState):
    questions = state['_questions']
    if len(questions) > 0:
        return 'chooser'
    return 'finalizer'

def gate_condition(state : GraphState):
    questions = state['_questions']
    if len(questions) > 0:
        return 'chooser'
    return 'planner'

def create_qa_graph(n_agents : int = 1):
    graph = StateGraph(GraphState)
    graph.add_node('master', invoke_master)
    graph.add_node('planner', invoke_planner)
    graph.add_node('chooser', invoke_chooser)
    graph.add_node('gate', invoke_gate)
    graph.add_node('finalizer', invoke_finalizer)

    for i in range(n_agents):
        name = f'agent_{i+1}'
        invoke_agent = create_invoke_agent([ddgs_tool])
        graph.add_node(name, invoke_agent)
        graph.add_edge('chooser', name)
        graph.add_edge(name, 'gate')

    graph.add_edge(START, 'master')
    graph.add_edge('master', 'planner')
    graph.add_conditional_edges('planner', planner_condition, ['chooser', 'finalizer'])
    graph.add_conditional_edges('gate', gate_condition, ['chooser', 'planner'])
    graph.add_edge('finalizer', END)

    return graph

