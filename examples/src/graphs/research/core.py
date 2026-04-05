from typing import Annotated, TypedDict
from operator import or_, add
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

MAX_ITERS = 1

class Params(BaseModel):
    location : str
    year : int
    plan : dict[str, list[str]]

class GraphState(TypedDict):
    params : Params
    tasks : list
    results : Annotated[dict, or_]
    result : dict
    max_iters : int

def init(state : GraphState):
    params = Params(**state['params'])
    tasks = [(topic, question) for topic, questions in params.plan.items() for question in questions]
    return {
        'params': params,
        'tasks': tasks,
        'results': {},
        'max_iters': state.get('max_iters', MAX_ITERS)
    }

def gate(state : GraphState):
    results = state['results']
    tasks = state['tasks']
    return {
        'tasks': [t for t in tasks if t not in results]
    }

def loop(state : GraphState):
    return {}

def get_worker(i : int, qa_graph):

    def worker(state : GraphState):
        tasks = state['tasks']
        params = state['params']
        if i<len(tasks):
            topic, question = tasks[i]
            state = qa_graph.invoke({'params': {
                'location': params.location,
                'year': params.year,
                'topic': topic,
                'question': question
            }, 'max_iters': state['max_iters']})
            result = state['result']
            return {
                'results': {(topic, question): result}
            }
        return {
            'results': {}
        }
    
    return worker

def finalize(state : GraphState):
    results = state['results']
    plan = state['params'].plan
    return {'result':{t:{q: results[t,q] for q in qs} for t,qs in plan.items()}}

def gate_conditional_edges(state : GraphState):
    tasks = state['tasks']
    if len(tasks) == 0:
        return 'finalize'
    return 'loop'

def create_research_graph(qa_graph, n_workers : int = 8, **compile_kwargs):
    graph = StateGraph(GraphState)

    graph.add_node('init', init)
    graph.add_node('gate', gate)
    graph.add_node('loop', loop)
    graph.add_node('finalize', finalize)

    for i in range(n_workers):
        name = f'worker_{i+1}'
        worker = get_worker(i, qa_graph)
        graph.add_node(name, worker)
        graph.add_edge('loop', name)
        graph.add_edge(name, 'gate')

    graph.add_edge(START, 'init')
    graph.add_edge('init', 'gate')
    graph.add_conditional_edges('gate', gate_conditional_edges, ['finalize', 'loop'])
    graph.add_edge('finalize', END)

    return graph.compile(**compile_kwargs)