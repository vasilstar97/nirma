from pydantic import BaseModel, Field
from langchain.messages import SystemMessage
from ._prompts import *
from .board import Board, BaseNote
from ..agent import Agent
from ..tools import ddgs_tool

class ControllerResponse(BaseModel):
    agents_ids : list[str] = Field(min_length=1, description='Упорядоченный список ID агентов')

class CleanerResponse(BaseModel):

    notes_ids : list[str] = Field(default=[], description='Список ID записей к удалению')

class DeciderResponse(BaseModel):
    note : BaseNote = Field(description='Запись для добавления на доску')
    is_final : bool = Field(default=False, description='Сигнал о завершении процесса работы над задачей')

class BlackBoard():

    def __init__(self, question : str, web : bool = True, store = None):
        self.question = question
        self.board = Board()
        
        experts = []
        if web:
            experts.append(self.create_worker_agent(
                RAG_PROMPT,
                'Эксперт по поиску в интернете',
                'Ищет информацию в интернете',
                tools=[ddgs_tool],
                response_format=BaseNote
            ))
        if store is not None:
            experts.append(self.create_worker_agent(
                RAG_PROMPT,
                'Эксперт по работе с документами',
                'Работает с документами стратегического и территориального планирований, а также с нормативными документами',
                tools=[store.tool],
                response_format=BaseNote
            ))
        self.experts = experts
        
        self.planner = self.create_worker_agent(
            PLANNER_PROMPT,
            role_name='Планировщик',
            role_description='Разрабатывает пошаговый план решения задачи на основе содержимого доски',
            response_format=BaseNote
        )
        self.critic = self.create_worker_agent(
            CRITIC_PROMPT,
            role_name='Критик',
            role_description='Выявляет ошибочные или вводящие в заблуждение записи на доске',
            response_format=BaseNote
        )
        self.cleaner = self.create_worker_agent(
            CLEANER_PROMPT,
            role_name='Уборщик',
            role_description='Анализирует доску, выявляет и удаляет бесполезные или избыточные записи',
            response_format=CleanerResponse
        )
        self.decider = self.create_worker_agent(
            DECIDER_PROMPT,
            role_name='Арбитр',
            role_description='Оценивает полноту информации. Останавливает либо инициирует продолжение обсуждения',
            response_format=DeciderResponse
        )
        self.controller = Agent(
            system_prompt=CONTROLLER_PROMPT,
            metadata={
                'workers': [{
                    'id': w.id,
                    'role_name': w.metadata['role_name'],
                    'role_description': w.metadata['role_description']
                } for w in self.workers.values()],
                'question': self.question,
            },
            response_format=ControllerResponse
        )

    def create_worker_agent(
        self,
        system_prompt : str, 
        role_name : str, 
        role_description : str, 
        tools : list | None = None, 
        response_format : type[BaseModel] | None = None
    ) -> Agent:
        return Agent(
            WORKER_PROMPT + '\n' + system_prompt, 
            tools=tools, 
            response_format=response_format,
            metadata={
                'role_name': role_name,
                'role_description': role_description,
                'question': self.question
            }
        )

    @property
    def workers(self) -> dict[str, Agent]:
        return {a.id: a for a in [
            self.planner,
            *self.experts,
            self.critic,
            self.cleaner,
            self.decider
        ]}
    
    @property
    def _invoke_message(self) -> SystemMessage:
        return SystemMessage(self.board.to_str())
    
    def _loop(self) -> bool:
        from tqdm import tqdm

        is_final = False
     
        workers_ids = self.controller.invoke(self._invoke_message).agents_ids
        for worker_id in tqdm(workers_ids):
            worker = self.workers[worker_id]
            response = worker.invoke(self._invoke_message)

            if worker == self.decider:
                note = response.note
                is_final = response.is_final
            elif worker == self.cleaner:
                notes_ids = response.notes_ids
                self.board.remove_notes(notes_ids)
                continue
            else:
                note = response

            self.board.add_note(note, worker.id)

            if is_final:
                break

        return is_final
    
    def run(self, iterations : int = 3) -> bool:

        is_final = False

        for i in range(iterations):
            print(f'{i+1} итерация')
            is_final = self._loop()
            if is_final:
                break

        return is_final
    
    def summarize(self, system_prompt : str = SUMMARIZER_PROMPT):
        summarizer = Agent(
            system_prompt=system_prompt,
            metadata={
                'question': self.question
            }
        )
        return summarizer.invoke(self._invoke_message)
    
__all__=[
    'BlackBoard'
]