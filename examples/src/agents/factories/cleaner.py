from pydantic import BaseModel, Field
from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board

CLEANER_PROMPT = """
Проанализируйте записи на общей доске и определите любые бесполезные или избыточные.

Если вы обнаружите такие записи:
- Перечислите их
- Для каждого объясните, почему оно бесполезно или избыточно

Если бесполезных записей нет, просто укажите, что бесполезных записей нет и вы ожидаете дополнительной информации.
"""
ROLE_NAME='Уборщик'
ROLE_DESCRIPTION ='Анализирует доску, выявляет и удаляет бесполезные или избыточные записи'

_role = Role(
    name=ROLE_NAME,
    description=ROLE_DESCRIPTION
)

class CleanerResponse(BaseModel):
    notes_ids : list[str] = Field(default=[], description='Список ID записей к удалению')

def create_cleaner_agent(board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + CLEANER_PROMPT
    agent = RoleAgent(
        role=_role,
        system_prompt=system_prompt,
        tools=board.get_ro_tools(),
        response_format=CleanerResponse,
    )
    return agent