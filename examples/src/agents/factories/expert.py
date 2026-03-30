from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board, BaseNote
from ...tools import ddgs_tool

EXPERT_PROMPT = """
Вы выдающийся специалист:
{role_name}
Описание:
{role_description}

Основываясь на ваших экспертных знаниях и текущем содержимом общей доски, решите задачу, изложите свои идеи и информацию, которую вы хотите записать на доску.
Совершенно не обязательно полностью соглашаться с точкой зрения, представленной на доске.
"""

def create_expert_agent(role : Role, board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + EXPERT_PROMPT
    agent = RoleAgent(
        role=role,
        system_prompt=system_prompt,
        tools=[*board.get_ro_tools(), ddgs_tool],
        response_format=BaseNote,
    )
    return agent