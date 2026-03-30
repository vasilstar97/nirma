from pydantic import BaseModel, Field, model_validator
from ..utils import get_id

PREVIEW_SIZE = 140
PREVIEW_SPLIT = ' '
PREVIEW_ELLIPSIS = '...'

class BaseNote(BaseModel):
    """Запись для добавления на доску"""
    content : str = Field(description='Содержимое записи')

class Note(BaseNote):
    id : str = Field(default='', description='ID записи')
    author_id : str = Field(description='ID автора записи')
    author_role : str = Field(description='Роль автора записи')

    @property
    def preview(self) -> str:
        split = self.content[:PREVIEW_SIZE].split(PREVIEW_SPLIT)
        if len(self.content) > PREVIEW_SIZE:
            split[-1] = PREVIEW_ELLIPSIS
        return PREVIEW_SPLIT.join(split)

    @model_validator(mode='after')
    def _validate_model(self):
        self.id = get_id()
        return self
    
    def print(self, color = 'yellow', width : int = 100):
        from rich import print
        from rich.panel import Panel

        panel_title = f"📌#{self.id} by {self.author_id} ({self.author_role})"
        panel_content = self.content
            
        print(Panel(
            panel_content,
            title=panel_title,
            border_style=color,
            width=width
        ))
    
__all__=[
    'BaseNote',
    'Note'
]