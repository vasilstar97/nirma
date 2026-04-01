from pydantic import BaseModel, Field, model_validator
from ..utils import get_id

class BaseNote(BaseModel):
    """Запись для добавления на доску"""
    content : str = Field(description='Содержимое записи')

class Note(BaseNote):
    id : str = Field(default='', description='ID записи')
    author_id : str = Field(description='ID автора записи')

    @model_validator(mode='after')
    def _validate_model(self):
        self.id = get_id()
        return self
    
class Board(BaseModel):
    notes : list[Note] = Field(default=[], description='Записи на доске')

    def add_note(self, base_note : BaseNote, author_id : str) -> str:
        note = Note(author_id=author_id, **base_note.model_dump())
        self.notes.append(note)
        return note.id
    
    def remove_notes(self, notes_ids : list[str]):
        self.notes = [n for n in self.notes if n.id not in notes_ids]

    def to_str(self):
        return 'Содержимое доски:\n' + str(self.notes)
    
__all__=[
    'BaseNote',
    'Note',
    'Board'
]