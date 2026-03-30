from pydantic import BaseModel, Field
from langchain.tools import tool
from .note import Note

class Board(BaseModel):
    question : str = Field(description='Вопрос пользователя')
    notes : list[Note] = Field(default=[], description='Список записей')

    def get_notes(self) -> list[dict]:
        """
        Возвращает список актуальных записей на доске с краткой информацией.
        """
        return [{
            'id': note.id,
            'author_id': note.author_id,
            'author_role': note.author_role,
            'preview': note.preview,
        } for note in self.notes]
    
    def get_note(self, note_id : str) -> Note | None:
        """
        Возвращает запись с указанным id.
        Возвращает None, если запись не найдена.
        """
        notes = [n for n in self.notes if n.id == note_id]
        if len(notes) == 0:
            return None
        return notes[0]
    
    def add_note(self, content : str, author_id : str, author_role : str) -> str:
        """
        Добавляет запись на доску.
        Возвращает ID записи.
        """
        note = Note(content=content, author_id=author_id, author_role=author_role)
        self.notes.append(note)
        return note.id
    
    def remove_note(self, note_id : str):
        self.notes = [n for n in self.notes if n.id != note_id]

    def remove_notes(self, notes_ids : list[str]):
        self.notes = [n for n in self.notes if n.id not in notes_ids]

    def print(self, *args, **kwargs):
        for note in self.notes:
            note.print(*args, **kwargs)

    @property
    def tools(self):
        return [
            tool(self.get_notes),
            tool(self.get_note)
        ]
    
    def get_ro_tools(self):
        return [
            tool(self.get_notes),
            tool(self.get_note)
        ]
    
    def get_rw_tools(self):
        ro_tools = self.get_ro_tools()

        return [
            *ro_tools,
            tool(self.add_note)
        ]

__all__=[
    'Board'
]