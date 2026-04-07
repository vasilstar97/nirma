import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.tools import tool
from ..llms import embedding

SECTION_COLUMN = 'Раздел'
CODE_COLUMN = 'Код показателя'
NAME_COLUMN = 'Название показателя'
UNIT_COLUMN = 'Единица измерения'

class IndicatorsStore():

    def __init__(self, path : str):
        store = InMemoryVectorStore(embedding=embedding)
        self._store = store

        if path.endswith(('.xlsx', '.xls')):
            read = pd.read_excel
        elif path.endswith(('.csv')):
            read = pd.read_csv
        else:
            raise ValueError('Тип файла не поддерживается')
        
        df = read(path)[[SECTION_COLUMN, CODE_COLUMN, NAME_COLUMN, UNIT_COLUMN]].copy()
        self._df = df.set_index(CODE_COLUMN)

        docs = []
        for _,row in df.iterrows():
            data = row.to_dict()
            page_content = data[NAME_COLUMN]
            del data[NAME_COLUMN]
            doc = Document(id=data[CODE_COLUMN], page_content=page_content, metadata=data)
            docs.append(doc)

        store.add_documents(docs)
        self._docs = docs

    @property
    def tool(self):

        @tool
        def indicators_tool(query : str):
            """
            Поиск по базе показателей
            """
            return self._store.similarity_search(query, k=10)
        
        return indicators_tool