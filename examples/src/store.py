from typing import Literal
from langchain.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from .llms import llm, embedding

def _filter_metadata(docs):
    for i,d in enumerate(docs):
        metadata = d.metadata
        
        for key in [
            # 'source',
            'link_texts',
            'link_urls',
            # 'coordinates',
            # 'file_directory',
            # 'filename',
            'filetype',
            'languages',
            'last_modified',
            # 'text_as_html',
            # 'emphasized_text_contents',
            # 'emphasized_text_tags',
            'orig_elements',
        ]:
            if key in metadata:
                del metadata[key]

        # metadata['chunk_index'] = i

    return docs

class Store():

    def __init__(self, mode : str='elements'):

        self._mode = mode
        self._store = InMemoryVectorStore(embedding=embedding)
        self._docs = []

    def add_document(self, path : str):
        if '.pdf' in path:
            Loader = UnstructuredPDFLoader
        elif '.docx' in path:
            Loader = UnstructuredWordDocumentLoader
        else:
            raise ValueError('Формат файла не поддерживается')      
        
        loader = Loader(
            path, 
            mode=self._mode, 
            languages=['rus', 'eng'],
            chunking_strategy="by_title",  # или "basic"
            max_characters=1000,  # Максимальный размер чанка
            new_after_n_chars=1200,  # "Мягкий" максимум
            overlap=100,  # Перекрытие между чанками
            combine_text_under_n_chars=600
        )
        docs = loader.load()
        _filter_metadata(docs)
        self._store.add_documents(docs)
        self._docs.extend(docs)

    @property
    def tool(self):

        def search(query : str) -> dict:
            """
            Поиск информации по документам в векторной базе данных.
            """
            retriever = self._store.as_retriever(
                search_type='mmr',
                search_kwargs={
                    'k': 4,
                }
            )
            # retriever = MultiQueryRetriever.from_llm(
            #         retriever=retriever,
            #         llm=llm
            #     )
            docs = retriever.invoke(query)
            return {
                'results': docs 
            }
        
        return tool(search)

__all__=[
    'Store'
]
    