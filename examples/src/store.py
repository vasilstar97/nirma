from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from .llms import embedding

class Store():

    def __init__(self, persist_directory : str = './data/chroma_db'):

        self._persist_directory = persist_directory
        
        self._store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )

    def add_document(self, path: str, mode: str = 'elements'):
        if '.pdf' in path:
            Loader = UnstructuredPDFLoader
        elif '.docx' in path:
            Loader = UnstructuredWordDocumentLoader
        else:
            raise ValueError('Формат файла не поддерживается')      
        
        loader = Loader(
            path, 
            mode=mode, 
            languages=['rus', 'eng'],
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=1200,
            overlap=100,
            combine_text_under_n_chars=600
        )
        
        docs = loader.load()
        self._store.add_documents(documents=docs)

    def delete_document(self, filename: str):
        """
        Удаление всех чанков конкретного документа
        """
        # Получаем ID всех чанков этого документа
        results = self._store.get(where={"source": filename})
        if results and results.get('ids'):
            self._store.delete(ids=results['ids'])
            return True
        return False

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
                    'fetch_k':10,
                    'lambda_mult':0.5
                }
            )
            # retriever = MultiQueryRetriever.from_llm(
            #         retriever=retriever,
            #         llm=llm
            #     )
            docs = retriever.invoke(query)
            return {
                'results': [{
                    'source': d.metadata['source'],
                    'content':d.page_content
                } for d in docs] 
            }
        
        return tool(search)

__all__=[
    'Store'
]
    