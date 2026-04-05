import requests
from langchain.tools import tool

@tool
def lightrag_tool(query : str):
    """
    Поиск по документам в LightRAG
    Если возвращает ошибку, попробуйте позже или воспользуйтесь другим инструментом.
    """
    json = {
        'query': query,
        'mode': 'mix',
        'only_need_context': True,
        'only_need_prompt': False,
        'top_k': 5,
        'chunk_top_k': 10,
        'max_total_tokens': 1500,
        'enable_rerank': False,
        'include_references': True,
        'include_chunk_content': True
    }
    try:
        res = requests.post('http://localhost:9621/query', json=json, timeout=10)
        return res.text
    except Exception as e:
        return str(e)