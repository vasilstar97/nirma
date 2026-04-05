from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool

@tool
def ddgs_tool(query : str) -> str:
    """
    Поиск по интернету в DuckDuckGo.
    Если возвращает ошибку, попробуйте позже или воспользуйтесь другим инструментом.
    """
    tool = DuckDuckGoSearchResults(
        num_results=4,
    )
    try:
        return tool.invoke(query)
    except Exception as e:
        return str(e)