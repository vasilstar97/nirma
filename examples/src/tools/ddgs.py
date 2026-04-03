from langchain_community.tools import DuckDuckGoSearchResults

# 1. Инициализируем инструмент поиска
ddgs_tool = DuckDuckGoSearchResults(
    name="ddgs_tool",
    num_results=4,
    # output_format='list'
)