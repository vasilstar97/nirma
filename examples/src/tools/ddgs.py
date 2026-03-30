from langchain_community.tools import DuckDuckGoSearchResults

# 1. Инициализируем инструмент поиска
ddgs_tool = DuckDuckGoSearchResults(
    name="ddgs_tool",
    output_format='list'
)