import osmnx as ox
from langchain.tools import tool

@tool
def osm_address_tool(address : str, tags : dict, dist : int) -> list[dict]:
    """
    Запрашивает объекты из OpenStreetMap в радиусе dist метров от адреса address, которые соответствуют тегам tags.
    Возвращает список словарей, каждый из которых содержит информацию об объекте.
    """
    try:
        features = ox.features_from_address(address, tags=tags, dist=dist).drop(columns=['geometry'])
        results = [
            f[~f.isna()].to_dict()
            for _,f in features.iterrows()
        ]
        return results
    except Exception as e:
        return [{"error": str(e)}]
    
@tool
def osm_place_tool(place : str, tags : dict) -> list[dict]:
    ...