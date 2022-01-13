from pydantic import BaseModel

class DataLakePath(BaseModel):
    """
    Clase de acceso al lago de datos.

    Attributes
    ----------
    data_lake_paht : str
        Folder del lago de datos
    raw_path : str
        Folder de los archivos en la capa Raw
    bronze_path : str
        Folder de los archivos en la capa Bronze
    silver_path : str
        Folder de los archivos en la capa Silver
    gold_path : str
        Folder de los archivos en la capa Gold
    """
    data_lake_paht : str
    raw_path : str
    bronze_path : str
    silver_path : str
    gold_path : str