from pydantic import BaseModel

class DataLakePath(BaseModel):
    """
    Clase de acceso al lago de datos.

    Attributes
    ----------
    data_lake_paht : str
        Path base del lago de datos
    raw_path : str
        Raw path con base en el lago de datos
    bronze_path : str
        Bronze path con base en el lago de datos
    silver_path : str
        Silver path con base en el lago de datos
    gold_path : str
        Gold path con base en el lago de datos
    """
    data_lake_paht : str
    raw_path : str
    bronze_path : str
    silver_path : str
    gold_path : str