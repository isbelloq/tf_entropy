from tf_entropy.datamodels.environment import DataLakePath
import os

def get_data_paths() -> DataLakePath:
    """
    Funcion de recopilación de rutas del lago de datos

    Returns
    -------
    paths : DataLakePath
        Objeto con los paths del lago de datos.
    """
    paths = DataLakePath(
        data_lake_paht= os.environ['DATALAKE_PATH'],
        raw_path= os.path.join(os.environ['DATALAKE_PATH'], os.environ['RAW_PATH']),
        bronze_path= os.path.join(os.environ['DATALAKE_PATH'], os.environ['BRONZE_PATH']),
        silver_path= os.path.join(os.environ['DATALAKE_PATH'], os.environ['SILVER_PATH']),
        gold_path= os.path.join(os.environ['DATALAKE_PATH'], os.environ['GOLD_PATH'])
    )

    return paths