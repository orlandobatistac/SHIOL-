import pandas as pd
from sqlalchemy import create_engine, inspect
from loguru import logger
import configparser

class PersistenceManager:
    def __init__(self, db_file):
        self.db_uri = f'sqlite:///{db_file}'
        self.engine = create_engine(self.db_uri)
        logger.info(f"PersistenceManager initialized for database: {db_file}")

    def save_dataframe(self, df, table_name, if_exists='fail', index=False):
        """
        Saves a pandas DataFrame to a specified table in the database.
        """
        try:
            with self.engine.connect() as connection:
                df.to_sql(table_name, connection, if_exists=if_exists, index=index)
                logger.info(f"Successfully saved DataFrame to table '{table_name}'.")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to table '{table_name}': {e}")
            raise

    def load_dataframe(self, table_name):
        """
        Loads a table from the database into a pandas DataFrame.
        """
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_table(table_name, connection)
                logger.info(f"Successfully loaded table '{table_name}' into DataFrame.")
                return df
        except Exception as e:
            logger.error(f"Failed to load table '{table_name}': {e}")
            return pd.DataFrame()

    def table_exists(self, table_name):
        """
        Checks if a table exists in the database.
        """
        inspector = inspect(self.engine)
        return inspector.has_table(table_name)

def get_persistence_manager():
    """
    Factory function to get an instance of PersistenceManager.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    db_file = config['paths']['db_file']
    return PersistenceManager(db_file)