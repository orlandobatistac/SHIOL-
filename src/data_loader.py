import pandas as pd
from loguru import logger
import configparser

class DataLoader:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        logger.info(f"DataLoader initialized for file: {data_file_path}")

    def load_historical_data(self):
        """
        Loads and cleans the historical Powerball data from the CSV file.
        """
        try:
            df = pd.read_csv(self.data_file_path)
            logger.info(f"Successfully loaded historical data from {self.data_file_path}")

            # Clean and structure the data
            df.rename(columns={
                'Date': 'draw_date',
                'Number 1': 'n1', 'Number 2': 'n2', 'Number 3': 'n3', 'Number 4': 'n4', 'Number 5': 'n5',
                'Powerball': 'pb'
            }, inplace=True)
            
            df['draw_date'] = pd.to_datetime(df['draw_date'], format='%m/%d/%Y')
            
            # Select only the necessary columns
            required_cols = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'pb']
            df = df[required_cols]
            
            logger.info("Data cleaning and structuring complete.")
            return df

        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            return pd.DataFrame()

def get_data_loader():
    """
    Factory function to get an instance of DataLoader.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    data_file = config['paths']['data_file']
    return DataLoader(data_file)