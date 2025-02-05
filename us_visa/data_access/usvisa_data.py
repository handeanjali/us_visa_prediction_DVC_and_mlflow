from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException
import pandas as pd
import sys
from typing import Optional
import numpy as np


class USVisaData:
    """
    A utility class to interact with MongoDB and export collection data as a Pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the MongoDB client with the specified database.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise USvisaException(e, sys)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Exports the entire MongoDB collection as a Pandas DataFrame.

        Args:
            collection_name (str): The name of the MongoDB collection to export.
            database_name (Optional[str]): The name of the database (default: None, uses the default database).

        Returns:
            pd.DataFrame: A DataFrame containing the collection's data.

        Raises:
            USvisaException: If an error occurs while fetching data from MongoDB.
        """
        try:
            # Use the default database if none is specified
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Convert collection data to DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Remove MongoDB's default "_id" column if it exists
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Replace placeholder "na" values with NaN for consistency
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise USvisaException(e, sys)
