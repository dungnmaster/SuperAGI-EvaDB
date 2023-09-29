import json
import evadb
import os
import pandas as pd

from enum import Enum, auto
from typing import Type, Optional

from pydantic import BaseModel, Field

from superagi.helper.google_search import GoogleSearchWrap
from superagi.helper.token_counter import TokenCounter
from superagi.llms.base_llm import BaseLlm
from superagi.tools.base_tool import BaseTool
from urllib.parse import urlparse


class TaskType(Enum):
    CLASSIFICATION = 1
    FORECASTING = 2

class DatasetType(Enum):
    LOCAL_CSV = 1
    EXTERNAL_CSV = 2


class ModelTrainSchema(BaseModel):
    dataset_path: str = Field(
        ...,
        description="""
            The path dataset which should be used to train the model. 
            It should be file path when the dataset type is LOCAL_CSV.
            It should be table name when the dataset type is DB.
            It should be CSV file url if the dataset type is EXTERNAL_CSV.
        """,
    )
    task_type: TaskType = Field(
        ...,
        description="The type of training task. This should be the value for one of the TaskType Enums. Like 1 for CLASSIFICATION and 2 for FORECASTING",
    )
    dataset_type: DatasetType = Field(
        ...,
        description="The type of the dataset. This should be the value for one of the DatasetType Enums. LIKE 1 for LOCAL_CSV and 2 for EXTERNAL_CSV",
    )
    task_name: str = Field(
        ...,
        description="The name in camelcase for this prediction task. This should represent what we are trying to predict",
    )
    prediction_column: str = Field(
        ...,
        description="The dataset attribute that should be predicted by the trained model. This should exactly match to one of the columns in the dataset",
    )

    
class ModelTrainTool(BaseTool):
    """
    Trains an ML model type `TaskType` on the provided dataset.

    Attributes:
        name : The name of the tool.
        description : The description of the tool.
        args_schema : The args schema.
    """
    llm: Optional[BaseLlm] = None
    name = "ModelTrainTool"
    description = "A tool for training an ML model using on the provided dataset"
    args_schema: Type[ModelTrainSchema] = ModelTrainSchema
    cursor: object = None 

    class Config:
        arbitrary_types_allowed = True

    def _load_dataset(self, dataset_path: str, dataset_type: DatasetType) -> (str, str):
        table = dataset_path
        try:
            filename = os.path.basename(urlparse(dataset_path).path)
            if dataset_type == DatasetType.EXTERNAL_CSV:
                dataset_df = pd.read_csv(dataset_path)
                dataset_df.to_csv(filename)
                dataset_path = os.path.join(os.getcwd(), filename)

            print("## data download done")
            #TODO: generate the schema using df.head + llm
            # Load the dataset into table if dataset is type CSV

            # create a table for the dataset
            print("## starting data load")
            # cursor = evadb.connect().cursor()
            table = 'dataset__' + os.path.splitext(filename)[0]

            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    number_of_rooms INTEGER,
                    number_of_bathrooms INTEGER,
                    sqft INTEGER,
                    location TEXT(128),
                    days_on_market INTEGER,
                    initial_price INTEGER,
                    neighborhood TEXT(128),
                    rental_price FLOAT(1,1)
                )
            """
            print("## CREATE_QUERY ", create_table_query)
            create_response = self.cursor.query(create_table_query).df()
            print(f"## CREATE RESPONSE: {create_response}")

            # load the dataset into the crated table
            load_response = self.cursor.query(f"LOAD CSV '{dataset_path}' INTO {table}").df()
            print(f"## load response: {load_response}")

        except Exception as e:
            error_msg= f"Failed to download the dataset locally due to error: {e}"
            print(error_msg)
            return (None, error_msg)
        
        if dataset_type == DatasetType.EXTERNAL_CSV:
            try:
                os.remove(dataset_path)
            except:
                pass
        return (table, None)
            

    def _execute(self, dataset_path: str, task_type: TaskType, dataset_type: DatasetType, task_name: str, prediction_column: str) -> tuple:
        """
        Execute the EvaDB model training tool

        Args:
            dataset_path: The path to the dataset CSV
            task_type: The type of model to train.  This should be one of the TaskType Enums.
            dataset_type: The type of the dataset. This should be one of the Dataset Enums.
            task_name: The name in camelcase for this prediction task.
            prediction_column: The dataset attribute that should be predicted by the trained model. This should exactly match to one of the columns in the dataset.

        Returns:
            Response of the training query on EvaDB.
        """
        print("####^^^^###")
        # 1. Create a table for the dataset in the underlying database.
        # 2. Load the dataset into the created table.

        self.cursor = evadb.connect().cursor()
        dataset_table, error = self._load_dataset(dataset_path, dataset_type)
        if dataset_table == None:
            print("LOAD FAILED")
            return ("Failed to compete the tool execution", f"reason: {error}")
        
        # 3. Train a model using the data in the created table.
        print(f"## task_type: {task_type} {TaskType.CLASSIFICATION.value}")
        function_type =  'Ludwig' #task_type == 'Ludwig' if task_type == TaskType.CLASSIFICATION.value else 'Forecasting'
        model_train_query = f"""
            CREATE OR REPLACE FUNCTION {task_name} FROM
            ( SELECT * FROM {dataset_table} )
            TYPE {function_type}
            PREDICT '{prediction_column}'
            TIME_LIMIT 3600;
        """
        print(f"## train query: {model_train_query}")
        train_response = self.cursor.query(model_train_query).df()

        print(f"## model train response {train_response}")

        comp_result = self.cursor.query(f"""
            SELECT rental_price, predicted_rental_price FROM {dataset_table}
            JOIN LATERAL {task_name}(*) AS Predicted(predicted_rental_price) LIMIT 10;
        """).df()

        print(f"## comp result: {comp_result}")
        return ("Successfully executed the provided query", "result: ")
        



# cursor.query(
# """
#     CREATE OR REPLACE FUNCTION rental_price_prediction FROM
#     ( SELECT * FROM dataset__dataset_home_rentals )
#     TYPE Ludwig
#     PREDICT 'rental_price'
#     TIME_LIMIT 3600;
# """).df()