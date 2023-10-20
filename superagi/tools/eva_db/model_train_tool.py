import json
import evadb
import os
import pandas as pd

from enum import Enum, auto
from evadb.interfaces.relational.db import connect_remote
from typing import Type, Optional

from pydantic import BaseModel, Field

from superagi.helper.error_handler import ErrorHandler
from superagi.helper.google_search import GoogleSearchWrap
from superagi.helper.token_counter import TokenCounter
from superagi.llms.base_llm import BaseLlm
from superagi.tools.base_tool import BaseTool
from urllib.parse import urlparse

class TaskType(Enum):
    CLASSIFICATION = 1

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
        description="The type of training task. This should be the value for one of the TaskType Enums. Like 1 for CLASSIFICATION."
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
    agent_id: int = None
    agent_execution_id: int = None
    table_schema_prompt = """
        This is top 5 rows of a CSV file: {csv_rows}
        Generate a SQL query to create a table to store the above csv data. ENFORCE the following constraints in the generated SQL query.
        1. Use IF NOT EXISTS clause.
        2. DO NOT ADD any fields not present in the csv rows provided above.
        3. USE the following template to decide the datatype of table attributes and use max values for the variables based on the csv row data above:
            - INTEGER for INT
            - TEXT(max_length) for VARCHAR(max_length)
            - FLOAT(precision, scale) for FLOAT
            - NDARRAY FLOAT32(precision) for N-dimensional array
    """

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

            # create a table for the dataset
            table = 'dataset__' + os.path.splitext(filename)[0]
            prompt = self.table_schema_prompt.format(
                csv_rows = pd.read_csv(dataset_path).head(5)
            )
            messages = [{"role": "system", "content": prompt}]
            result = self.llm.chat_completion(messages, max_tokens=self.max_token_limit)

            if 'error' in result and result['message'] is not None:
                ErrorHandler.handle_openai_errors(self.toolkit_config.session, self.agent_id, self.agent_execution_id, result['message'])
            create_table_query =  result["content"]
            self.cursor.query(create_table_query).df()

            # load the dataset into the created table
            self.cursor.query(f"LOAD CSV '{dataset_path}' INTO {table}").df()

        except Exception as e:
            error_msg= f"Failed to load the dataset into evadb due to error: {e}"
            print(error_msg)
            return (None, error_msg)
        
        if dataset_type == DatasetType.EXTERNAL_CSV:
            try:
                os.remove(dataset_path)
                print('removed downloaded csv file')
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
        # 1. Create a table for the dataset in the underlying database.
        # 2. Load the dataset into the created table.
        # 3. Train a model using the data in the created table.

        self.cursor = connect_remote('evadb',8803).cursor()
        dataset_table, error = self._load_dataset(dataset_path, dataset_type)
        if dataset_table == None:
            print("LOAD FAILED")
            return ("Failed to compete the tool execution", f"reason: {error}")
        
        function_type =  'Ludwig'
        model_train_query = f"""
            CREATE OR REPLACE FUNCTION {task_name} FROM
            ( SELECT * FROM {dataset_table} )
            TYPE {function_type}
            PREDICT '{prediction_column}'
            TIME_LIMIT 3600;
        """
        train_response = self.cursor.query(model_train_query).df()
        return ("Successfully executed the provided query", f"result: {train_response}")