import sys
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    tokenizer_obj_file_path = os.path.join('artifacts', "tokenizer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_tokenizer(self, model_name):
        '''
        This function is responsible for loading the tokenizer for a given model
        '''
        try:
            if model_name in ["codebert-base", "codet5-base", "gpt2", "t5-base"]:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                raise CustomException(f"Model {model_name} is not supported")
            return tokenizer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, model_name):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Loading tokenizer")

            tokenizer = self.get_tokenizer(model_name)

            target_column_name = "target_text"
            input_column_name = "input_text"

            train_dataset = Dataset.from_pandas(train_df[[input_column_name, target_column_name]])
            test_dataset = Dataset.from_pandas(test_df[[input_column_name, target_column_name]])

            def tokenize_function(examples):
                return tokenizer(examples[input_column_name], padding="max_length", truncation=True)

            logging.info("Tokenizing train and test datasets")

            tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
            tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

            logging.info(f"Saved tokenizer object.")

            save_object(
                file_path=self.data_transformation_config.tokenizer_obj_file_path,
                obj=tokenizer
            )

            return (
                tokenized_train_dataset,
                tokenized_test_dataset,
                self.data_transformation_config.tokenizer_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
