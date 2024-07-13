import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.utils import evaluate_models

if __name__ == "__main__":
    try:
        logging.info("Starting the data ingestion process")
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        logging.info("Starting the data transformation process")
        data_transformation = DataTransformation()
        train_dataset, test_dataset, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        logging.info("Starting the model training process")
        model_trainer = ModelTrainer()
        models = {
            "codebert-base": "codebert-base",
            "codet5-base": "codet5-base",
            "gpt2": "gpt2",
            "t5-base": "t5-base"
        }
        model_reports = evaluate_models(train_dataset, test_dataset, models)
        best_model_name = max(model_reports, key=model_reports.get)
        best_model_score = model_reports[best_model_name]

        logging.info(f"Best model: {best_model_name} with accuracy {best_model_score}")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution")
        raise CustomException(e, sys)
