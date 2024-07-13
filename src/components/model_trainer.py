import os
import sys
from dataclasses import dataclass

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from datasets import load_metric, load_dataset

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def load_model_and_tokenizer(self, model_name):
        if model_name in ["codebert-base", "codet5-base", "gpt2", "t5-base"]:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise CustomException(f"Model {model_name} is not supported")
        return model, tokenizer

    def train_and_evaluate_model(self, model, tokenizer, train_dataset, test_dataset):
        try:
            logging.info("Training and evaluating model")

            training_args = TrainingArguments(
                output_dir='./results',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
            )

            def compute_metrics(pred):
                metric = load_metric("accuracy")
                logits, labels = pred
                predictions = logits.argmax(-1)
                return metric.compute(predictions=predictions, references=labels)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            trainer.train()

            eval_result = trainer.evaluate()

            accuracy = eval_result["eval_accuracy"]

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_dataset, test_dataset):
        try:
            logging.info("Preparing to train models")

            model_names = ["codebert-base", "codet5-base", "gpt2", "t5-base"]
            model_accuracies = {}

            for model_name in model_names:
                model, tokenizer = self.load_model_and_tokenizer(model_name)
                accuracy = self.train_and_evaluate_model(model, tokenizer, train_dataset, test_dataset)
                model_accuracies[model_name] = accuracy

            best_model_name = max(model_accuracies, key=model_accuracies.get)
            best_model_accuracy = model_accuracies[best_model_name]

            if best_model_accuracy < 0.6:
                raise CustomException("No sufficiently accurate model found")

            logging.info(f"Best model found: {best_model_name} with accuracy {best_model_accuracy}")

            best_model, best_tokenizer = self.load_model_and_tokenizer(best_model_name)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=(best_model, best_tokenizer)
            )

            return best_model_accuracy

        except Exception as e:
            raise CustomException(e, sys)
