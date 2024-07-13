import os
import sys
import pickle
import dill
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_metric, Dataset
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(train_dataset, test_dataset, models):
    try:
        report = {}
        accuracy_metric = load_metric("accuracy")

        for model_name in models:
            model = models[model_name]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            training_args = TrainingArguments(
                output_dir='./results',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
            )

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = logits.argmax(-1)
                return accuracy_metric.compute(predictions=predictions, references=labels)

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
            report[model_name] = accuracy

        return report
    except Exception as e:
        raise CustomException(e, sys)
