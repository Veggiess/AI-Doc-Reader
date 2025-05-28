import torch
from transformers import (
    LayoutLMv3ForTokenClassification, 
    TrainingArguments, 
    Trainer,
    LayoutLMv3TokenizerFast
)
from src.preprocess import prepare_dataset
from datasets import load_metric
import numpy as np

def train_model():
    # Prepare dataset and get label mappings
    encoded_dataset, label2id, id2label = prepare_dataset()
    
    # Load model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Metric for evaluation
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/layoutlmv3-finetuned-funsd",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
        fp16=True if torch.cuda.is_available() else False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=compute_metrics
    )

    # Train and save
    trainer.train()
    model.save_pretrained("./models/layoutlmv3-finetuned-funsd")
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    tokenizer.save_pretrained("./models/layoutlmv3-finetuned-funsd")
    
    print("✅ Model and tokenizer saved to ./models/layoutlmv3-finetuned-funsd")

if __name__ == "__main__":
    train_model()