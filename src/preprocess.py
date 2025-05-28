from datasets import load_dataset  
from transformers import AutoTokenizer
import torch

def prepare_dataset(model_name="microsoft/layoutlmv3-base", max_length=512):
    # Load dataset
    dataset = load_dataset("nielsr/funsd")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    # Label setup
    label_list = ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", 
                "B-ANSWER", "I-ANSWER", "O"]
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}

    def process_example(example):
        # Normalize bounding boxes to 0-1000 scale
        # (Assuming original image size is 1000x1000 for simplicity)
        # You should adjust this based on your actual document dimensions
        normalized_boxes = []
        for box in example["bboxes"]:
            xmin, ymin, xmax, ymax = box
            normalized_boxes.append([
                max(0, min(1000, int(xmin))),
                max(0, min(1000, int(ymin))),
                max(0, min(1000, int(xmax))),
                max(0, min(1000, int(ymax)))
            ])

        # Tokenize and process
        encoding = tokenizer(
            example["words"],
            boxes=normalized_boxes,
            word_labels=example["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # Convert labels to tensor and ensure proper length
        labels = encoding.pop("word_labels")
        # Pad labels to max_length with -100 (ignored in loss)
        if len(labels) < max_length:
            labels += [-100] * (max_length - len(labels))
        encoding["labels"] = torch.tensor(labels[:max_length])
        
        return {k: v.squeeze(0) for k, v in encoding.items()}

    # Process dataset
    encoded_dataset = dataset.map(process_example, batched=False)
    return encoded_dataset, label2id, id2label