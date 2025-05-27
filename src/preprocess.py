from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(model_name="microsoft/layoutlmv3-base", max_length=512):
    dataset = load_dataset("nielsr/funsd")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    label_list = ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "O"]
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}

    def process_example(example):
        words = example["words"]
        boxes = example["bboxes"]
        labels = [label2id[label] for label in example["ner_tags"]]

        encoding = tokenizer(
            example["words"],
            boxes=example["bboxes"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        encoding["labels"] = torch.tensor(labels[:max_length])
        return {k: v.squeeze(0) for k, v in encoding.items()}

    encoded_dataset = dataset.map(process_example, batched=False)
    return encoded_dataset, label2id, id2label