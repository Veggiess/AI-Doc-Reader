from dataset import load_dataset
from transformers import AutoTokenizer
import torch  # Added missing import

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
        # Tokenize and process
        encoding = tokenizer(
            example["words"],
            boxes=example["bboxes"],
            word_labels=example["ner_tags"],  # Let tokenizer handle label alignment
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # Convert labels to tensor
        encoding["labels"] = torch.tensor(encoding.pop("word_labels"))
        
        return {k: v.squeeze(0) for k, v in encoding.items()}

    # Process dataset
    encoded_dataset = dataset.map(process_example, batched=False)
    return encoded_dataset, label2id, id2label