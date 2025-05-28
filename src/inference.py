from PIL import Image
from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
import torch
from collections import defaultdict

class DocumentExtractor:
    def __init__(self, model_path="./models/layoutlmv3-finetuned-funsd"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base", add_prefix_space=True)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.id2label = {
            0: "B-HEADER", 1: "I-HEADER", 2: "B-QUESTION",
            3: "I-QUESTION", 4: "B-ANSWER", 5: "I-ANSWER", 6: "O"
        }
        self.model.eval()  # Set model to evaluation mode

    def _run_ocr(self, image: Image.Image):
        """Simulate OCR - in practice, use Tesseract or other OCR"""
        # This is a placeholder - implement real OCR here
        words = ["Name:", "Nguyễn", "Văn", "A", "ID:", "0123456789", "DOB:", "01/01/1990"]
        boxes = [
            [100, 100, 200, 120],  # Name:
            [210, 100, 300, 120],  # Nguyễn
            [310, 100, 350, 120],  # Văn
            [360, 100, 380, 120],   # A
            [100, 150, 150, 170],  # ID:
            [160, 150, 300, 170],  # 0123456789
            [100, 200, 180, 220],  # DOB:
            [190, 200, 300, 220]   # 01/01/1990
        ]
        return words, boxes

    def _normalize_boxes(self, boxes, width, height):
        """Normalize boxes to 0-1000 scale"""
        return [
            [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height))
            ] for box in boxes
        ]

    def _process_entities(self, words, predictions):
        """Convert model predictions to structured data"""
        entities = defaultdict(list)
        current_entity = None
        
        for word, label in zip(words, predictions):
            if label.startswith("B-"):
                current_entity = label[2:]
                entities[current_entity].append(word)
            elif label.startswith("I-") and current_entity == label[2:]:
                entities[current_entity][-1] += " " + word
            else:
                current_entity = None
        
        return {
            "name": " ".join(entities.get("NAME", [])),
            "id_number": " ".join(entities.get("ID", [])),
            "date_of_birth": " ".join(entities.get("DOB", []))
        }

    def predict(self, image: Image.Image):
        # 1. Run OCR to get text and positions
        words, boxes = self._run_ocr(image)
        width, height = image.size
        
        # 2. Normalize boxes
        normalized_boxes = self._normalize_boxes(boxes, width, height)
        
        # 3. Tokenize and predict
        inputs = self.tokenizer(
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            is_split_into_words=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 4. Process predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_predictions = []
        
        # Map predictions to original words
        word_ids = inputs.word_ids()
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            word_predictions.append(self.id2label[predictions[idx]])
            previous_word_idx = word_idx
        
        # 5. Extract structured data
        result = self._process_entities(words, word_predictions[:len(words)])
        return result