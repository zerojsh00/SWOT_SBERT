import torch
from torch.utils.data import Dataset


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer # inherited tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True, # Depending on the mini batch. Therefore, padding=true
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels): # Read data
        self.texts = texts # Total dataset
        self.labels = labels # Labels by sample
    
    def __len__(self): # Size of the dataset
        return len(self.texts)
    
    def __getitem__(self, item): # Return samples for mini-batch
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }
