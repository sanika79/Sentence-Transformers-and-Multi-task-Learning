## PyTorch implementation that demonstrates how to apply layer-wise learning rates for a multi-task sentence transformer model:

import torch.nn as nn
import torch
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel


class DistilBERTForSentenceClassification(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTForSentenceClassification, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):

        ## Extract DistillBERT's outputs

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size] OR num of sentences x 768
        
        logits = self.classifier(cls_embedding)   # Shape: [batch_size, num_classes]
        return logits


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


num_classes = 2     ### for 2 sentence classes
model = DistilBERTForSentenceClassification(num_classes)


## consider input sentences that belong to 2 different classes
sentences = ["I was very happy when I visited San Diego", "The weather is bad today."]


## The inputs are tokenized into a fixed-length sequence
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Define layer-wise learning rates
optimizer_grouped_parameters = [
    {'params': model.distilbert.embeddings.parameters(), 'lr': 1e-5},  ## embedding layer
    {'params': model.distilbert.transformer.layer[:2].parameters(), 'lr': 2e-5},  ## lower layer
    {'params': model.distilbert.transformer.layer[2:4].parameters(), 'lr': 3e-5}, ## lower layer
    {'params': model.distilbert.transformer.layer[4:6].parameters(), 'lr': 4e-5}, ## higher layer
    {'params': model.classifier.parameters(), 'lr': 5e-5},     ## classification head
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

# Forward pass and optimizer step
logits = model(inputs['input_ids'], inputs['attention_mask'])
loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([1, 0]))  # Dummy target
loss.backward()
optimizer.step()
