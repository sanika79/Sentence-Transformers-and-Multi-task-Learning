##Expand the Model for Sentence Classification

## Added a linear layer on top of the CLS token embedding.
## Used the output from the CLS token as the representation of the whole sentence.

import torch.nn as nn
import torch
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel


class DistilBERTForSentenceClassification(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTForSentenceClassification, self).__init__()

        ## DistilmBERT and DistilBERT model have the same backbone except the data on which DistillmBERT is trained on. 
        ## Hence, we can directly import the pre-trained DistilBert model.

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        ## Because we are fine tuning , the linear layer is the learnable parameter of the model.

        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):

        ## Extract DistillBERT's outputs

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        ## Input IDs: Numerical representation of tokens in the input text.
        # Attention Masks: Binary mask indicating which tokens should be attended to and which should be ignored

        ### The classification head is a simple linear layer that maps the CLS token embedding to the number of classes.

        ## Extract the [CLS] token's hidden state (first token)

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size] OR num of sentences x 768

        ## Apply classification head
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

# Get logits for sentence classification
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])

predicted_class = torch.argmax(logits, dim=-1)
print("Predicted Class:", predicted_class)

# Map the predicted class to Positive/Negative
for i, sentence in enumerate(sentences):
    sentiment = "Positive" if predicted_class[i] == 1 else "Negative"
    print(f"Sentence: {sentence}")
    print(f"Sentence Label: {sentiment}")
    print()

