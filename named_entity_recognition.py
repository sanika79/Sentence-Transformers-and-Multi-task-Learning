### Expand the Model for Named Entity Recognition (NER)

## Added a linear layer on top of each token embedding.
## Used the output for each token to predict its label (e.g., B-PER, I-PER, O, etc.).
## this model will output logits for each token, which will be used to classify the token into one of the NER labels.

import torch
from transformers import DistilBertModel, DistilBertTokenizer

class NERMultiTaskModel(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=7):
        super(NERMultiTaskModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        # Adding a token classification head
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get the last hidden state (outputs) from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get all token embeddings
        token_embeddings = outputs.last_hidden_state
        #  The classifier now predicts labels for each token in the sequence.
        logits = self.classifier(token_embeddings)
        ## Output shape torch.size([2x8x7])
        ## Batch size (num of sentences) x sequence length x number of labels
        return logits

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = NERMultiTaskModel(num_labels=7)

# Example sentences
sentences = ["I love the weather in San Diego", "Italian food is my favourite cuisine."]

# Tokenize the sentences
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, is_split_into_words=False)

# Get logits for NER
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])

# Example NER label list
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

# For each token in the sentence, the model predicts an NER label.
predictions = torch.argmax(logits, dim=-1)

# Convert token predictions back to words
for i, sentence in enumerate(sentences):
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
    pred_labels = [label_list[pred] for pred in predictions[i].cpu().numpy()]
    print(f"Sentence: {sentence}")
    print("Token predictions:")
    for token, label in zip(tokens, pred_labels):
        print(f"{token}: {label}")
    print()
