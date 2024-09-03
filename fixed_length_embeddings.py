import torch
from transformers import DistilBertModel, DistilBertTokenizer

## I decided to use a DistilmBert model as the sentence transformer 
## Since DistilBERT and DistilmBERT have the same model architechure, we can use either of them

class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super(SentenceTransformerModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        # Get the last hidden state (outputs) from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # We will use the [CLS] token's embedding as the sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = SentenceTransformerModel()

# Test sentences
# We pass sentences which contain a either a mix of uppercase and lower case OR all lower case 
# characters since we are using an uncased model

sentences = [
    "This is an example sentence",
    "I love the weather in New York City.",
    "my parents are visiting me this month."
]

# Tokenize the input sentences and return them in the form of a pytorch tensor
# the sequence of tokens is created
# padding will make sure all input sequences are of the same length 
# truncation will ensure sequences do not exceed a maximum length

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)


# Pass the tokenized inputs through the DistilBERT model
# Obtain the fixed-length embeddings 

with torch.no_grad():
    sentence_embeddings = model(inputs['input_ids'], inputs['attention_mask'])

    ## sentence embeddings shape - number of sentences (input batch size) x size of CLS token's embedding (768 in case of DistilBERT)

## The default hidden size for the distilbert-base-uncased model is 768.
## Therefore, the size of the fixed-length embeddings output by the model will be a vector of size 768.

# Print the embeddings
for i, embedding in enumerate(sentence_embeddings):
    print(f"Sentence: {sentences[i]}")
    print(f"Embedding size : {len(embedding)}")
    print(f"Embedding: {embedding}\n")
