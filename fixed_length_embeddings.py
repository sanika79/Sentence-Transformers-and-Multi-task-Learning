import torch
from transformers import DistilBertModel, DistilBertTokenizer


## I decided to use a DistilmBert model as the sentence transformer 
## Since DistilBERT and DistilmBERT have the same model architechure, we can use either of them

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

def encode_sentences(sentences):
    # Tokenize the input sentences and return them in the form of a pytorch tensor
    # the sequence of tokens is created
    # padding will make sure all input sequences are of the same length 
    # truncation will ensure sequences do not exceed a maximum length

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Obtain the fixed-length embeddings (pooled output)
    # Pass the tokenized inputs through the DistilBERT model
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Extract the [CLS] token's output as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

        ###Embeddings shape: torch.Size([3, 768])
     
    return embeddings

# Test sentences
# We pass sentences which contain a either a mix of uppercase and lower case OR all lower case 
# characters since we are using an uncased model
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I love the weather in New York City.",
    "my parents are visiting me this month."
]

# Get the embeddings
embeddings = encode_sentences(sentences)

# Convert the embeddings to numpy for better readability
embeddings = embeddings.numpy()


# Print the embeddings
for i, embedding in enumerate(embeddings):
    print(f"Sentence: {sentences[i]}")
    print(f"Embedding: {embedding}\n")
