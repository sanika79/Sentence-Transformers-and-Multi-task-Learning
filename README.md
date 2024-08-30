# Sentence-Transformers-and-Multi-task-Learning

Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. This model
should be able to encode input sentences into fixed-length embeddings. Test your implementation with a
few sample sentences and showcase the obtained embeddings. 

Describe any choices you had to make regarding the model architecture outside of the transformer backbone.

Backbone used - DistilmBERT

Advantages of the Model Architechture (Why was this model chosen?)

1. Uncased model
The model used is DistilmBERT. This model was chosen because it is a version of mBERT that is uncased and does not distinguish between uppercase and lowercase letters and will treat all text as lowercase. This property is useful when the case of the input text is not important for the task that we are working on. For example, many text classification tasks.
2. Small Size
Because of its small size (66 million parameters), it is easy to store the model and it also consumes less memory. DistilBERT has 44M fewer parameters and in total is 40% smaller than BERT.
3. Faster Inference
As it is small, it performs faster model inference than other larger models and also uses less resources. During inference, DistilBERT is 60% faster than BERT.
4. Multilinguil capability
DistilmBERT is trained on multiple languages, making it suitable for multilingual tasks such as classification and named entity recognition (NER) across different languages.
5. Performance similar to mBERT
Despite its smaller size, it achieves around 97% of mBERT's performance across various tasks, making it a good trade-off between size and accuracy.



Task 2: Multi-Task Learning Expansion
Expand the sentence transformer to handle a multi-task learning setting.
Task A: Sentence Classification – Classify sentences into predefined classes (you can make these up).
Task B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.]
(you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.



Task 3
Discuss the implications and advantages of each scenario and explain your rationale as to how the model
should be trained given the following:

If the entire network should be frozen.

Implications 
1. This model uses pre-trained embeddings without further tuning for any task.
2. This scenario is assumes that the model already performs well on any specific task and does not need further improvement.
   
Advantages 
1. This scenario is useful when computational resources are limited as we just have to use the knowledge of the available pre-trained embedding weights for any given task and fewer parameters are being updated.
3. Also leads to faster inference.

Explanation
Since the entire model is frozen, the model can disrectly be used as a feature extracter, as the embeddings are directly used by the task-specific heads for text classification or any other desired task. This approach is good for small datasets where the model prevents overfitting to this limited data.

If only the transformer backbone should be frozen.

Implications 
1. This approach makes use of the pre-trainbed embeddings of the backbone while the task specific heads are fine-tuned. Th general language understanding of the BERT is preserved.
   
Advantages
1. The information from the pre-trained backbone can be extracted and passed on to the task specific heads that are fine-tuned.
2. Leads to reduction in learnable parameters

Explanation

If only one of the task-specific heads (either for Task A or Task B) should be frozen.

Implications 
1. This approach is particularly relevant in multi-task learning scenarios (as in Task 2)  where the model is used for more than one downstream task. Here, you freeze one task-specific head while allowing the other head (or heads) to be fine-tuned.
2. The frozen head will preserve its performance on one task, but the other unfrozen head will adapt to a new task.

Advantages 
1. Great for multi-task learning where large model backbone embeddings can be used for multiple tasks by fine tuning whichever head is necessary for the task at hand.
2. Serves as a great advantage when you want to maintain performance on an existing task while also adapting to a new one without losing the model's previous capabilities.

Explanation

In a multi-task learning setting, you apply in parallel the attention mechanism to multiple sets of the queue, keys and values that you can get by transforming the original embeddings. In multi-head attention, the number of times that you apply the attention mechanism is the number of heads in the model. For instance, you will need two sets of queries, keys and values in a model with two heads. The first head would use a set of representations and the second head would use a different set.  Using different sets of representations, allow your model to learn multiple relationships between the words from the query and key matrices


Consider a scenario where transfer learning can be beneficial. Explain how you would approach the
transfer learning process, including:
The choice of a pre-trained model.
The layers you would freeze/unfreeze.
The rationale behind these choices.

Task 4: Layer-wise Learning Rate Implementation (BONUS)
Implement layer-wise learning rates for the multi-task sentence transformer. Explain the rationale for the
specific learning rates you&#39;ve set for each layer. Describe the potential benefits of using layer-wise
learning rates for training deep neural networks. Does the multi-task setting play into that?
