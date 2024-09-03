## Sentence-Transformers-and-Multi-task-Learning

# Installation

Requires python 3.7+

Install dependencies

**pip install requirements.txt**

## Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. This model
should be able to encode input sentences into fixed-length embeddings. Test your implementation with a
few sample sentences and showcase the obtained embeddings. 

**Refer to fixed_length_embeddings.py**

Describe any choices you had to make regarding the model architecture outside of the transformer backbone.

# Backbone used - DistilmBERT

**Choices regarding the Model Architechture (Advantages of this model)**

**1. Uncased model**

The model used is DistilmBERT. This model was chosen because it is a version of mBERT that is uncased and does not distinguish between uppercase and lowercase letters and will treat all text as lowercase. This property is useful when the case of the input text is not important for the task that we are working on. For example, many text classification tasks.

**2. Small Size**

Because of its small size (66 million parameters), it is easy to store the model and it also consumes less memory. DistilBERT has 44M fewer parameters and in total is 40% smaller than BERT.

**3. Faster Inference**

As it is small, it performs faster model inference than other larger models and also uses less resources. During inference, DistilBERT is 60% faster than BERT.

**4. Multilinguil capability**

DistilmBERT is trained on multiple languages, making it suitable for multilingual tasks such as classification and named entity recognition (NER) across different languages.

**5. Performance similar to mBERT**

Despite its smaller size, it achieves around 97% of mBERT's performance across various tasks, making it a good trade-off between size and accuracy.



## Task 2: Multi-Task Learning Expansion
Expand the sentence transformer to handle a multi-task learning setting.

**Task A: Sentence Classification – Classify sentences into predefined classes (you can make these up).**

**Refer to sentence_classification.py**

**Task B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.]
(you can make the labels up)**

**Refer to named_entity_recognition.py**

**Describe the changes made to the architecture to support multi-task learning.**

**Changes made for Classification**

The same BERT model is used for both sentence embedding and classification, but with an additional linear layer for classification.

We use the embedding of the [CLS] token as the fixed-length representation of the sentence and add a head (linear layer) to output logits corresponding to the number of classes.

**Changes made for NER**

To expand the existing model to handle Named Entity Recognition (NER), we need to modify the architecture to include a token classification head instead of a sentence classification head. 

It consists of a linear layer on the top of each token's hidden states to predict NER labels for each token (e.g., B-PER, I-ORG, etc.).

**MODEL TESTING**

The linear layer in both the tasks has not been trained. Since this is a model definition without training, the predictions might not be meaningful until the model is fine-tuned. 

**NOTE** 

This was because training a model was not expected for this assignment.


## Task 3
Discuss the implications and advantages of each scenario and explain your rationale as to how the model
should be trained given the following:

## If the entire network should be frozen.

**Implications** 

1. This model uses pre-trained embeddings without further tuning for any task.
2. This scenario is assumes that the model already performs well on any specific task and does not need further improvement.
   
**Advantages**

1. This scenario is useful when computational resources are limited as we just have to use the knowledge of the available pre-trained embedding weights for any given task and fewer parameters are being updated.
3. Also leads to faster inference.

**Explanation**

Since the entire model is frozen, the model can disrectly be used as a feature extracter, as the embeddings are directly used by the task-specific heads for text classification or any other desired task. This approach is good for small datasets where the model prevents overfitting to this limited data.

## If only the transformer backbone should be frozen.

**Implications**  
1. This approach makes use of the pre-trainbed embeddings of the backbone while the task specific heads are fine-tuned. Th general language understanding of the BERT is preserved.
   
**Advantages**

1. The information from the pre-trained backbone can be extracted and passed on to the task specific heads that are fine-tuned.
2. Leads to reduction in learnable parameters
3. Reduces training time.
4. Smaller dataset

**Explanation**

Considering a frozen backbone, we are applying transfer learning which will alllow us to improve predictions because you'll learn a few things from different tasks that might be helpful and useful for your currents predictions on the task you're training on in NLP.
And since your model has already learnt a lot , you will need less additional data to fine-tune the heads.


## If only one of the task-specific heads (either for Task A or Task B) should be frozen.

**Implications** 
1. This approach is particularly relevant in multi-task learning scenarios (as in Task 2)  where the model is used for more than one downstream task. Here, you freeze one task-specific head while allowing the other head (or heads) to be fine-tuned.
2. The frozen head will preserve its performance on one task, but the other unfrozen head will adapt to a new task.

**Advantages** 

1. Great for multi-task learning where large model backbone embeddings can be used for multiple tasks by fine tuning whichever head is necessary for the task at hand.
2. Serves as a great advantage when you want to maintain performance on an existing task while also adapting to a new one without losing the model's previous capabilities.

**Explanation**

In a multi-task learning setting, you apply in parallel the attention mechanism to multiple sets of the queue, keys and values that you can get by transforming the original embeddings. In multi-head attention, the number of times that you apply the attention mechanism is the number of heads in the model. For instance, you will need two sets of queries, keys and values in a model with two heads. The first head would use a set of representations and the second head would use a different set.  Using different sets of representations, allow your model to learn multiple relationships between the words from the query and key matrices.

Since each head uses different linear transformations to represent words, different heads learn different relationships between words. Hence, by using only 1 head and freezing the other, only those representations and patterns of the unfrozen head will be transferred by making it easy for the model to adapt to the task at hand.


## Consider a scenario where transfer learning can be beneficial. Explain how you would approach the transfer learning process, including:

**The choice of a pre-trained model.**

In general, transfer learning scenarios use pre-trained models so that we can easily extract feature based word embeddings or perform fine-tuning on downstream tasks. 

**Feature based Transfer Learning**

You learn word embeddings by training one model and then you use those word embeddings in a different model for a different task.

**Fine tuning**

In this, you can use the exact same model and use it on a different task or you can keep the model weights fixed and just add a new layer that you will train or you can slowly unfreeze the layers one at a time. Therefore, we can use pre-training tasks like language modeling, mask sentence or next sentence for our model. For example, a model that is pre-trained to predict movie reviews is fine-tuned to predict course reviews.

BERT models are usualy pre-trained on unlabeled data and fine-tuned on labeled data for some downstream task and use Next sentence prediction and mask language modeling during pre-training. 

Keeping these concepts in mind, I chose DistillmBERT for the above tasks - sentence classification and Named Entity Recognition. We have a sentence classifier model of DistillmBERT which is already pre-trained with a large corpus of data. This pre-trained model is uncased and multilingual. Hence, can be fine-tuned on a variety of different use cases especially for multi-task learning as the above task demands.

**The layers you would freeze/unfreeze**

In transfer learning implemented for Language models, there are 2 strategies used for freezing of layers 

**1. Gradual unfreezing**

Unfreeze one layer at a time where you unfreeze the last one then fine-tune using that and keep the others fixed then unfreeze the next one, and then you fine tune using that and similarly you keep unfreezing each layer.

**2. Adaptive layers**

You add feed-forward networks to each block of the transformer and only let these new layers train.

**3. PEFT Technique of transfer learning**

Many a times, PEFT - Parameter Efficient Fine Tuning is designed to fine-tune large pre-trained models with minimum additional parameters. This way we can adapt a large model to a new task without modifying all the original model's parameters and saves on computational resources.

**- LORA - Low Rank Adaptation** 

This is a PEFT tecchnique where low-rank matrices are added to the weight matrices of the model, usually between layers or inside the attention mechanism. The other parameters of the model remain frozen and only the low rank matrices are updated during fine-tuning.

This reduces number of trainable parameters, usues less resources and can be used for multi-task learning.

**- Adapters**

Small additional layers are added between the layers of a pre-trained model and only these layers are trained while the other layers remain frozen.

Some key decisions were made based on the various techniqes described above (Adapters, PEFT,etc). How to structure the Task specific heads was also important as the 2 tasks were very different (Classification and NER)

Hence, these are various methods of transfer learning that were kept in mind for multi-task learning.

**The rationale behind these choices**

The training strategies for a large model like BERT while applying transfer learning were kept in mind while fine-tuning the model for multi-task learning.

In the tasks above, the initial layers of BERT were kept frozen to preserve the learned representations while the upper layers were fine tuned for the specific task. This way most of the parameters were shared across all the tasks.

Task specific heads design

**Classification Head :**

For the sentence classification task, a task-specific head (usually a fully connected layer followed by a softmax or sigmoid activation) is added on top of BERT’s output (the [CLS] token) to output logits corresponding to the number of classes.

**NER Head :**

For NER, a token classification head is added which consists of a linear layer on the top of each token's hidden states to predict NER labels for each token (e.g., B-PER, I-ORG, etc.).


## Task 4: Layer-wise Learning Rate Implementation (BONUS)
Implement layer-wise learning rates for the multi-task sentence transformer. 

**Refer to layer_wise_learning.py**

Explain the rationale for the specific learning rates you have set for each layer. Describe the potential benefits of using layer-wise
learning rates for training deep neural networks. Does the multi-task setting play into that?

**Rationale for specific learning rates**

*Top Layers (Task Specific Heads)*

This includes the task specific heads which are generally fine tuned to any downstream task. They need most significant updates to their learning to adapt the pre-trained model to new tasks. Hence, higher learning rates are set for these layers

*Middle Layers (Intermediate transformer layers)*

These start capturing more task-specific features and extract more context than lower layers making them more relevant to the particular tasks you're fine-tuning the model for. Hence, a moderate learning rate would suffice for these layers.

*Lower layers (Embeddings and lower transformer layers)*

These layers capture basic language patterns and word embeddings, which are generally transferable to downstream tasks. Since these layers are already well-trained on large datasets (e.g., during the pre-training of DistilmBERT), they often require less aggressive updating. The embedding layer is very close to the input, where basic token representations are created. We set a low learning rate here to avoid drastic changes to these fundamental representations.

**Potential benefits of using layer-wise learning rates**

**Stable training**

By applying smaller learning rates to lower layers, we prevent the model from losing useful pre-trained features, leading to more stable training.

**Preserve pretrianed knowledge**

When using pre-trained models like BERT, the lower layers often contain general linguistic knowledge. Smaller learning rates for these layers help preserve this knowledge, allowing the model to leverage its pre-training effectively.

**Faster convergence**

Higher layers in the model, which are more task-specific, can be adapted more rapidly to new tasks with higher learning rates. This enables faster convergence for task-specific fine-tuning, as these layers require more significant updates to specialize in the new task.

**Impact on validation loss (Early stopping)**

Training with different learning rates reduces the Val loss earlier than fixed learning rates and it achieves the early stopping state much earlier. This will help us to stop the training much sooner and help us to avoid overfitting.


## Benefit for multi-task setting

Multi-task setting benefits from layer-wise learning rates because different tasks may utilize or leverage different parts of the model. 

For example, the classifier head may need to adapt quickly to the specific labels of one task (e.g., sentence classification), while the backbone might need only minor changes to accommodate a new task like NER. 

Using layer-wise learning rates allows fine-tuning of the model in a way that balances the learning needs of all tasks involved.







