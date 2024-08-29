# Sentence-Transformers-and-Multi-task-Learning

Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. This model
should be able to encode input sentences into fixed-length embeddings. Test your implementation with a
few sample sentences and showcase the obtained embeddings. Describe any choices you had to make
regarding the model architecture outside of the transformer backbone.

Backbone used - DistilmBERT

Advantages of the Model Architechture (Why was this model chosen?)
1. The model used is DistilmBERT. This model was chosen because it is a version of mBERT that is uncased and does not distinguish between uppercase and lowercase letters and will treat all text as lowercase. 
This property is useful when the case of the input text is not important for the task that we are working on. For example, many text classification tasks.
2. Because of its small size (66 million parameters), it is easy to store the model and it also consumes less memory.
3. As it is small, it performs faster model inference than other larger models and also uses less resources.
4. So, DistilmBERT uncased can be a good choice if you're working on a multilingual task and want a model that is smaller and faster than mBERT, with the added benefit of being uncased.


Task 2: Multi-Task Learning Expansion
Expand the sentence transformer to handle a multi-task learning setting.
Task A: Sentence Classification – Classify sentences into predefined classes (you can make these up).
Task B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.]
(you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.




DistilmBERT, a distilled version of the multilingual BERT (mBERT), offers several advantages over other similar models, particularly in the context of multilingual natural language processing tasks. Here are some key advantages:

1. Smaller Model Size
Reduced Parameters: DistilmBERT has about 66 million parameters, compared to 110 million in mBERT. This smaller size makes it more efficient in terms of storage and memory usage.
Faster Inference: Due to its smaller size, DistilmBERT can perform inference faster than larger models, which is crucial for real-time applications or when deploying on resource-constrained devices.
2. Comparable Performance
Maintained Accuracy: Despite being a smaller model, DistilmBERT retains a significant portion of the performance of mBERT. It achieves around 97% of mBERT's performance across various tasks, making it a good trade-off between size and accuracy.
3. Multilingual Capability
Support for Multiple Languages: Like mBERT, DistilmBERT is trained on multiple languages, making it suitable for multilingual tasks such as classification and named entity recognition (NER) across different languages.
Cross-Lingual Generalization: DistilmBERT retains the cross-lingual capabilities of mBERT, which is useful for tasks involving multiple languages without requiring separate models.
4. Efficiency in Training and Deployment
Faster Training: The reduced number of parameters allows DistilmBERT to train faster compared to larger models like mBERT or XLM-R, which can be advantageous when working with large datasets or multiple iterations.
Lower Computational Costs: With fewer parameters and reduced complexity, DistilmBERT requires less computational power for both training and inference, reducing overall operational costs.
5. Flexibility and Integration
Compatibility with Existing Frameworks: DistilmBERT can be easily integrated into popular NLP frameworks like Hugging Face's Transformers, making it accessible and easy to use for a wide range of applications.
Pretrained Versions Available: There are pretrained versions of DistilmBERT available, allowing for quick implementation and fine-tuning on specific tasks without needing to train from scratch.
6. Energy Efficiency
Lower Energy Consumption: With fewer parameters and faster processing, DistilmBERT consumes less energy, which is beneficial for sustainable AI practices and running models on edge devices.
7. Versatility
Suitable for Diverse Tasks: DistilmBERT is versatile and can be used for a wide range of NLP tasks beyond just classification and NER, such as sentiment analysis, translation, and question answering, across different languages.
In summary, DistilmBERT offers a balanced approach by providing the advantages of multilingual capabilities and high performance, while being more efficient and faster than larger models like mBERT or XLM-R. This makes it an attractive option for applications where both accuracy and efficiency are important.

Task 3
Discuss the implications and advantages of each scenario and explain your rationale as to how the model
should be trained given the following:
If the entire network should be frozen.
If only the transformer backbone should be frozen.
If only one of the task-specific heads (either for Task A or Task B) should be frozen.
