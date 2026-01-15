Chapter-1: Understanding Large Language Model

An LLM is a neural network designed to understand, generate, and respond to human-like text. These models are deep neural networks trained on massive amounts of text data, sometimes encompassing large portions of the entire publicly available text on the internet.

LLMs utilize an architecture called the transformer, which allows them to pay selective attention to different parts of the input when making predictions

Since LLMs are capable of generating text, LLMs are also often referred to as a form of generative artificial intelligence, often abbreviated as generative AI or GenAI.

The algorithms used to implement AI are the focus of the field of machine learning. Specifically, machine learning involves the development of algorithms that can learn from and make predictions or decisions based on data without being explicitly programmed.

deep learning is a subset of machine learning that focuses on utilizing neural networks with three or more layers to model complex patterns and abstractions in data. In contrast to deep learning, traditional machine learning requires manual feature extraction. This means that human experts need to identify and select the most relevant features for the model.

LLMs are invaluable for automating almost any task that involves parsing and generating text.

Using custom-built LLMs offers several advantages, particularly regarding data privacy. For instance, companies may prefer not to share sensitive data with third-party LLM providers like OpenAI due to confidentiality concerns. Additionally, developing smaller custom LLMs enables deployment directly on customer devices, such as laptops and smartphones, which is something companies like Apple are currently exploring.

Stages of building and using LLMs:
	The first step in creating an LLM is to train it on a large corpus of text data, sometimes referred to as raw text. Here, “raw” refers to the fact that this data is just regular text without any labeling information.
	In this phase, LLMs use self-supervised learning, where the model generates its own labels from the input data.
	This first training stage of an LLM is also known as pretraining, creating an initial pretrained LLM, often called a base or foundation model.

	After obtaining a pretrained LLM from training on large text datasets, where the LLM is trained to predict the next word in the text, we can further train the LLM on labeled data, also known as fine-tuning.
	The two most popular categories of fine-tuning LLMs are instruction fine-tuning and classification fine-tuning. In instruction fine-tuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. In classification fine-tuning, the labeled dataset consists of texts and associated class labels—for example, emails associated with “spam” and “not spam” labels.

Introducing the transformer architecture:
	Most modern LLMs rely on the transformer architecture, which is a deep neural network architecture introduced in the 2017 paper “Attention Is All You Need”
	The original transformer was developed for machine translation, translating English texts to German and French.
	The transformer architecture consists of two submodules: an encoder and a decoder. The encoder module processes the input text and encodes it into a series of numerical representations or vectors that capture the contextual information of the input. Then, the decoder module takes these encoded vectors and generates the output text. In a translation task, for example, the encoder would encode the text from the source language into vectors, and the decoder would decode these vectors to generate text in the target language. Both the encoder and decoder consist of many layers connected by a so-called self-attention mechanism.

	A key component of transformers and LLMs is the self-attention mechanism, which allows the model to weigh the importance of different words or tokens in a sequence relative to each other. This mechanism enables the model to capture long-range dependencies and contextual relationships within the input data, enhanc- ing its ability to generate coherent and contextually relevant output.

	BERT, which is built upon the original transformer’s encoder submodule, differs in its training approach from GPT. While GPT is designed for generative tasks, BERT and its variants specialize in masked word prediction, where the model predicts masked or hidden words in a given sentence

	GPT, on the other hand, focuses on the decoder portion of the original transformer architecture and is designed for tasks that require generating texts. This includes machine translation, text summarization, fiction writing, writing computer code, and more. 
	they are considered a type of autoregressive model. Autoregressive models incorporate their previous outputs as inputs for future predictions. Consequently, in GPT, each new word is chosen based on the sequence that precedes it, which improves the coherence of the resulting text.


