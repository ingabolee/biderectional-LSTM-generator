# Bidirectional LSTM Text Generator

An advanced Deep Learning solution leveraging TensorFlow and Keras to produce highly contextual machine-generated text using a Bidirectional Long Short-Term Memory (LSTM) neural network.

## Overview
As opposed to traditional unidirectional networks that only evaluate preceding tokens, this project highlights the capabilities of Bi-LSTMs by scanning sequences in both forward and backward directions. This allows the model to learn deep contextual relationships within natural language. Utilizing TensorFlow and robust text tokenization strategies, the model learns n-gram representations of an input corpus to continuously predict and generate comprehensive string continuations.

## Features
- Tokenization of textual corpus data leveraging Keras `Tokenizer`.
- N-gram sequence restructuring to prepare data for supervised sequence prediction.
- Deployment of a Bidirectional LSTM hidden layer.
- Sequential pattern learning and forward inference capabilities.
- Implementation of standard TensorFlow/Keras deep learning pipelines entirely contained within Jupyter.

## Tech Stack
- Python
- TensorFlow
- Keras
- Jupyter Notebook

## Project Architecture
```text
biderectional-LSTM-generator-master/
  bidirectional-LSTM-generator.ipynb    # Core notebook documenting the LSTM topology, input pipelines, and training
  input.txt                             # (Expected) Natural language corpus defining the ground truth for generation
```

## Installation
Ensure you have a Python environment set up with TensorFlow and Jupyter installed to execute the pipeline:
```bash
pip install tensorflow jupyter numpy
```

## Running the Project
Start the Jupyter Notebook server from the project directory:
```bash
jupyter notebook bidirectional-LSTM-generator.ipynb
```

## Usage Examples
The notebook automatically handles reading the text data, splitting it via the tokenizer, and fitting the data into sequence boundaries.
Example sequence compilation from the notebook:
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# Expects a UTF-8 text file as input for training sequences
data = open("input.txt", encoding="utf-8").read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
```

After training, the model's `.predict()` protocol can be leveraged over newly vectorized string seeds to recursively build out new sentences.

## Model Card

### Model Overview
The architecture is heavily reliant on a continuous sequential modeling paradigm primarily dealing with embedded text parameters to infer subsequent tokens based on combined past-and-future context windows.

### Model Architecture
- **Embedding Layer**: Transforms integer-encoded vocabularies into dense fixed-size vectors.
- **Bidirectional LSTM Layer**: Core processing block; evaluates dependencies going left-to-right and right-to-left to mitigate vanishing contexts and optimize long-range dependencies.
- **Dense Output Layer**: Softmax activation projecting onto the size of the overall vocabulary array.

### Training Process
- Text sequences are transformed into moving windows of n-grams (e.g., input shapes of length *N-1* predicting label *N*).
- Uses Categorical Crossentropy as its evaluation loss metric due to the multi-class (multi-word) nature of prediction.
- Employs Adam optimizers to adjust weights dynamically.

### Limitations
- **Vocabulary Size**: Generative scope is strictly bounded by the exact dictionary extracted from `input.txt`. Missing vocabulary is defaulted to out-of-vocab parameters.
- **Context Length Limitations**: While Bidirectional LSTMs improve over standard RNNs, they remain computationally expensive limiting contextual spans before attention mechanisms (Transformers) become optimal.

## Professional Highlights
- **Engineered an enterprise-grade NLP pipeline** by wrapping tokenization and sequence padding around Keras utilities.
- **Optimized context understanding** implementing Bidirectional recurrent layers for maximum textual coherence.
- **Mastered fundamental Natural Language Processing algorithms**, serving as a primary stepping stone towards transformer/LLM architectures.

## License
MIT License

## Contributing
Contributions are welcome. Feel free to open issues or submit pull requests for enhancements.

## Author
Lih Ingabo
