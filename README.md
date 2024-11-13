

# FAISS-LLM Interrogator Using meta/llama-3.1-nemotron-70b-instrcut
A Project By Praneeth Kilari
## Overview

FAISS-LLM Interrogator is an advanced question-answering system that combines the power of Facebook AI Similarity Search (FAISS) for efficient similarity search and retrieval with Meta's state-of-the-art Llama 3 70B Instruct language model. This project enables users to perform accurate and context-aware question answering on large document collections.

FAISS-LLM Interrogator Demo

## Features

<img width="873" alt="Screenshot 2024-11-14 at 01 03 12" src="https://github.com/user-attachments/assets/2cafb7b9-ed48-4566-ad1d-00f0f5cba781">



- **Efficient Document Indexing**: Utilizes FAISS for fast and memory-efficient indexing of large document collections.
- **Advanced Language Understanding**: Leverages the Llama 3 70B Instruct model for superior natural language processing and generation.
- **Context-Aware Responses**: Retrieves relevant document snippets to provide accurate and contextual answers.
- **Scalable Architecture**: Designed to handle large-scale document collections and high query volumes.
- **Customizable Retrieval**: Allows fine-tuning of retrieval parameters for optimal performance.
- **Multi-language Support**: Capable of processing and answering questions in multiple languages.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [License](#license)
  

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/faiss-llm-interrogator.git
   cd faiss-llm-interrogator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Llama 3 70B Instruct model:
   ```bash
   python scripts/download_model.py
   ```

5. Index your document collection:
   ```bash
   python scripts/index_documents.py --input_dir /path/to/documents
   ```

## Usage

### Command Line Interface

To use the FAISS-LLM Interrogator via the command line:

```bash
python interrogator.py --query "What is the capital of France?"
```

### Python API

```python
from faiss_llm_interrogator import Interrogator

interrogator = Interrogator()
answer = interrogator.ask("What is the capital of France?")
print(answer)
```

### Web Interface

To launch the web interface:

```bash
python web_app.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Configuration

The system can be configured by modifying the `config.yaml` file. Key configuration options include:

```yaml
faiss:
  index_type: "IVF1024,Flat"
  metric: "inner_product"
  nprobe: 64

llama:
  model_path: "models/llama3-70b-instruct"
  max_tokens: 512
  temperature: 0.7

retrieval:
  top_k: 5
  min_similarity: 0.75
```

## API Reference

### `Interrogator` Class

#### `__init__(config_path='config.yaml')`

Initializes the Interrogator with the specified configuration.

#### `ask(query: str) -> str`

Processes the given query and returns an answer based on the indexed documents.

#### `index_documents(documents: List[str])`

Indexes the provided documents for future querying.

#### `load_index(index_path: str)`

Loads a pre-built FAISS index from the specified path.

#### `save_index(index_path: str)`

Saves the current FAISS index to the specified path.

### `FAISSIndexer` Class

#### `__init__(config: Dict)`

Initializes the FAISS indexer with the given configuration.

#### `build_index(vectors: np.ndarray)`

Builds a FAISS index from the provided vectors.

#### `search(query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]`

Performs a similarity search in the FAISS index.

### `LlamaModel` Class

#### `__init__(config: Dict)`

Initializes the Llama 3 70B Instruct model with the given configuration.

#### `generate(prompt: str) -> str`

Generates a response based on the given prompt.

## Performance Benchmarks

We've conducted extensive benchmarks to evaluate the performance of FAISS-LLM Interrogator:

| Metric                   | Value     |
|--------------------------|-----------|
| Average Query Time       | 0.25s     |
| Accuracy (QA Benchmark)  | 92.5%     |
| Memory Usage             | 12GB      |
| Max Documents Indexed    | 10 million|

For detailed benchmark results and methodology, please refer to our [Benchmark Report](docs/benchmark_report.md).
## Model Representation And Score
Response Time iS Attested Below 

<img width="1320" alt="Screenshot 2024-11-13 at 19 49 43" src="https://github.com/user-attachments/assets/15352d78-902a-428a-a6ed-46e610e0cefa">

Content Accuracy Is Very High

<img width="1727" alt="Screenshot 2024-11-13 at 19 01 33" src="https://github.com/user-attachments/assets/cfef7dcd-796c-4ed5-925f-e8245c7560b3">

<img width="1072" alt="Screenshot 2024-11-14 at 01 02 10" src="https://github.com/user-attachments/assets/27ab9b11-3f65-4590-ac8a-0afdcd05219e">



Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Meta AI](https://ai.meta.com/) for the Llama 3 70B Instruct model
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- All contributors who have helped improve this project

