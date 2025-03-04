# Universal Model Adapter Fusion (UMAF) - Capability Extractor

This is a production-ready implementation of the Universal Model Adapter Fusion (UMAF) Capability Extractor, designed to create a universal representation of model capabilities across diverse architectures and domains.

## Features

- Extract semantically rich, architecture-agnostic model capability fingerprints
- Enable cross-model capability transfer and comparison
- Provide a flexible, scalable approach to understanding model capabilities
- Support for various similarity metrics (Cosine, KL Divergence, Pearson)
- Comprehensive monitoring and logging functionality

## Requirements

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers
- NumPy
- SciPy
- scikit-learn (optional for visualization)

## Usage

```python
from umaf.extractor import CapabilityExtractor
from umaf.config import CapabilityExtractorConfig
from transformers import AutoModel, AutoTokenizer

# Initialize the extractor
config = CapabilityExtractorConfig(input_dim=768, output_dim=512)
extractor = CapabilityExtractor(config)

# Load a model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
text = "Example text for capability extraction"
inputs = tokenizer(text, return_tensors="pt")

# Extract capability fingerprint
fingerprint = extractor.extract_fingerprint(model, inputs)
```