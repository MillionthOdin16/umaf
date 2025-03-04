# Universal Model Adapter Fusion (UMAF) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
The **Universal Model Adapter Fusion (UMAF)** project is an experimental framework designed to explore capability transfer and integration across large language models (LLMs) with varying architectures and scales. The aim is to develop a method for extracting and adapting specialized capabilities—such as reasoning or sentiment analysis—from source models to enhance a target model efficiently, using lightweight adapters instead of full retraining. This repository represents an initial exploration of a modular, scalable, and interpretable system, with ongoing work to expand its components.

### Purpose
UMAF investigates the feasibility of combining diverse model strengths into a unified framework. It focuses on laying the groundwork for capability extraction and transfer, providing a starting point for future research and practical applications in AI model adaptation. The current implementation includes only the **Capability Extractor**, with additional components planned for later development. For a detailed overview of the full framework, see the [Full Framework Specification](UMAF_Framework_Specification.md).

## Framework Visualization

```plaintext
+-------------------------+       +-------------------------+
| Source Model(s)         |       | Target Model            |
| (e.g., BERT, LLaMA)     |       | (e.g., BERT)            |
+-------------------------+       +-------------------------+
          |                            |
          | Activations                |
          v                            |
+-------------------------+            |
| Capability Extractor    |            |
| - Input Processor       |            |
| - Transformer Encoder   |            |
| - Projection Head       |            |
| - Similarity Computation|            |
| Output: Fingerprint     |            |
+-------------------------+            |
          |                            |
          | 512D Fingerprint           |
          v                            |
+-------------------------+            |
| Size Interpolator       |            |
| (Planned)               |            |
| - Adjusts for scale     |            |
+-------------------------+            |
          |                            |
          | Adjusted Fingerprint       |
          v                            |
+-------------------------+            |
| Latent Fusion Module    |            |
| (Planned)               |            |
| - Combines fingerprints |            |
+-------------------------+            |
          |                            |
          | Fused Fingerprint          |
          v                            |
+-------------------------+            |
| Adapter Generator       |            |
| (Planned)               |            |
| - Creates LoRA adapters |            |
+-------------------------+            |
          |                            |
          | LoRA Adapter               |
          v                            v
+-------------------------+       +-------------------------+
|                         |       | Enhanced Target Model   |
|                         |<------+ - Applies adapter       |
+-------------------------+       +-------------------------+
```

## Capability Extractor
The **Capability Extractor** is the initial component of UMAF, developed to process model activations and produce a 512-dimensional capability fingerprint. This fingerprint is intended to represent a model’s functional strengths in a way that is independent of its architecture, serving as a foundation for capability comparison and transfer experiments.

### Features
- Generates capability fingerprints from model activations.
- Designed to work with various model architectures (e.g., BERT, LLaMA).
- Uses a configurable transformer-based architecture for activation processing.
- Includes similarity metrics: Cosine Similarity, KL Divergence, and Pearson Correlation.
- Provides monitoring and logging for experimental evaluation.

### Requirements
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **HuggingFace Transformers**
- **NumPy**
- **SciPy**
- **scikit-learn** (optional, for visualization)

### Installation
To set up the current implementation:
```bash
git clone https://github.com/millionthodin16/umaf.git
cd umaf
pip install -r requirements.txt
```

### Usage
The following example demonstrates how to use the Capability Extractor with a BERT model:
```python
from umaf.extractor import CapabilityExtractor
from umaf.config import CapabilityExtractorConfig
from transformers import AutoModel, AutoTokenizer

# Initialize the extractor with configuration
config = CapabilityExtractorConfig(input_dim=768, output_dim=512)
extractor = CapabilityExtractor(config)

# Load a pre-trained model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Process input text
text = "Sample text for capability analysis"
inputs = tokenizer(text, return_tensors="pt")

# Extract the capability fingerprint
fingerprint = extractor.extract_fingerprint(model, inputs)
print(f"Capability Fingerprint Shape: {fingerprint.shape}")  # Outputs: torch.Size([1, 512])
```

### Training the Extractor
To experiment with training the extractor:
1. Prepare a list of models and corresponding datasets.
2. Use `CapabilityDataset` to organize training data.
3. Apply the `info_nce_loss` function for contrastive learning (see specification for details).

Example training code:
```python
from umaf.dataset import CapabilityDataset
from torch.utils.data import DataLoader

# Prepare models and datasets
dataset = CapabilityDataset(models_list, datasets_list, tokenizer, config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop (simplified)
for epoch in range(5):
    for inputs, model_idx, _ in dataloader:
        # Add training logic here
        pass
```

### Monitoring
Track experimental metrics with `CapabilityExtractionMonitor`:
```python
from umaf.monitor import CapabilityExtractionMonitor

monitor = CapabilityExtractionMonitor()
monitor.log_metrics(fingerprint_clustering=0.6, training_time_per_epoch=3600)
report = monitor.generate_report()
print(report)
```

## Project Status and Roadmap
- **Current Implementation**: Capability Extractor (v1.0, initial version).
- **Planned Components**: 
  - Size Interpolator for handling models of different scales.
  - Latent Fusion Module for combining capabilities.
  - Adapter Generator for creating model adapters.
- **Timeline**: This is an early-stage implementation; further development is planned but not yet scheduled. Future components will build on insights from this initial work.

## License
This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Citation
If you use this implementation in your work, please cite it as:
```
@misc{umaf2025,
  title = {Universal Model Adapter Fusion (UMAF): Capability Extractor},
  author = {millionthodin16},
  year = {2025},
  url = {https://github.com/millionthodin16/umaf},
  note = {Initial implementation of the UMAF framework}
}
```
