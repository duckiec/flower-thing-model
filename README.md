# flower-thing-model

Deep learning model for flower classification with automatic GPU optimization.

Created by [duckiec](https://github.com/duckiec/flower-thing-model)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/duckiec/flower-thing-model.git
cd flower-thing-model

# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run everything with one command
python src/train.py
```

## What it does

1. **Auto-detects** your hardware (CPU/GPU)
2. **Downloads** flower datasets
3. **Trains** an efficient model
4. **Generates** visualizations and results

## Features

- 🚀 Single command operation
- 🎯 Automatic GPU/CPU optimization
- 🔄 Efficient training pipeline
- 📊 Clear visualizations
- 💾 Saves best model automatically

## Results

Find your results in:
- `model/` - Trained model
- `plots/` - Training graphs
- `results/` - Evaluation metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU optional (automatically uses CPU if no GPU)
