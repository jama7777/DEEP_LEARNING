# Enterprise Small Language Model (SLM) - Complete Guide

A production-ready implementation for building and training Small Language Models on enterprise data sources including SharePoint, Google Drive, Slack, GitHub, and local documents.

## 🎯 Features

- **Fully Private & On-Premise**: No external API dependencies, complete data control
- **Heterogeneous Data Support**: Documents (PDF, Word, Excel), code, chat logs, transcripts
- **Multilingual**: Supports 8+ languages with domain-specific vocabulary
- **Memory Efficient**: Optimized for single GPU training (8-16GB VRAM)
- **Production Ready**: Includes checkpointing, evaluation metrics, logging, and deployment scripts
- **Modular Design**: Easy to customize and extend

---

## 📋 Project Structure

```
enterprise-slm/
├── config.py              # Central configuration file
├── model.py              # Transformer model architecture
├── data_loader.py        # Data loading and preprocessing
├── tokenizer.py          # Custom BPE tokenizer
├── dataset.py            # PyTorch dataset classes
├── train.py              # Main training script
├── inference.py          # Inference and generation
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── data/
│   ├── raw/             # Place source documents here
│   ├── processed/       # Auto-generated processed data
│   └── cache/           # Temporary cache
│
├── models/
│   ├── checkpoints/     # Training checkpoints
│   └── tokenizer.pkl    # Trained tokenizer
│
└── logs/
    ├── training.log     # Training logs
    └── tensorboard/     # TensorBoard logs
```

---

## 🚀 Quick Start (5 Steps)

### Step 1: Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
# Create data directory
mkdir -p data/raw

# Add your documents (PDF, Word, TXT, code files, etc.)
# Example:
cp /path/to/your/documents/* data/raw/
```

### Step 3: Configure (Optional)

Edit `config.py` to customize:
- Model size
- Training hyperparameters
- Data sources
- Hardware settings

### Step 4: Train Model

```bash
# Start training
python train.py

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir=logs/tensorboard
```

### Step 5: Run Inference

```bash
# Test the model
python inference.py
```

---

## 💻 Running in VS Code

### Initial Setup

1. **Open Project**
```bash
code .
```

2. **Select Python Interpreter**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Python: Select Interpreter"
   - Select your virtual environment

3. **Install Extensions**
   - Python (Microsoft)
   - Pylance
   - TensorBoard (optional)

### Running Scripts

#### Option 1: Using Terminal
```bash
# In VS Code terminal (Ctrl+`)
python train.py
python inference.py
python data_loader.py  # To process data only
```

#### Option 2: Using Run Button
- Open any script (e.g., `train.py`)
- Click the ▶️ Run button in top-right
- Or press `F5`

#### Option 3: Using Debug
- Set breakpoints by clicking left of line numbers
- Press `F5` to start debugging
- Use debug controls to step through code

---

## 📊 Dataset Requirements

### Minimum Requirements
- **Size**: 1,000+ documents or 1M+ tokens
- **Format**: Any supported file type
- **Quality**: Clean, relevant text

### Recommended
- **Size**: 10,000+ documents or 10M+ tokens
- **Diversity**: Mix of document types
- **Quality**: Deduplicated, properly formatted

### Supported File Types

| Category | Formats |
|----------|---------|
| Documents | .txt, .md, .pdf, .docx, .doc |
| Spreadsheets | .xlsx, .xls, .csv |
| Code | .py, .js, .java, .cpp, .html, .css |
| Data | .json, .xml |

### Data Organization

```
data/raw/
├── documents/          # Business documents
│   ├── reports/
│   ├── memos/
│   └── policies/
├── code/               # Code repositories
│   ├── python/
│   └── javascript/
├── chat/               # Chat/Slack logs
└── transcripts/        # Meeting transcripts
```

---

## 🎓 Training Process

### What Happens During Training

1. **Data Loading** (5-10 minutes)
   - Scans data/raw/ directory
   - Extracts text from all supported files
   - Creates processed documents

2. **Tokenizer Training** (10-30 minutes)
   - Learns vocabulary from your data
   - Creates subword units (BPE)
   - Saves to models/tokenizer.pkl

3. **Model Training** (hours to days)
   - Trains transformer model
   - Saves checkpoints every N steps
   - Evaluates on validation set
   - Logs metrics to TensorBoard

### Training Configuration

Key parameters in `config.py`:

```python
MODEL_CONFIG = {
    "d_model": 512,        # Model size (smaller=faster)
    "n_layers": 6,         # Depth (fewer=faster)
    "max_seq_length": 512, # Context window
}

TRAINING_CONFIG = {
    "batch_size": 16,      # Reduce if OOM
    "num_epochs": 10,      # More=better (usually)
    "learning_rate": 5e-4, # Adjust if not converging
}
```

### Monitor Training

#### TensorBoard (Recommended)
```bash
tensorboard --logdir=logs/tensorboard
# Open http://localhost:6006
```

#### Log File
```bash
tail -f logs/training.log
```

#### Console Output
Watch for:
- Loss decreasing over time
- Perplexity getting lower
- No OOM errors

### Expected Timeline

| Dataset | Hardware | Time |
|---------|----------|------|
| 1K docs | RTX 3090 | 2-4 hours |
| 10K docs | RTX 3090 | 1-2 days |
| 100K docs | RTX 3090 | 1-2 weeks |

---

## 🔮 Using the Trained Model

### Basic Generation

```python
from inference import InferenceEngine

engine = InferenceEngine()
text = engine.generate_text("Your prompt here", max_length=100)
print(text[0])
```

### Complete Example Script

```python
from inference import InferenceEngine

# Initialize
engine = InferenceEngine()

# 1. Text Generation
result = engine.generate_text(
    prompt="Write a project summary",
    max_length=200,
    temperature=0.8
)
print(result[0])

# 2. Code Completion
code = engine.complete_code(
    code_context="def calculate_sum(numbers):\n    ",
    max_length=50
)
print(code)

# 3. Document Search
docs = ["doc1 text", "doc2 text", "doc3 text"]
similar = engine.similarity_search(
    query="search query",
    documents=docs,
    top_k=3
)
for idx, score, doc in similar:
    print(f"{score:.3f}: {doc[:50]}")

# 4. Question Answering
answer = engine.answer_question(
    question="What is the deadline?",
    context="The project deadline is June 30th.",
    max_length=30
)
print(answer)
```

---

## ⚙️ Configuration Guide

### Model Size Presets

#### Tiny (Fast, Low Memory)
```python
MODEL_CONFIG = {
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
}
# ~10M parameters, 4GB VRAM
```

#### Small (Balanced)
```python
MODEL_CONFIG = {
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
}
# ~45M parameters, 8GB VRAM
```

#### Medium (High Quality)
```python
MODEL_CONFIG = {
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
}
# ~125M parameters, 16GB VRAM
```

### Hardware-Specific Settings

#### CPU Only
```python
HARDWARE_CONFIG = {
    "device": "cpu",
    "num_workers": 4,
}
TRAINING_CONFIG["batch_size"] = 4
```

#### Single GPU (8GB)
```python
HARDWARE_CONFIG = {
    "device": "cuda",
    "gradient_checkpointing": True,
}
TRAINING_CONFIG["batch_size"] = 8
```

#### Single GPU (16GB+)
```python
HARDWARE_CONFIG = {
    "device": "cuda",
}
TRAINING_CONFIG["batch_size"] = 16
TRAINING_CONFIG["fp16"] = True
```

---

## 🐛 Troubleshooting

### Error: CUDA Out of Memory

**Solution 1**: Reduce batch size
```python
TRAINING_CONFIG["batch_size"] = 8  # or 4
```

**Solution 2**: Enable gradient checkpointing
```python
HARDWARE_CONFIG["gradient_checkpointing"] = True
```

**Solution 3**: Reduce model size
```python
MODEL_CONFIG["d_model"] = 384
MODEL_CONFIG["n_layers"] = 4
```

### Error: No data loaded

**Check**:
1. Files in `data/raw/` directory?
2. Supported file formats?
3. File permissions correct?

**Fix**:
```bash
ls -la data/raw/
# Add files if empty
cp /path/to/documents/* data/raw/
```

### Error: Training not converging

**Solutions**:
1. Lower learning rate: `learning_rate = 1e-4`
2. More training data
3. Longer warmup: `warmup_steps = 2000`
4. Check data quality

### Slow Training

**Check**:
```bash
nvidia-smi  # Check GPU utilization
```

**Solutions**:
1. Enable FP16: `fp16 = True`
2. Increase batch size
3. Use faster GPU
4. Reduce logging frequency

---

## 📈 Performance Optimization

### Speed Up Training

1. **Enable Mixed Precision**
```python
TRAINING_CONFIG["fp16"] = True
```
Results: 2x faster, same quality

2. **Increase Batch Size**
```python
TRAINING_CONFIG["batch_size"] = 32
```
Results: Faster training, more memory

3. **Reduce Logging**
```python
TRAINING_CONFIG["logging_steps"] = 500
TRAINING_CONFIG["eval_steps"] = 5000
```
Results: Less overhead

### Improve Model Quality

1. **More Training Data**
   - Target: 10M+ tokens
   - Diverse sources
   - High quality

2. **Longer Training**
```python
TRAINING_CONFIG["num_epochs"] = 20
```

3. **Better Hyperparameters**
```python
TRAINING_CONFIG["learning_rate"] = 3e-4
TRAINING_CONFIG["warmup_steps"] = 2000
```

---

## 🚢 Deployment

### Save for Production

```python
import torch
from model import create_model

model = create_model()
checkpoint = torch.load("models/checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Save deployment version
torch.save({
    'model': model.state_dict(),
    'config': MODEL_CONFIG,
}, "production_model.pt")
```

### Create API Server

```python
# api_server.py
from flask import Flask, request, jsonify
from inference import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = engine.generate_text(
        prompt=data['prompt'],
        max_length=data.get('max_length', 100)
    )
    return jsonify({'result': result[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Run server:
```bash
pip install flask
python api_server.py
```

Test:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_length":50}'
```

---

## 📚 Additional Resources

### Learning Materials
- Transformer architecture: "Attention Is All You Need" paper
- PyTorch documentation: pytorch.org
- BPE tokenization: SentencePiece, Hugging Face Tokenizers

### Enterprise Integration
- SharePoint API: Office365-REST-Python-Client docs
- Google Drive API: Google Drive API v3 docs
- Slack API: slack-sdk documentation
- GitHub API: PyGithub documentation

---

## ❓ FAQ

**Q: How much data do I need?**
A: Minimum 1,000 documents. Recommended: 10,000+

**Q: How long does training take?**
A: 2-4 hours for 1K docs on a good GPU, longer for more data

**Q: Can I use CPU only?**
A: Yes, but training will be much slower (10-20x)

**Q: Is my data private?**
A: Yes, everything runs locally. No external API calls.

**Q: Can I fine-tune on new data?**
A: Yes, load checkpoint and continue training

**Q: What GPU do I need?**
A: 8GB minimum. 16GB+ recommended.

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Add training data to `data/raw/`
3. ✅ Run `python train.py`
4. ✅ Monitor training with TensorBoard
5. ✅ Test with `python inference.py`
6. ✅ Deploy to production

Good luck with your Enterprise SLM! 🚀

For questions or issues, check the logs and TensorBoard metrics first.
