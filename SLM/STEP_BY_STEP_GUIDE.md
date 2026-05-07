# Enterprise SLM Training Guide - Step by Step

## 📋 Complete Setup and Training Instructions

This guide will walk you through every step needed to build and train your Enterprise Small Language Model.

---

## PART 1: Environment Setup (15-30 minutes)

### Step 1.1: Install Python

**Windows:**
1. Download Python 3.10+ from python.org
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Verify: Open Command Prompt, type `python --version`

**Mac:**
```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3-pip python3-venv

# Verify
python3 --version
```

### Step 1.2: Create Project Directory

```bash
# Create and enter project folder
mkdir enterprise-slm
cd enterprise-slm
```

### Step 1.3: Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 1.4: Copy Project Files

Copy all the Python files to your `enterprise-slm` directory:
- config.py
- model.py
- data_loader.py
- tokenizer.py
- dataset.py
- train.py
- inference.py
- generate_sample_data.py
- quick_start.py
- requirements.txt
- README.md

Your directory should look like:
```
enterprise-slm/
├── venv/
├── config.py
├── model.py
├── data_loader.py
├── tokenizer.py
├── dataset.py
├── train.py
├── inference.py
├── generate_sample_data.py
├── quick_start.py
├── requirements.txt
└── README.md
```

### Step 1.5: Install Dependencies

```bash
# Make sure virtual environment is activated (you see "(venv)")
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True  (or False if CPU-only)
```

---

## PART 2: Data Preparation (30-60 minutes)

### Option A: Use Your Own Data (Recommended for Production)

#### Step 2A.1: Create Data Directory
```bash
mkdir -p data/raw
```

#### Step 2A.2: Add Your Documents

Place your enterprise documents in `data/raw/`:

**Supported formats:**
- Text: .txt, .md
- Documents: .pdf, .docx, .doc
- Spreadsheets: .xlsx, .csv
- Code: .py, .js, .java, .cpp, .html
- Data: .json, .xml

**Example:**
```bash
# Copy your documents
cp /path/to/your/documents/* data/raw/
cp /path/to/your/code/*.py data/raw/
```

**Recommended structure:**
```
data/raw/
├── project_docs/
│   ├── spec1.pdf
│   └── requirements.docx
├── code/
│   ├── script1.py
│   └── app.js
├── knowledge_base/
│   ├── guide1.md
│   └── faq.txt
└── meetings/
    ├── notes1.txt
    └── transcript1.txt
```

**Minimum requirements:**
- At least 1,000 files OR
- At least 1 million words

### Option B: Generate Sample Data (For Testing)

```bash
# Generate synthetic data for testing
python generate_sample_data.py
```

This creates:
- 100 business documents
- 50 code files (Python/JavaScript)
- 30 chat logs
- 20 meeting transcripts

---

## PART 3: Open in VS Code (10 minutes)

### Step 3.1: Open Project

```bash
# From project directory
code .
```

### Step 3.2: Select Python Interpreter

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: "Python: Select Interpreter"
3. Choose: `./venv/bin/python` or `./venv/Scripts/python.exe`

### Step 3.3: Install VS Code Extensions (Optional but Recommended)

Open Extensions (Ctrl+Shift+X) and install:
- **Python** by Microsoft
- **Pylance** by Microsoft
- **Error Lens** (shows errors inline)

### Step 3.4: Verify Setup

Open the integrated terminal in VS Code (Ctrl+`):
```bash
# Check Python
python --version

# Check virtual environment is active
which python  # Mac/Linux
where python  # Windows

# Should show path to venv
```

---

## PART 4: Configuration (10 minutes)

### Step 4.1: Review config.py

Open `config.py` and adjust based on your hardware:

**For GPU with 8GB VRAM:**
```python
MODEL_CONFIG = {
    "d_model": 384,        # Smaller model
    "n_layers": 4,
    "max_seq_length": 512,
}

TRAINING_CONFIG = {
    "batch_size": 8,
    "fp16": True,
}
```

**For GPU with 16GB+ VRAM:**
```python
MODEL_CONFIG = {
    "d_model": 512,        # Default
    "n_layers": 6,
    "max_seq_length": 512,
}

TRAINING_CONFIG = {
    "batch_size": 16,
    "fp16": True,
}
```

**For CPU Only:**
```python
HARDWARE_CONFIG = {
    "device": "cpu",
}

TRAINING_CONFIG = {
    "batch_size": 4,
    "fp16": False,
}
```

### Step 4.2: Save Changes

Save `config.py` after making changes.

---

## PART 5: Training (2-24 hours depending on data and hardware)

### Step 5.1: Quick Start Method

```bash
# Automated setup and training
python quick_start.py
```

This script will:
1. Check your environment
2. Verify dependencies
3. Check for training data
4. Offer to generate sample data if needed
5. Start training

### Step 5.2: Manual Training Method

```bash
# Start training directly
python train.py
```

### Step 5.3: Monitor Training

**Option 1: TensorBoard (Recommended)**

Open a new terminal:
```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Start TensorBoard
tensorboard --logdir=logs/tensorboard

# Open browser to: http://localhost:6006
```

You'll see:
- Training loss over time
- Learning rate schedule
- Validation metrics

**Option 2: Watch Log File**

In VS Code, open `logs/training.log` or:
```bash
tail -f logs/training.log
```

**Option 3: Console Output**

Watch the terminal where training is running.

### Step 5.4: What to Expect During Training

**Phase 1: Data Loading (5-10 minutes)**
```
Loading Enterprise Data...
Found 200 files in data/raw
Loaded 200 documents from local files
```

**Phase 2: Tokenizer Training (10-30 minutes)**
```
Training BPE tokenizer on 200 documents...
Found 5000 unique words
Learning 30000 BPE merges...
Training complete. Vocabulary size: 32000
```

**Phase 3: Dataset Creation (5-10 minutes)**
```
Tokenizing 200 documents...
Created 500 training sequences
Train dataset size: 450
Validation dataset size: 50
```

**Phase 4: Model Training (main phase)**
```
Epoch 1/10 | Step 100 | Loss: 8.5234 | LR: 0.000100
Epoch 1/10 | Step 200 | Loss: 7.2134 | LR: 0.000200
...
Validation metrics: {'loss': 6.5432, 'perplexity': 693.2, 'accuracy': 0.234}
```

**What's normal:**
- Initial loss: 8-10 (high is normal)
- Loss should decrease over time
- Perplexity should decrease
- Accuracy should increase

**Signs of good training:**
- Loss consistently decreasing
- Validation loss close to training loss
- Perplexity under 100 by end

**Warning signs:**
- Loss not decreasing after 1000 steps
- Validation loss much higher than training loss
- GPU memory errors (reduce batch_size)

### Step 5.5: Training Time Estimates

| Data Size | Hardware | Time |
|-----------|----------|------|
| 200 files (sample) | RTX 3090 | 2-3 hours |
| 200 files (sample) | RTX 2060 | 4-6 hours |
| 200 files (sample) | CPU | 10-15 hours |
| 1,000 files | RTX 3090 | 4-8 hours |
| 10,000 files | RTX 3090 | 1-2 days |

### Step 5.6: Stopping and Resuming Training

**To stop training:**
Press `Ctrl+C` in the terminal

**To resume:**
```bash
python train.py
# Training automatically resumes from last checkpoint
```

---

## PART 6: Testing the Model (10 minutes)

### Step 6.1: Run Inference Demo

```bash
python inference.py
```

You should see:
```
Enterprise SLM Inference Demo
================================================================================

1. Text Generation
--------------------------------------------------------------------------------
Prompt: The future of artificial intelligence in enterprise
Generated: The future of artificial intelligence in enterprise will focus on...

2. Code Completion
--------------------------------------------------------------------------------
...

3. Document Similarity Search
--------------------------------------------------------------------------------
...
```

### Step 6.2: Interactive Testing

Create `test_model.py`:

```python
from inference import InferenceEngine

# Load model
engine = InferenceEngine()

# Test 1: Generate text
print("Test 1: Text Generation")
text = engine.generate_text(
    prompt="Our Q4 sales results show",
    max_length=50,
    temperature=0.7
)
print(text[0])
print()

# Test 2: Code completion
print("Test 2: Code Completion")
code = engine.complete_code(
    code_context="def calculate_revenue(sales, costs):\n    profit = ",
    max_length=30
)
print(code)
print()

# Test 3: Document search
print("Test 3: Document Search")
docs = [
    "Our sales increased by 25% in Q4",
    "The new product launch was successful",
    "Customer satisfaction scores improved"
]
results = engine.similarity_search(
    query="sales performance",
    documents=docs,
    top_k=2
)
for idx, score, doc in results:
    print(f"Score {score:.3f}: {doc}")
```

Run it:
```bash
python test_model.py
```

---

## PART 7: Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solution:**
Edit `config.py`:
```python
TRAINING_CONFIG["batch_size"] = 4  # Reduce from 16
HARDWARE_CONFIG["gradient_checkpointing"] = True
```

Or reduce model size:
```python
MODEL_CONFIG["d_model"] = 256
MODEL_CONFIG["n_layers"] = 4
```

### Issue 2: No Training Data Found

**Error:**
```
No documents loaded! Please add training data
```

**Solution:**
```bash
# Check data directory
ls data/raw/

# If empty, either:
# 1. Add your documents
cp /path/to/docs/* data/raw/

# 2. Or generate sample data
python generate_sample_data.py
```

### Issue 3: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 4: Slow Training

**Check GPU usage:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**If GPU usage is low (<50%):**
1. Increase batch_size in config.py
2. Enable FP16: `TRAINING_CONFIG["fp16"] = True`
3. Reduce logging frequency

### Issue 5: Poor Model Quality

**Solutions:**
1. **More data**: Add more training documents
2. **Longer training**: Increase `num_epochs` to 20
3. **Better data**: Remove duplicates, fix encoding
4. **Larger model**: Increase `d_model` and `n_layers`

---

## PART 8: Next Steps

### Improving Your Model

1. **Add More Data**
   ```bash
   # Add new documents
   cp /path/to/new/docs/* data/raw/
   
   # Retrain
   python train.py
   ```

2. **Fine-tune on Specific Task**
   - Create task-specific dataset
   - Load pre-trained model
   - Train for fewer epochs with lower learning rate

3. **Evaluate Performance**
   - Test on held-out data
   - Measure task-specific metrics
   - Compare with baseline

### Deploying Your Model

1. **Save final model:**
   ```python
   import torch
   from model import create_model
   
   model = create_model()
   checkpoint = torch.load("models/checkpoints/best_model.pt")
   model.load_state_dict(checkpoint['model_state_dict'])
   
   torch.save(model.state_dict(), "production_model.pt")
   ```

2. **Create API** (see README.md for Flask example)

3. **Docker deployment** (see README.md for Dockerfile)

---

## PART 9: Dataset Requirements for Production

### Minimum Production Dataset

| Metric | Minimum | Recommended | Ideal |
|--------|---------|-------------|-------|
| Documents | 1,000 | 10,000 | 100,000+ |
| Total Words | 1M | 10M | 100M+ |
| Vocabulary | 5,000 | 20,000 | 50,000+ |
| Domain Coverage | 1 domain | 3-5 domains | All relevant |

### Data Quality Checklist

- [ ] Removed duplicate documents
- [ ] Fixed encoding issues (UTF-8)
- [ ] Removed PII and sensitive information
- [ ] Verified text is readable
- [ ] Organized by category/type
- [ ] Representative of use cases
- [ ] Recent and up-to-date

### Example Production Dataset

```
data/raw/
├── policies/              (100 files, ~500K words)
├── procedures/            (200 files, ~1M words)
├── technical_docs/        (500 files, ~2M words)
├── code_repositories/     (1000 files, ~3M words)
├── chat_logs/            (2000 files, ~2M words)
├── meeting_transcripts/   (500 files, ~1M words)
└── knowledge_base/        (700 files, ~1.5M words)

Total: 5,000 files, ~11M words
```

---

## PART 10: Monitoring and Maintenance

### Daily Monitoring

1. Check training progress:
   ```bash
   tail -20 logs/training.log
   ```

2. View TensorBoard metrics

3. Test model periodically:
   ```bash
   python inference.py
   ```

### Weekly Maintenance

1. Review model quality
2. Collect feedback on outputs
3. Identify improvement areas
4. Plan data additions

### Monthly Tasks

1. Retrain with new data
2. Evaluate on test set
3. Update documentation
4. Backup checkpoints

---

## Quick Reference Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py

# Training
python train.py

# Monitor
tensorboard --logdir=logs/tensorboard
tail -f logs/training.log

# Inference
python inference.py

# Test
python test_model.py
```

---

## Getting Help

1. **Check logs**: `logs/training.log`
2. **Review TensorBoard**: metrics and graphs
3. **Verify config**: `config.py` settings
4. **Test GPU**: `nvidia-smi`
5. **Check data**: `ls -lR data/raw/`

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Training data added (1,000+ files recommended)
- [ ] Config.py reviewed and adjusted
- [ ] Training started successfully
- [ ] TensorBoard showing metrics
- [ ] Loss decreasing over time
- [ ] Model saved to checkpoints/
- [ ] Inference working

---

Congratulations! You've built and trained your Enterprise SLM! 🎉

For more details, see README.md or individual Python files.
