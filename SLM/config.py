"""
Enterprise Small Language Model (SLM) Configuration
====================================================
Central configuration file for all model hyperparameters, paths, and training settings.
Modify these parameters to customize the model for your enterprise needs.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODEL_DIR, CHECKPOINT_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SOURCE CONNECTORS
# ============================================================================
# Configure your enterprise data sources here
DATA_SOURCES = {
    "sharepoint": {
        "enabled": False,
        "base_url": "https://yourcompany.sharepoint.com",
        "site_name": "YourSite",
        "document_library": "Documents",
    },
    "google_drive": {
        "enabled": False,
        "folder_ids": [],  # List of folder IDs to crawl
        "credentials_path": "credentials/google_drive_creds.json",
    },
    "slack": {
        "enabled": False,
        "token": "",  # Bot token
        "channels": [],  # Channel IDs to process
    },
    "github": {
        "enabled": False,
        "repos": [],  # List of repo URLs
        "token": "",  # Personal access token
    },
    "local_files": {
        "enabled": True,
        "paths": [str(RAW_DATA_DIR)],
    }
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Optimized for single GPU training (8-16GB VRAM)
MODEL_CONFIG = {
    # Model size parameters
    "vocab_size": 32000,  # Vocabulary size
    "d_model": 512,  # Embedding dimension (reduced for efficiency)
    "n_layers": 6,  # Number of transformer layers (small for SLM)
    "n_heads": 8,  # Number of attention heads
    "d_ff": 2048,  # Feed-forward dimension (4x d_model)
    "max_seq_length": 512,  # Maximum sequence length
    "dropout": 0.1,  # Dropout rate
    
    # Advanced features
    "use_flash_attention": False,  # Enable if available for faster training
    "use_rotary_embeddings": True,  # RoPE for better positional encoding
    "use_glu_variants": True,  # Gated Linear Units for better capacity
    "tie_word_embeddings": True,  # Tie input/output embeddings to save memory
    
    # Model type
    "architecture": "decoder_only",  # decoder_only, encoder_decoder
}

# ============================================================================
# TOKENIZER CONFIGURATION
# ============================================================================
TOKENIZER_CONFIG = {
    "type": "bpe",  # bpe, wordpiece, unigram
    "vocab_size": 32000,
    "min_frequency": 2,
    "special_tokens": [
        "<PAD>",  # Padding token
        "<UNK>",  # Unknown token
        "<BOS>",  # Beginning of sequence
        "<EOS>",  # End of sequence
        "<SEP>",  # Separator token
        "<MASK>",  # Mask token for MLM
        "<DOC>",  # Document start
        "<CODE>",  # Code snippet marker
        "<CHAT>",  # Chat message marker
    ],
    
    # Multilingual support
    "languages": ["en", "es", "fr", "de", "zh", "ja", "pt", "hi"],
    "normalize_unicode": True,
    "lowercase": False,  # Keep casing for enterprise data
    
    # Domain-specific handling
    "preserve_patterns": [
        r'\b[A-Z]{2,}\b',  # Acronyms (e.g., API, SQL, AWS)
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        r'\w+@\w+\.\w+',  # Email addresses
        r'https?://[^\s]+',  # URLs
        r'\d+\.\d+\.\d+',  # Version numbers
    ],
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    # Basic training parameters
    "batch_size": 16,  # Adjust based on GPU memory
    "gradient_accumulation_steps": 4,  # Effective batch size = 64
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,  # Gradient clipping
    
    # Training schedule
    "num_epochs": 10,
    "warmup_steps": 1000,
    "save_steps": 5000,
    "eval_steps": 1000,
    "logging_steps": 100,
    
    # Learning rate schedule
    "lr_scheduler": "cosine",  # linear, cosine, constant
    "lr_scheduler_kwargs": {
        "num_cycles": 0.5,
        "min_lr_ratio": 0.1,
    },
    
    # Mixed precision training (for faster training)
    "fp16": True,  # Use FP16 if GPU supports it
    "fp16_opt_level": "O1",
    
    # Checkpoint management
    "save_total_limit": 3,  # Keep only 3 best checkpoints
    "load_best_model_at_end": True,
    
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
}

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
PREPROCESSING_CONFIG = {
    # Text cleaning
    "remove_html": True,
    "remove_urls": False,  # Keep URLs for enterprise context
    "remove_emails": False,  # Keep emails for enterprise context
    "fix_unicode": True,
    "normalize_whitespace": True,
    
    # Document handling
    "max_document_length": 10000,  # Characters per document
    "document_overlap": 128,  # Overlapping tokens when splitting
    "min_document_length": 50,  # Minimum characters to keep
    
    # Supported file types
    "supported_extensions": [
        ".txt", ".md", ".pdf", ".docx", ".doc", 
        ".xlsx", ".csv", ".json", ".xml",
        ".py", ".js", ".java", ".cpp", ".html", ".css",
    ],
    
    # Data augmentation
    "augmentation": {
        "enabled": False,
        "techniques": ["synonym_replacement", "random_deletion"],
        "augmentation_ratio": 0.1,
    },
}

# ============================================================================
# TRAINING OBJECTIVES
# ============================================================================
OBJECTIVES_CONFIG = {
    # Primary objective
    "primary_objective": "clm",  # clm (Causal Language Modeling)
    
    # Multi-task learning (optional)
    "multi_task": {
        "enabled": True,
        "tasks": {
            "clm": 0.7,  # Causal Language Modeling weight
            "mlm": 0.2,  # Masked Language Modeling weight
            "nsp": 0.1,  # Next Sentence Prediction weight
        },
    },
    
    # Domain adaptation
    "domain_weights": {
        "documents": 0.4,
        "code": 0.2,
        "chat": 0.2,
        "transcripts": 0.2,
    },
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_CONFIG = {
    "metrics": [
        "perplexity",
        "loss",
        "accuracy",
    ],
    "eval_batch_size": 32,
    "eval_accumulation_steps": 1,
}

# ============================================================================
# HARDWARE OPTIMIZATION
# ============================================================================
HARDWARE_CONFIG = {
    "device": "cuda",  # cuda, cpu, mps (for Apple Silicon)
    "num_workers": 4,  # DataLoader workers
    "pin_memory": True,
    "prefetch_factor": 2,
    
    # Memory optimization
    "gradient_checkpointing": True,  # Trade compute for memory
    "use_8bit_adam": False,  # Requires bitsandbytes library
    
    # Distributed training (if available)
    "distributed": False,
    "world_size": 1,
}

# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
DEPLOYMENT_CONFIG = {
    "model_name": "enterprise_slm_v1",
    "model_version": "1.0.0",
    "deployment_mode": "on_premise",  # on_premise, edge, hybrid
    
    # Inference optimization
    "quantization": {
        "enabled": False,
        "bits": 8,  # 8-bit or 4-bit quantization
    },
    
    # Security
    "encryption": {
        "enabled": True,
        "algorithm": "AES-256",
    },
    
    # Privacy
    "pii_detection": True,
    "data_retention_days": 90,
}

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================
LOGGING_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_to_file": True,
    "log_file": LOGS_DIR / "training.log",
    
    # Experiment tracking
    "use_tensorboard": True,
    "tensorboard_dir": LOGS_DIR / "tensorboard",
    
    # Weights & Biases (optional, disable for full privacy)
    "use_wandb": False,
    "wandb_project": "enterprise-slm",
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_device():
    """Get the appropriate device for training."""
    import torch
    if HARDWARE_CONFIG["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif HARDWARE_CONFIG["device"] == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def print_config():
    """Print the current configuration."""
    print("=" * 80)
    print("Enterprise SLM Configuration")
    print("=" * 80)
    print(f"Model Architecture: {MODEL_CONFIG['architecture']}")
    print(f"Model Size: {MODEL_CONFIG['d_model']}d x {MODEL_CONFIG['n_layers']} layers")
    print(f"Vocabulary Size: {MODEL_CONFIG['vocab_size']}")
    print(f"Max Sequence Length: {MODEL_CONFIG['max_seq_length']}")
    print(f"Training Device: {get_device()}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
