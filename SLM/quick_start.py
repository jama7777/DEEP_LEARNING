"""
Quick Start Script
==================
Automated setup and first training run for Enterprise SLM.
This script will guide you through the initial setup process.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        print("Please upgrade Python and try again.")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required = ['torch', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_gpu():
    """Check GPU availability."""
    print_header("Checking GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU (slower)")
            print("   Consider using a machine with GPU for faster training")
            return False
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False

def check_data():
    """Check if training data exists."""
    print_header("Checking Training Data")
    
    from config import RAW_DATA_DIR
    
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created data directory: {RAW_DATA_DIR}")
    
    # Count files
    supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.py', '.js']
    files = []
    for ext in supported_extensions:
        files.extend(list(RAW_DATA_DIR.rglob(f"*{ext}")))
    
    if len(files) == 0:
        print(f"⚠️  No training data found in {RAW_DATA_DIR}")
        print("\nOptions:")
        print("1. Add your own documents to data/raw/")
        print("2. Generate sample data: python generate_sample_data.py")
        return False
    else:
        print(f"✅ Found {len(files)} training files")
        return True

def setup_directories():
    """Create necessary directories."""
    print_header("Setting Up Directories")
    
    from config import DATA_DIR, MODEL_DIR, LOGS_DIR, CHECKPOINT_DIR
    
    dirs = [DATA_DIR, MODEL_DIR, LOGS_DIR, CHECKPOINT_DIR]
    
    for directory in dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {directory}")
        else:
            print(f"✅ Exists: {directory}")

def run_sample_data_generation():
    """Ask user if they want to generate sample data."""
    print_header("Sample Data Generation")
    
    response = input("Would you like to generate sample data for testing? (y/n): ").lower()
    
    if response == 'y':
        print("\nGenerating sample data...")
        try:
            from generate_sample_data import generate_all
            generate_all()
            return True
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            return False
    else:
        print("Skipping sample data generation.")
        print("Please add your training data to data/raw/")
        return False

def print_training_tips():
    """Print helpful tips for training."""
    print_header("Training Tips")
    
    tips = [
        "Monitor training with: tensorboard --logdir=logs/tensorboard",
        "Training logs are saved to: logs/training.log",
        "Checkpoints are saved to: models/checkpoints/",
        "Press Ctrl+C to stop training (progress is saved)",
        "Reduce batch_size in config.py if you get OOM errors",
        "Training time depends on data size and hardware",
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

def main():
    """Main quick start function."""
    print_header("Enterprise SLM - Quick Start")
    print("This script will help you set up and train your first model.\n")
    
    # Step 1: Check Python
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Check GPU
    has_gpu = check_gpu()
    
    # Step 4: Setup directories
    setup_directories()
    
    # Step 5: Check data
    has_data = check_data()
    
    # Step 6: Offer to generate sample data
    if not has_data:
        generated = run_sample_data_generation()
        if not generated:
            print("\n⚠️  No training data available.")
            print("Add files to data/raw/ and run this script again, or:")
            print("  python generate_sample_data.py")
            sys.exit(1)
    
    # Step 7: Print training tips
    print_training_tips()
    
    # Step 8: Ask if user wants to start training
    print_header("Ready to Train")
    print("All checks passed! You're ready to train your model.")
    print("\nEstimated time for sample data (~200 files):")
    if has_gpu:
        print("  - With GPU: 2-4 hours")
    else:
        print("  - With CPU: 8-12 hours")
    
    response = input("\nStart training now? (y/n): ").lower()
    
    if response == 'y':
        print("\nStarting training...")
        print("(You can stop anytime with Ctrl+C - progress will be saved)\n")
        
        try:
            # Import and run training
            from train import main as train_main
            train_main()
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("Progress has been saved. Resume by running: python train.py")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print("Check logs/training.log for details")
    else:
        print("\nTraining skipped. To train later, run:")
        print("  python train.py")
    
    print_header("Quick Start Complete")
    print("Next steps:")
    print("1. Monitor training: tensorboard --logdir=logs/tensorboard")
    print("2. Check logs: tail -f logs/training.log")
    print("3. Test model: python inference.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user. Run again when ready.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error and try again.")
