#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install essential libraries for learning from scratch
echo "Installing NumPy and Matplotlib..."
pip install numpy matplotlib

echo "Setup complete! Run 'source venv/bin/activate' to start."
