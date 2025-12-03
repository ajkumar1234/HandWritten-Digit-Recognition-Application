# HandWritten-Digit-Recognition-Application


A simple, modular Handwritten Recognition project using **PyTorch**.  
Supports training on MNIST and running inference on custom images via a Streamlit demo.

## Features
- CNN model for handwritten digit/character recognition
- Training script with checkpoint saving
- Evaluation and metrics (accuracy, confusion matrix)
- Streamlit app for uploading an image and getting predictions
- Easy to extend for custom datasets

## Project structure
See repository root for full tree. Key folders:
- `src/` : model, dataset loader, training and inference scripts
- `app/` : Streamlit demo
- `models/` : saved model checkpoints

## Installation

```bash
git clone https://github.com/your-username/handwritten-recognition.git
cd handwritten-recognition

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
