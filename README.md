# Speech Emotion Recognition using Deep Learning

A deep learning system that classifies emotions from speech audio using a 2D Convolutional Neural Network trained on mel-spectrogram representations.

## Live Demo

**Try it out:** [Speech Emotion Recognition on Hugging Face](https://huggingface.co/spaces/abdulmuqit/speech-emotion-recognition)

## Overview

This project addresses the challenge of automatically detecting emotional states from voice recordings. The system classifies speech into six emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad

## Datasets

We combined two publicly available emotional speech datasets:

- **RAVDESS** - 24 professional actors, ~2,880 audio files
- **CREMA-D** - 91 actors, ~7,442 audio files

Total: ~10,000 labeled samples from 115 unique speakers.

## Model Architecture

The model uses a VGG-inspired 2D CNN architecture:

| Component | Details |
|-----------|---------|
| Input | 128-band mel-spectrograms (128 × 130 × 1) |
| Conv Blocks | 4 blocks with filters: 32 → 64 → 128 → 256 |
| Regularization | BatchNorm + Dropout (0.4-0.7) |
| Dense Layers | 256 → 128 → 6 (softmax) |
| Parameters | ~4.8 million |
| Optimizer | Adam (lr=1e-4) |

## Training

- **Speaker-independent splitting:** Train/Val/Test sets contain completely different speakers
- **Data augmentation:** Time masking, frequency masking, spectral noise (SpecAugment)
- **Early stopping:** Patience of 20 epochs on validation accuracy
- **Training time:** ~30-40 minutes on L4 GPU

## Results

The model achieves approximately 55-60% accuracy on speaker-independent test data, which is strong performance for this challenging task where the model must generalize to voices it has never heard during training.

## Model Weights & Deployment

Due to file size constraints, the trained model weights and deployment files are hosted on Hugging Face:

📦 **[Hugging Face Repository](https://huggingface.co/spaces/abdulmuqit/speech-emotion-recognition/tree/main)**

Files available there:
- `best_ser_2dcnn.keras` - Trained model weights (57.9 MB)
- `label_encoder.pkl` - Label encoder for emotion classes
- `audio_params.pkl` - Audio preprocessing parameters
- `app.py` - Gradio web application
- `requirements.txt` - Python dependencies

## Usage

### Running the Demo
Visit the [Hugging Face Space](https://huggingface.co/spaces/abdulmuqit/speech-emotion-recognition) to:
- Upload an audio file
- Record directly from your microphone
- Get real-time emotion predictions with confidence scores

## Technologies Used

- Python 3.10
- TensorFlow / Keras
- Librosa (audio processing)
- Gradio (web interface)
- NumPy, Pandas, Scikit-learn

## Contributors

- **Abdul Muqit Afzal**
- **Amir Abou-el-hassan**
