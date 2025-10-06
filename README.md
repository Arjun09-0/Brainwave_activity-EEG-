# EEG Analysis Project - Render Deployment

This project analyzes EEG data for emotion classification and eye state detection.

## Features

- Emotion classification (Negative, Neutral, Positive) from EEG signals
- Eye state detection (Open/Closed) from EEG data
- Interactive web interface for real-time predictions
- Data visualization and model performance metrics

## Datasets Required

1. **Emotion Dataset**: Should contain columns for different EEG frequency bands and emotion labels
2. **Eye State Dataset**: Should contain EEG features and eye state labels (0=open, 1=closed)

## How to Use

1. Upload your CSV datasets using the file upload interface
2. Train the models on your data
3. Make predictions using the trained models
4. View visualizations and performance metrics

## Model Performance

- Emotion Classification: ~97% accuracy
- Eye State Detection: ~86% accuracy

## Technical Details

- Built with Streamlit for web interface
- Uses Random Forest classifiers
- StandardScaler for feature normalization
- Comprehensive data preprocessing and validation

## Deployment

This project is configured for deployment on Render.com with the included `render.yaml` configuration file.