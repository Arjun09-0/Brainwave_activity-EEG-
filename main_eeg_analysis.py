# filepath: c:\Users\NANNURI ARJUN REDDY\OneDrive\Desktop\brainwave_eeg_simulation\main_eeg_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configurable parameters
EMOTION_PATH = r'C:\Users\NANNURI ARJUN REDDY\Downloads\brainwave_dataset1\emotions.csv'
EYE_PATH = r'C:\Users\NANNURI ARJUN REDDY\Downloads\eyedetection_dataset1\EEG_Eye_State_Classification.csv'
EMOTION_BANDS = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a']
EYE_BANDS = ['af3', 'f7', 'f3', 'fc5', 'fc6', 'f4', 'f8', 'af4']

def load_data(emotion_path, eye_path):
    """Load emotion and eye state datasets."""
    try:
        emotion_df = pd.read_csv(emotion_path)
        eye_df = pd.read_csv(eye_path)
        return emotion_df, eye_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(emotion_df, eye_df, emotion_bands, eye_bands):
    """Clean, normalize, and encode datasets."""
    emotion_df = emotion_df.rename(columns=str.lower)
    eye_df = eye_df.rename(columns=str.lower)
    emotion_df.columns = emotion_df.columns.str.strip().str.replace('# ', '')
    eye_df.columns = eye_df.columns.str.strip()
    emotion_df = emotion_df.dropna()
    eye_df = eye_df.dropna()
    scaler = StandardScaler()
    emotion_df[emotion_bands] = scaler.fit_transform(emotion_df[emotion_bands])
    eye_df[eye_bands] = scaler.fit_transform(eye_df[eye_bands])
    emotion_label_encoder = LabelEncoder()
    eye_label_encoder = LabelEncoder()
    emotion_df['label_encoded'] = emotion_label_encoder.fit_transform(emotion_df['label'])
    eye_df['eye_state_encoded'] = eye_label_encoder.fit_transform(eye_df['eyedetection'])
    return emotion_df, eye_df, emotion_label_encoder, eye_label_encoder

def plot_eeg_lines(df, bands, title):
    plt.figure(figsize=(12,6))
    for band in bands:
        plt.plot(df[band][:200], label=band)
    plt.legend()
    plt.title(title)
    plt.show()

def plot_corr_heatmap(df, bands, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(df[bands].corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def predict_samples(df, bands, model, label_encoder, label_col, n=10):
    for idx in range(n):
        sample_features = pd.DataFrame([df[bands].iloc[idx].values], columns=bands)
        pred = model.predict(sample_features)[0]
        label = label_encoder.inverse_transform([pred])[0]
        true_label = df[label_col].iloc[idx]
        print(f'Sample {idx+1}: True={true_label}, Predicted={label}')

def plot_feature_importance(model, bands, title):
    importances = model.feature_importances_
    plt.figure(figsize=(8,4))
    sns.barplot(x=bands, y=importances)
    plt.title(title)
    plt.ylabel('Importance')
    plt.show()

def main():
    print('--- EEG Brainwave Analysis: Emotion & Eye State Classification ---')
    emotion_df, eye_df = load_data(EMOTION_PATH, EYE_PATH)
    emotion_df, eye_df, emotion_label_encoder, eye_label_encoder = preprocess_data(
        emotion_df, eye_df, EMOTION_BANDS, EYE_BANDS
    )
    print('\nSTEP 1: Data Loaded Successfully')
    print('Emotion Dataset (first 5 rows):')
    print(emotion_df.head())
    print('\nEye State Dataset (first 5 rows):')
    print(eye_df.head())
    print('\nSTEP 2: Data Preprocessing Completed')
    print('Emotion features used:', EMOTION_BANDS)
    print('Eye state features used:', EYE_BANDS)
    print('\nSTEP 3: Visualization')
    plot_eeg_lines(emotion_df, EMOTION_BANDS, 'EEG Brainwave Activity (Emotion, Sample 200)')
    plot_eeg_lines(eye_df, EYE_BANDS, 'EEG Brainwave Activity (Eye State, Sample 200)')
    plot_corr_heatmap(emotion_df, EMOTION_BANDS, 'Correlation between Brainwave Bands (Emotion)')
    plot_corr_heatmap(eye_df, EYE_BANDS, 'Correlation between Brainwave Bands (Eye State)')
    print('\nSTEP 4: Feature Engineering')
    emotion_features = emotion_df[EMOTION_BANDS]
    emotion_labels = emotion_df['label_encoded']
    eye_features = eye_df[EYE_BANDS]
    eye_labels = eye_df['eye_state_encoded']
    print('\nSTEP 5: ML Model Training & Evaluation')
    print('Emotion Model:')
    model_emotion = train_model(emotion_features, emotion_labels)
    if model_emotion is not None:
        print("Emotion model is trained.")
    else:
        print("Emotion model is NOT trained.")

    print('Eye State Model:')
    model_eye = train_model(eye_features, eye_labels)
    if model_eye is not None:
        print("Eye state model is trained.")
    else:
        print("Eye state model is NOT trained.")
    print('\n--- Sample Predictions: Emotion ---')
    predict_samples(emotion_df, EMOTION_BANDS, model_emotion, emotion_label_encoder, 'label')
    print('\n--- Sample Predictions: Eye State ---')
    predict_samples(eye_df, EYE_BANDS, model_eye, eye_label_encoder, 'eyedetection')
    plot_feature_importance(model_emotion, EMOTION_BANDS, 'Emotion Model Feature Importance')
    plot_feature_importance(model_eye, EYE_BANDS, 'Eye State Model Feature Importance')

if __name__ == "__main__":
    main()