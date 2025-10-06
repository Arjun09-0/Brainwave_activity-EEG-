import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io

# Configure page
st.set_page_config(
    page_title="EEG Brainwave Analysis",
    page_icon="🧠",
    layout="wide"
)

# Define EEG bands
EMOTION_BANDS = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a']
EYE_BANDS = ['af3', 'f7', 'f3', 'fc5', 'fc6', 'f4', 'f8', 'af4']

@st.cache_data
def preprocess_data(emotion_df, eye_df, emotion_bands, eye_bands):
    """Clean, normalize, and encode datasets."""
    emotion_df = emotion_df.rename(columns=str.lower)
    eye_df = eye_df.rename(columns=str.lower)
    emotion_df.columns = emotion_df.columns.str.strip().str.replace('# ', '')
    eye_df.columns = eye_df.columns.str.strip()
    emotion_df = emotion_df.dropna()
    eye_df = eye_df.dropna()
    
    scaler_emotion = StandardScaler()
    scaler_eye = StandardScaler()
    emotion_df[emotion_bands] = scaler_emotion.fit_transform(emotion_df[emotion_bands])
    eye_df[eye_bands] = scaler_eye.fit_transform(eye_df[eye_bands])
    
    emotion_label_encoder = LabelEncoder()
    eye_label_encoder = LabelEncoder()
    emotion_df['label_encoded'] = emotion_label_encoder.fit_transform(emotion_df['label'])
    eye_df['eye_state_encoded'] = eye_label_encoder.fit_transform(eye_df['eyedetection'])
    
    return emotion_df, eye_df, emotion_label_encoder, eye_label_encoder, scaler_emotion, scaler_eye

@st.cache_resource
def train_models(emotion_df, eye_df, emotion_bands, eye_bands):
    """Train machine learning models."""
    X_emotion = emotion_df[emotion_bands]
    y_emotion = emotion_df['label_encoded']
    X_eye = eye_df[eye_bands]
    y_eye = eye_df['eye_state_encoded']
    
    model_emotion = RandomForestClassifier(random_state=42)
    model_eye = RandomForestClassifier(random_state=42)
    
    model_emotion.fit(X_emotion, y_emotion)
    model_eye.fit(X_eye, y_eye)
    
    return model_emotion, model_eye

def main():
    st.title("🧠 EEG Brainwave Analysis: Emotion & Eye State Classification")
    st.markdown("Upload your EEG datasets and analyze emotional states and eye states using machine learning.")
    
    # Sidebar for file upload
    st.sidebar.header("Upload Datasets")
    emotion_file = st.sidebar.file_uploader("Upload Emotion CSV", type=['csv'])
    eye_file = st.sidebar.file_uploader("Upload Eye State CSV", type=['csv'])
    
    if emotion_file and eye_file:
        # Load data
        emotion_df = pd.read_csv(emotion_file)
        eye_df = pd.read_csv(eye_file)
        
        st.success("✅ Datasets loaded successfully!")
        
        # Display data preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emotion Dataset Preview")
            st.dataframe(emotion_df.head())
        with col2:
            st.subheader("Eye State Dataset Preview")
            st.dataframe(eye_df.head())
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            emotion_df, eye_df, emotion_label_encoder, eye_label_encoder, scaler_emotion, scaler_eye = preprocess_data(
                emotion_df, eye_df, EMOTION_BANDS, EYE_BANDS
            )
        
        st.success("✅ Data preprocessing completed!")
        
        # Train models
        with st.spinner("Training models..."):
            model_emotion, model_eye = train_models(emotion_df, eye_df, EMOTION_BANDS, EYE_BANDS)
        
        st.success("✅ Models trained successfully!")
        
        # Model performance
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emotion Classification")
            X_emotion = emotion_df[EMOTION_BANDS]
            y_emotion = emotion_df['label_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)
            y_pred = model_emotion.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")
            
        with col2:
            st.subheader("Eye State Classification")
            X_eye = eye_df[EYE_BANDS]
            y_eye = eye_df['eye_state_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X_eye, y_eye, test_size=0.2, random_state=42)
            y_pred = model_eye.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        # Visualization
        st.header("📈 Data Visualization")
        
        # EEG band correlation heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emotion Dataset Correlation")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(emotion_df[EMOTION_BANDS].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Eye State Dataset Correlation")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(eye_df[EYE_BANDS].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Prediction interface
        st.header("🔮 Make Predictions")
        st.subheader("Enter EEG Band Values:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            mean_0_a = st.number_input("mean_0_a", value=0.0)
        with col2:
            mean_1_a = st.number_input("mean_1_a", value=0.0)
        with col3:
            mean_2_a = st.number_input("mean_2_a", value=0.0)
        with col4:
            mean_3_a = st.number_input("mean_3_a", value=0.0)
        with col5:
            mean_4_a = st.number_input("mean_4_a", value=0.0)
        
        if st.button("🎯 Predict", type="primary"):
            # Prepare input for prediction
            emotion_input = np.array([[mean_0_a, mean_1_a, mean_2_a, mean_3_a, mean_4_a]])
            emotion_input_scaled = scaler_emotion.transform(emotion_input)
            
            # Make predictions
            emotion_pred = model_emotion.predict(emotion_input_scaled)
            emotion_label = emotion_label_encoder.inverse_transform(emotion_pred)[0]
            
            # For eye state, we'll use the same input (simplified for demo)
            eye_input = np.array([[mean_0_a, mean_1_a, mean_2_a, mean_3_a, mean_4_a, 0.0, 0.0, 0.0]])
            eye_input_scaled = scaler_eye.transform(eye_input)
            eye_pred = model_eye.predict(eye_input_scaled)
            eye_label = eye_label_encoder.inverse_transform(eye_pred)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🎭 Predicted Emotion: **{emotion_label}**")
            with col2:
                st.success(f"👁️ Predicted Eye State: **{eye_label}**")
    
    else:
        st.info("👆 Please upload both CSV files to get started!")
        st.markdown("""
        ### How to use:
        1. Upload your emotion and eye state CSV files using the sidebar
        2. The app will automatically preprocess the data and train models
        3. View model performance and data visualizations
        4. Make predictions by entering EEG band values
        """)

if __name__ == "__main__":
    main()