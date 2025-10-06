import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define EEG bands
EMOTION_BANDS = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a']
EYE_BANDS = ['af3', 'f7', 'f3', 'fc5', 'fc6', 'f4', 'f8', 'af4']

class EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title('EEG Brainwave Emotion & Eye State Predictor')
        self.root.geometry('500x400')
        self.emotion_df = None
        self.eye_df = None
        self.model_emotion = None
        self.model_eye = None
        self.emotion_label_encoder = None
        self.eye_label_encoder = None
        self.scaler_emotion = None
        self.scaler_eye = None
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text='EEG Brainwave Analysis', font=('Arial', 16)).pack(pady=10)
        tk.Button(self.root, text='Load Emotion CSV', command=self.load_emotion_csv).pack(pady=5)
        tk.Button(self.root, text='Load Eye State CSV', command=self.load_eye_csv).pack(pady=5)
        tk.Button(self.root, text='Train Models', command=self.train_models).pack(pady=10)
        tk.Label(self.root, text='Enter EEG Band Values:', font=('Arial', 12)).pack(pady=10)
        self.entry_vars = [tk.StringVar() for _ in EMOTION_BANDS]
        for i, band in enumerate(EMOTION_BANDS):
            tk.Label(self.root, text=band).pack()
            tk.Entry(self.root, textvariable=self.entry_vars[i]).pack()
        tk.Button(self.root, text='Predict Emotion & Eye State', command=self.predict_sample).pack(pady=15)
        self.result_label = tk.Label(self.root, text='', font=('Arial', 12))
        self.result_label.pack(pady=10)

    def load_emotion_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            self.emotion_df = pd.read_csv(file_path)
            self.emotion_df.columns = self.emotion_df.columns.str.strip().str.replace('# ', '')
            messagebox.showinfo('Loaded', 'Emotion CSV loaded successfully!')

    def load_eye_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            self.eye_df = pd.read_csv(file_path)
            # Standardize column names to lowercase and strip whitespace
            self.eye_df.columns = self.eye_df.columns.str.strip().str.lower()
            messagebox.showinfo('Loaded', 'Eye State CSV loaded successfully!')

    def train_models(self):
        if self.emotion_df is None or self.eye_df is None:
            messagebox.showerror('Error', 'Please load both CSV files first.')
            return
        self.scaler_emotion = StandardScaler()
        self.scaler_eye = StandardScaler()
        X_emotion = self.scaler_emotion.fit_transform(self.emotion_df[EMOTION_BANDS])
        X_eye = self.scaler_eye.fit_transform(self.eye_df[EYE_BANDS])
        self.emotion_label_encoder = LabelEncoder()
        self.eye_label_encoder = LabelEncoder()
        y_emotion = self.emotion_label_encoder.fit_transform(self.emotion_df['label'])
        y_eye = self.eye_label_encoder.fit_transform(self.eye_df['eyedetection'])
        self.model_emotion = RandomForestClassifier().fit(X_emotion, y_emotion)
        self.model_eye = RandomForestClassifier().fit(X_eye, y_eye)
        messagebox.showinfo('Trained', 'Models trained successfully!')

    def predict_sample(self):
        if self.model_emotion is None or self.model_eye is None:
            messagebox.showerror('Error', 'Please train the models first.')
            return
        try:
            sample = np.array([float(var.get()) for var in self.entry_vars]).reshape(1, -1)
            sample_emotion_scaled = self.scaler_emotion.transform(sample)
            sample_eye_scaled = self.scaler_eye.transform(sample)
            pred_emotion = self.model_emotion.predict(sample_emotion_scaled)
            pred_eye = self.model_eye.predict(sample_eye_scaled)
            emotion_label = self.emotion_label_encoder.inverse_transform(pred_emotion)[0]
            eye_label = self.eye_label_encoder.inverse_transform(pred_eye)[0]
            self.result_label.config(text=f'Predicted Emotion: {emotion_label}\nPredicted Eye State: {eye_label}')
        except Exception as e:
            messagebox.showerror('Error', f'Prediction failed: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    app = EEGApp(root)
    root.mainloop()
