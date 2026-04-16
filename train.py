import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Step 1: Load the data
def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if ';' in line:
                text, label = line.rsplit(';', 1)
                data.append({'text': text.strip(), 'emotion': label.strip()})
    return pd.DataFrame(data)

# Step 2: Preprocess and train the model
def train_model(df):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return model, vectorizer

# Step 3: Predict emotion
def predict_emotion(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction

# Step 4: GUI
def create_ui(model, vectorizer):
    root = tk.Tk()
    root.title("Emotion Classifier")
    
    tk.Label(root, text="Enter text to classify emotion:").pack(pady=10)
    text_entry = tk.Entry(root, width=50)
    text_entry.pack(pady=5)
    
    def on_predict():
        text = text_entry.get()
        if text:
            emotion = predict_emotion(text, model, vectorizer)
            messagebox.showinfo("Prediction", f"Predicted Emotion: {emotion}")
        else:
            messagebox.showwarning("Warning", "Please enter some text.")
    
    predict_button = tk.Button(root, text="Predict Emotion", command=on_predict)
    predict_button.pack(pady=10)
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    filepath = r"c:\Users\City College\Desktop\116\genai\train.txt"
    df = load_data(filepath)
    model, vectorizer = train_model(df)
    create_ui(model, vectorizer)