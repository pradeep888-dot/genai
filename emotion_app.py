import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox, Toplevel
from tkinter.scrolledtext import ScrolledText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "train.txt")

def load_data(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if ";" in line:
                text, label = line.rsplit(";", 1)
                data.append({"text": text.strip(), "emotion": label.strip()})
    return pd.DataFrame(data)

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, vectorizer, X_test, y_test, y_pred, accuracy, report

def predict_emotion(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

class EmotionApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1b263b")
        self.master = master
        self.master.title("Emotion Classifier")
        self.master.geometry("760x560")
        self.master.configure(bg="#1b263b")
        self.master.resizable(False, False)

        self.df = None
        self.model = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.report = None

        self.create_widgets()
        self.load_and_train()

    def create_widgets(self):
        title = tk.Label(
            self,
            text="Emotion Classifier",
            bg="#1b263b",
            fg="#f7f8fb",
            font=("Segoe UI", 24, "bold"),
        )
        title.pack(pady=(16, 8))

        subtitle = tk.Label(
            self,
            text="Automatically loads train.txt from this folder and predicts emotion.",
            bg="#1b263b",
            fg="#b8c6e5",
            font=("Segoe UI", 10),
        )
        subtitle.pack(pady=(0, 14))

        self.info_label = tk.Label(
            self,
            text="Loading training data...",
            bg="#1b263b",
            fg="#e2e8f0",
            font=("Segoe UI", 11),
            wraplength=720,
            justify="left",
        )
        self.info_label.pack(padx=20, pady=(0, 16), anchor="w")

        frame = tk.Frame(self, bg="#293553", bd=0, relief="flat")
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        box_label = tk.Label(
            frame,
            text="Enter sentence to classify:",
            bg="#293553",
            fg="#f7f8fb",
            font=("Segoe UI", 11, "bold"),
        )
        box_label.pack(anchor="w", padx=16, pady=(16, 8))

        self.input_text = tk.Text(
            frame,
            height=7,
            bg="#1b2a44",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            font=("Segoe UI", 12),
            wrap="word",
            bd=0,
            relief="flat",
            padx=12,
            pady=12,
        )
        self.input_text.pack(fill="both", expand=False, padx=16, pady=(0, 16))

        self.predict_label = tk.Label(
            frame,
            text="Type a sentence and click Predict.",
            bg="#293553",
            fg="#cbd5e1",
            font=("Segoe UI", 11),
            wraplength=680,
            justify="left",
        )
        self.predict_label.pack(anchor="w", padx=16, pady=(0, 16))

        button_frame = tk.Frame(frame, bg="#293553")
        button_frame.pack(fill="x", padx=16, pady=(0, 16))

        predict_button = tk.Button(
            button_frame,
            text="Predict Emotion",
            command=self.predict_sample,
            bg="#5b7ad1",
            fg="#ffffff",
            activebackground="#7488d2",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=10,
        )
        predict_button.pack(side="left", expand=True, fill="x", padx=(0, 8))

        results_button = tk.Button(
            button_frame,
            text="Show Test Results",
            command=self.show_results,
            bg="#2f4a8f",
            fg="#ffffff",
            activebackground="#4965a6",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=10,
        )
        results_button.pack(side="left", expand=True, fill="x", padx=(8, 0))

        self.status_label = tk.Label(
            self,
            text="Ready",
            bg="#1b263b",
            fg="#94a3b8",
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
        )
        self.status_label.pack(fill="x", padx=20)

        self.pack(fill="both", expand=True)

    def load_and_train(self):
        if not os.path.exists(DEFAULT_DATA_FILE):
            self.info_label.config(
                text=f"Could not find train.txt in this folder:\n{DEFAULT_DATA_FILE}"
            )
            self.status_label.config(text="train.txt not found.")
            return

        try:
            self.df = load_data(DEFAULT_DATA_FILE)
            (
                self.model,
                self.vectorizer,
                self.X_test,
                self.y_test,
                self.y_pred,
                self.accuracy,
                self.report,
            ) = train_model(self.df)
            self.info_label.config(
                text=(
                    f"Loaded {len(self.df)} records from train.txt.\n"
                    f"Model trained automatically. Accuracy: {self.accuracy:.2f}"
                )
            )
            self.status_label.config(text="Model loaded and trained.")
        except Exception as exc:
            self.info_label.config(text=f"Error loading or training: {exc}")
            self.status_label.config(text="Error during startup.")

    def predict_sample(self):
        if self.model is None:
            messagebox.showwarning(
                "Model not ready", "The model is not ready yet. Wait for training."
            )
            return
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Enter text", "Please type a sentence to classify.")
            return
        emotion = predict_emotion(text, self.model, self.vectorizer)
        self.predict_label.config(
            text=f"Predicted Emotion: {emotion}",
            fg="#a7f3d0",
        )
        self.status_label.config(text="Prediction complete.")

    def show_results(self):
        if self.model is None or self.X_test is None:
            messagebox.showwarning("No results", "Train the model first to view test results.")
            return
        window = tk.Toplevel(self.master)
        window.title("Test Results")
        window.geometry("720x520")
        window.configure(bg="#1b263b")

        title = tk.Label(
            window,
            text="Test Results",
            bg="#1b263b",
            fg="#f7f8fb",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(pady=(16, 8))

        summary = tk.Label(
            window,
            text=f"Accuracy: {self.accuracy:.2f}    Test set size: {len(self.X_test)}",
            bg="#1b263b",
            fg="#cbd5e1",
            font=("Segoe UI", 11),
        )
        summary.pack(pady=(0, 12))

        report_box = ScrolledText(
            window,
            wrap="word",
            font=("Segoe UI", 10),
            bg="#18213d",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            relief="flat",
            height=12,
        )
        report_box.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        report_box.insert("1.0", self.report)
        report_box.configure(state="disabled")

        sample_box = ScrolledText(
            window,
            wrap="word",
            font=("Segoe UI", 10),
            bg="#18213d",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            relief="flat",
            height=10,
        )
        sample_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        sample_box.insert("1.0", "Sample predictions:\n\n")
        for i in range(min(10, len(self.X_test))):
            sample_box.insert(
                "end",
                f"Text: {self.X_test.iloc[i]}\nActual: {self.y_test.iloc[i]}\nPredicted: {self.y_pred[i]}\n\n",
            )
        sample_box.configure(state="disabled")

        self.status_label.config(text="Displayed test results.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()