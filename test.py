import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

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

class EmotionApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=16)
        self.master = master
        self.master.title("Emotion Classifier App")
        self.master.geometry("680x520")
        self.master.resizable(False, False)

        self.dataset_path = None
        self.df = None
        self.model = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.report = None

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f5f5ff")
        self.style.configure("TLabel", background="#f5f5ff", font=("Segoe UI", 11))
        self.style.configure("TButton", font=("Segoe UI", 11), padding=8)
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background="#f5f5ff")

        self.create_widgets()

    def create_widgets(self):
        header = ttk.Label(self, text="Emotion Classifier", style="Header.TLabel")
        header.pack(pady=(0, 12))

        file_frame = ttk.LabelFrame(self, text="Dataset", padding=12)
        file_frame.pack(fill="x", pady=(0, 12))

        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=(0, 8))

        select_button = ttk.Button(
            file_frame, text="Choose File", command=self.choose_file
        )
        select_button.pack(side="right")

        stats_frame = ttk.LabelFrame(self, text="Training & Test", padding=12)
        stats_frame.pack(fill="x", pady=(0, 12))

        self.stats_label = ttk.Label(
            stats_frame,
            text="Load a dataset first to see statistics and train the model.",
            wraplength=620,
        )
        self.stats_label.pack(fill="x")

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", pady=(0, 12))

        train_button = ttk.Button(action_frame, text="Train Model", command=self.train)
        train_button.pack(side="left", expand=True, fill="x", padx=4)

        results_button = ttk.Button(
            action_frame, text="Show Test Results", command=self.show_results
        )
        results_button.pack(side="left", expand=True, fill="x", padx=4)

        save_button = ttk.Button(
            action_frame, text="Save Model", command=self.save_model
        )
        save_button.pack(side="left", expand=True, fill="x", padx=4)

        predict_frame = ttk.LabelFrame(self, text="Predict Emotion", padding=12)
        predict_frame.pack(fill="both", expand=True, pady=(0, 12))

        self.input_text = tk.Text(predict_frame, height=5, font=("Segoe UI", 11), wrap="word")
        self.input_text.pack(fill="x", pady=(0, 12))

        self.predict_result = ttk.Label(predict_frame, text="Enter text and click Predict.")
        self.predict_result.pack(anchor="w", pady=(0, 8))

        predict_button = ttk.Button(
            predict_frame, text="Predict Emotion", command=self.predict_sample
        )
        predict_button.pack(anchor="center", pady=(0, 6))

        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x")

        self.status_label = ttk.Label(status_frame, text="Ready", foreground="#333333")
        self.status_label.pack(anchor="w")

        self.pack(fill="both", expand=True)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select emotion training file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.dataset_path = path
        self.file_label.config(text=os.path.basename(path))
        try:
            self.df = load_data(path)
            self.stats_label.config(
                text=(
                    f"Loaded {len(self.df)} records.\n"
                    f"Emotions: {', '.join(sorted(self.df['emotion'].unique()))}"
                )
            )
            self.status_label.config(text="Dataset loaded successfully.")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load dataset:\n{exc}")
            self.status_label.config(text="Failed to load dataset.")

    def train(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please choose a dataset file first.")
            return
        try:
            (
                self.model,
                self.vectorizer,
                self.X_test,
                self.y_test,
                self.y_pred,
                self.accuracy,
                self.report,
            ) = train_model(self.df)
            self.stats_label.config(
                text=(
                    f"Loaded {len(self.df)} records.\n"
                    f"Training complete. Test accuracy: {self.accuracy:.2f}"
                )
            )
            self.status_label.config(text="Model trained successfully.")
            messagebox.showinfo("Training Complete", f"Accuracy: {self.accuracy:.2f}")
        except Exception as exc:
            messagebox.showerror("Error", f"Training failed:\n{exc}")
            self.status_label.config(text="Training failed.")

    def predict_sample(self):
        if self.model is None or self.vectorizer is None:
            messagebox.showwarning("No Model", "Train the model before predicting.")
            return
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter text to classify.")
            return
        emotion = predict_emotion(text, self.model, self.vectorizer)
        self.predict_result.config(text=f"Predicted Emotion: {emotion}", foreground="#005500")
        self.status_label.config(text="Prediction completed.")

    def show_results(self):
        if self.model is None or self.X_test is None:
            messagebox.showwarning("No Results", "Train the model first to view test results.")
            return
        window = tk.Toplevel(self.master)
        window.title("Test Results")
        window.geometry("720x540")
        window.configure(bg="#f5f5ff")

        title = ttk.Label(window, text="Test Results", style="Header.TLabel")
        title.pack(pady=(10, 4))

        summary = ttk.Label(
            window,
            text=f"Accuracy: {self.accuracy:.2f}\nTest set size: {len(self.X_test)}",
            wraplength=680,
        )
        summary.pack(pady=(0, 10))

        report_box = ScrolledText(window, wrap="word", font=("Segoe UI", 10), height=14)
        report_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        report_box.insert("1.0", self.report)
        report_box.configure(state="disabled")

        sample_box = ScrolledText(window, wrap="word", font=("Segoe UI", 10), height=10)
        sample_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        sample_box.insert("1.0", "Sample test predictions:\n\n")
        for i in range(min(12, len(self.X_test))):
            sample_box.insert(
                "end",
                f"Text: {self.X_test.iloc[i]}\n"
                f"Actual: {self.y_test.iloc[i]}\n"
                f"Predicted: {self.y_pred[i]}\n\n",
            )
        sample_box.configure(state="disabled")

        self.status_label.config(text="Displayed test results.")

    def save_model(self):
        if self.model is None or self.vectorizer is None:
            messagebox.showwarning("No Model", "Train the model before saving.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "vectorizer": self.vectorizer},
                f,
            )
        messagebox.showinfo("Saved", f"Model saved to:\n{path}")
        self.status_label.config(text="Model saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()