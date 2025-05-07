import csv
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset from local CSV files
fake_file = "Fake.csv"
real_file = "True.csv"

if not (os.path.exists(fake_file) and os.path.exists(real_file)):
    messagebox.showerror("Error", "Dataset files not found! Please place Fake.csv and True.csv in the folder.")
    exit()

# Read data
texts, labels = [], []
for file, label in [(fake_file, 1), (real_file, 0)]:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            texts.append(row[0])  # Assuming first column contains news text
            labels.append(label)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------- Tkinter UI --------------------
def check_news():
    headline = entry.get("1.0", tk.END).strip()
    if not headline:
        messagebox.showwarning("Warning", "Please enter a news headline.")
        return

    # Predict
    vectorized_input = vectorizer.transform([headline])
    prediction = model.predict(vectorized_input)[0]
    result_text = "❌ FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"

    result_label.config(text=result_text, fg="#FF5555" if prediction == 1 else "#50FA7B")

def search_news():
    headline = entry.get("1.0", tk.END).strip()
    if headline:
        search_url = f"https://www.google.com/search?q={headline}"
        webbrowser.open(search_url)

# iOS-style button effect
def on_enter(e):
    e.widget.config(bg="#5A5E6B")  # Lighter hover effect

def on_leave(e):
    e.widget.config(bg="#3A3F4B")  # Normal color

# Create UI
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("480x420")
root.configure(bg="#1E1E1E")  # Dark mode background
root.resizable(False, False)  # Prevent resizing

# Heading
title_label = tk.Label(root, text="📰 Fake News Detector", font=("Arial", 17, "bold"), fg="#8BE9FD", bg="#1E1E1E")
title_label.pack(pady=15)

# Input box
entry = scrolledtext.ScrolledText(root, height=4, width=50, font=("Arial", 12), bg="#282A36", fg="white", insertbackground="white", relief="flat", wrap=tk.WORD, padx=10, pady=10)
entry.pack(pady=10)

# Buttons Frame
btn_frame = tk.Frame(root, bg="#1E1E1E")
btn_frame.pack(pady=15)

button_style = {
    "font": ("Arial", 12, "bold"),
    "bg": "#3A3F4B",
    "fg": "white",
    "bd": 0,
    "padx": 15,
    "pady": 8,
    "relief": "flat",
    "cursor": "hand2",
    "activebackground": "#5A5E6B"
}

check_button = tk.Button(btn_frame, text="Check News", command=check_news, **button_style)
check_button.pack(side=tk.LEFT, padx=10)
check_button.bind("<Enter>", on_enter)
check_button.bind("<Leave>", on_leave)

search_button = tk.Button(btn_frame, text="Search News", command=search_news, **button_style)
search_button.pack(side=tk.LEFT, padx=10)
search_button.bind("<Enter>", on_enter)
search_button.bind("<Leave>", on_leave)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#1E1E1E", fg="white")
result_label.pack(pady=10)

# Run app
root.mainloop()
