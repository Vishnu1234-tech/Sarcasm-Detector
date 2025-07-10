import csv
import datetime
import tkinter as tk
from tkinter import messagebox
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from sarcasm_utils import analyze_sentiment, get_emoji, convert_to_literal

# Load model and vectorizer
model = pickle.load(open("sarcasm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Detection logic
def detect():
    user_input = entry.get()
    if not user_input:
        messagebox.showwarning("Input Missing", "Please enter a sentence.")
        return

    vec = tfidf.transform([user_input])
    result = model.predict(vec)[0]
    label = "Sarcastic 😏" if result else "Not Sarcastic ✅"

    sentiment = analyze_sentiment(user_input)
    emoji_icon = get_emoji(sentiment)

    literal = convert_to_literal(user_input)

    # Display result
    result_label.config(text=f"""
Prediction: {label}
Sentiment: {sentiment} {emoji_icon}
Literal Meaning: {literal}
""")

    # Log result
    with open("sarcasm_log.csv", "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["Timestamp", "Input Text", "Prediction", "Sentiment", "Literal Meaning"])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                         user_input, label, sentiment, literal])

# Mood chart
def show_chart():
    try:
        df = pd.read_csv("sarcasm_log.csv")
        
        # Remove emojis from prediction labels for compatibility
        df['Prediction_Clean'] = df['Prediction'].str.replace("✅", "", regex=False).str.replace("😏", "", regex=False)
        counts = df['Prediction_Clean'].value_counts()
        
        # Plot
        counts.plot(kind='bar', color=['orange', 'green'])
        plt.title("Sarcasm Detection Summary")
        plt.ylabel("Number of Sentences")
        plt.xlabel("Prediction")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        messagebox.showerror("Error", "Log file not found.")


# GUI Setup
app = tk.Tk()
app.title("SarcasmSensei - Lite")
app.geometry("420x370")

tk.Label(app, text="Enter a sentence:").pack()
entry = tk.Entry(app, width=50)
entry.pack(pady=10)

tk.Button(app, text="Detect", command=detect).pack(pady=5)

result_label = tk.Label(app, text="", wraplength=380, justify="left", font=("Arial", 10))
result_label.pack(pady=10)

tk.Button(app, text="Show Mood Chart", command=show_chart).pack(pady=5)

app.mainloop()
