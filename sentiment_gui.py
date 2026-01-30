import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)

# ---------- Core Logic ----------
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d+', '', text)
    return text.lower()

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Review Sentiment Analyzer')
        self.root.geometry('600x500')
        self.df = None
        self.sia = SentimentIntensityAnalyzer()

        # UI Elements
        top = tk.Frame(root)
        top.pack(pady=10)

        tk.Button(top, text='Load CSV', command=self.load_csv, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text='Analyze', command=self.analyze, width=15).pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(root, text='Load a CSV with column: reviews.text', fg='blue')
        self.status.pack(pady=5)

        # Single comment input
        input_frame = tk.LabelFrame(root, text='Analyze Single Comment')
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.comment_entry = tk.Entry(input_frame)
        self.comment_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        tk.Button(input_frame, text='Analyze Comment', command=self.analyze_comment).pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text='', font=('Arial', 12, 'bold'))
        self.result_label.pack(pady=5)

        # Main content frame
        content_frame = tk.Frame(root)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left side (inputs)
        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)

        # Right side (chart)
        self.chart_frame = tk.Frame(content_frame)
        self.chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Exit button (always visible)
        exit_frame = tk.Frame(root)
        exit_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        tk.Button(exit_frame, text='Exit', command=self.root.destroy, bg='red', fg='white', width=12).pack()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            if 'reviews.text' not in self.df.columns:
                raise ValueError('Column reviews.text not found')
            self.status.config(text=f'Loaded: {path}', fg='green')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def analyze(self):
        if self.df is None:
            messagebox.showwarning('Warning', 'Please load a CSV first')
            return

        df = self.df.copy()
        df['cleaned'] = df['reviews.text'].apply(clean_text)
        df['sentiment_score'] = df['cleaned'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
        df['sentiment'] = df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
        )

        # Clear old chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        df['sentiment'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status.config(text='Analysis complete', fg='green')

    def analyze_comment(self):
        comment = self.comment_entry.get().strip()
        if not comment:
            messagebox.showwarning('Warning', 'Please enter a comment')
            return

        cleaned = clean_text(comment)
        score = self.sia.polarity_scores(cleaned)['compound']

        if score > 0:
            sentiment = 'POSITIVE'
            color = 'green'
        elif score < 0:
            sentiment = 'NEGATIVE'
            color = 'red'
        else:
            sentiment = 'NEUTRAL'
            color = 'gray'

        self.result_label.config(text=f'Sentiment: {sentiment}', fg=color)

        # Change window background color based on sentiment
        if sentiment == 'POSITIVE':
            self.root.configure(bg='#b6e7b6')  # light green
        elif sentiment == 'NEGATIVE':
            self.root.configure(bg='#f2b6b6')  # light red
        else:
            self.root.configure(bg='#f2efb6')  # light yellow

# ---------- Run App ----------
if __name__ == '__main__':
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
