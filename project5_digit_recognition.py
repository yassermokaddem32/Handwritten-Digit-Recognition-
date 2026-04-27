"""
╔══════════════════════════════════════════════════════════════╗
║       PROJECT 5: Handwritten Digit Recognition (MNIST)       ║
║──────────────────────────────────────────────────────────────║
║  INSTALLATION:                                               ║
║    pip install tensorflow pillow numpy                       ║
║                                                              ║
║  RUN:                                                        ║
║    python project5_digit_recognition.py                     ║
║                                                              ║
║  NOTE: Trains a lightweight model on first launch (~30s).   ║
║  Saves to mnist_model.h5 so subsequent runs are instant.    ║
╚══════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import threading
import os

MODEL_PATH = "mnist_model.h5"
CANVAS_SIZE = 280          # 280×280 px canvas (10× MNIST 28×28)


def get_model():
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    # Quick train
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    model.save(MODEL_PATH)
    return model


class DigitApp:
    BG     = "#111"
    CANVAS = "#1a1a1a"
    ACCENT = "#a78bfa"
    TEXT   = "#f0f0f0"
    MUTED  = "#555"

    def __init__(self, root):
        self.root = root
        self.root.title("✏️ Digit Recognizer")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        self.model = None

        # drawing state
        self.drawing = False
        self.draw_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw_ctx   = ImageDraw.Draw(self.draw_image)

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg="#1a1a2e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="DIGIT RECOGNIZER", font=("Courier New", 14, "bold"),
                 fg=self.ACCENT, bg="#1a1a2e", pady=10).pack(side="left", padx=20)

        # Drawing canvas
        canvas_frame = tk.Frame(self.root, bg=self.ACCENT, padx=2, pady=2)
        canvas_frame.pack(padx=24, pady=16)

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=self.CANVAS, cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._start_draw)
        self.canvas.bind("<B1-Motion>",       self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # Hint text
        self.hint = self.canvas.create_text(
            CANVAS_SIZE//2, CANVAS_SIZE//2,
            text="Draw a digit here", fill=self.MUTED,
            font=("Courier New", 13)
        )

        # Result
        res_frame = tk.Frame(self.root, bg="#1a1a2e")
        res_frame.pack(fill="x", padx=24, pady=(0, 8))

        tk.Label(res_frame, text="PREDICTION", font=("Courier New", 9),
                 fg=self.MUTED, bg="#1a1a2e").pack(side="left", padx=0)

        self.digit_var = tk.StringVar(value="?")
        tk.Label(res_frame, textvariable=self.digit_var,
                 font=("Courier New", 52, "bold"),
                 fg=self.ACCENT, bg="#1a1a2e", width=3).pack(side="left")

        self.conf_var = tk.StringVar(value="")
        tk.Label(res_frame, textvariable=self.conf_var,
                 font=("Courier New", 13), fg=self.TEXT, bg="#1a1a2e").pack(side="left")

        # Confidence bars
        self.bars_frame = tk.Frame(self.root, bg=self.BG)
        self.bars_frame.pack(fill="x", padx=24)
        self.bar_widgets = []
        for d in range(10):
            row = tk.Frame(self.bars_frame, bg=self.BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=str(d), font=("Courier New", 9, "bold"),
                     fg=self.MUTED, bg=self.BG, width=2).pack(side="left")
            bar_bg = tk.Frame(row, bg="#222", height=8, width=200)
            bar_bg.pack(side="left", padx=4)
            fill = tk.Frame(bar_bg, bg=self.ACCENT, height=8, width=0)
            fill.place(x=0, y=0)
            pct_lbl = tk.Label(row, text="", font=("Courier New", 8),
                               fg=self.MUTED, bg=self.BG, width=6)
            pct_lbl.pack(side="left")
            self.bar_widgets.append((fill, pct_lbl))

        # Buttons
        btn_row = tk.Frame(self.root, bg=self.BG)
        btn_row.pack(pady=12)

        self.predict_btn = tk.Button(
            btn_row, text="🔍 PREDICT", font=("Courier New", 11, "bold"),
            bg=self.ACCENT, fg="#000", activebackground="#8b5cf6",
            bd=0, padx=20, pady=8, cursor="hand2",
            command=self.predict, state="disabled"
        )
        self.predict_btn.pack(side="left", padx=6)

        tk.Button(btn_row, text="🗑  CLEAR", font=("Courier New", 11, "bold"),
                  bg="#333", fg=self.TEXT, activebackground="#444",
                  activeforeground=self.TEXT, bd=0, padx=20, pady=8,
                  cursor="hand2", command=self.clear).pack(side="left", padx=6)

        # Status
        self.status_var = tk.StringVar(value="Loading model…")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Courier New", 9), fg=self.MUTED, bg=self.BG,
                 pady=8).pack()

    def _load_model(self):
        self.model = get_model()
        self.root.after(0, lambda: self.status_var.set("✓ Ready — draw a digit and click Predict"))
        self.root.after(0, lambda: self.predict_btn.configure(state="normal"))

    def _start_draw(self, e):
        if self.model is None:
            return
        self.drawing = True
        self.canvas.delete(self.hint)

    def _draw(self, e):
        if not self.drawing:
            return
        r = 14
        x0, y0, x1, y1 = e.x-r, e.y-r, e.x+r, e.y+r
        self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        self.draw_ctx.ellipse([x0, y0, x1, y1], fill=255)

    def _stop_draw(self, e):
        self.drawing = False

    def predict(self):
        arr = np.array(self.draw_image.resize((28, 28))).reshape(1, 28, 28, 1) / 255.0
        preds = self.model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(preds))
        conf  = float(preds[digit])
        self.digit_var.set(str(digit))
        self.conf_var.set(f"{conf*100:.1f}%")
        for i, (fill, pct_lbl) in enumerate(self.bar_widgets):
            w = int(preds[i] * 200)
            fill.configure(width=max(w, 0),
                           bg=self.ACCENT if i == digit else "#555")
            pct_lbl.configure(text=f"{preds[i]*100:.1f}%")

    def clear(self):
        self.canvas.delete("all")
        self.draw_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw_ctx   = ImageDraw.Draw(self.draw_image)
        self.digit_var.set("?")
        self.conf_var.set("")
        for fill, pct_lbl in self.bar_widgets:
            fill.configure(width=0)
            pct_lbl.configure(text="")
        self.hint = self.canvas.create_text(
            CANVAS_SIZE//2, CANVAS_SIZE//2,
            text="Draw a digit here", fill=self.MUTED,
            font=("Courier New", 13)
        )


if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()
