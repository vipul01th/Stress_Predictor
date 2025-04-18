import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
import numpy as np

# === Sample training data: [sleep, work, exercise, social] ===
X = np.array([
    [8, 4, 5, 7],   # well balanced
    [6, 8, 2, 3],   # overworked, low social
    [5, 10, 0, 1],  # high stress
    [7, 6, 3, 4],
    [4, 9, 1, 2],
    [9, 5, 4, 6],
    [6, 6, 2, 2],
    [8, 3, 5, 7],
    [5, 8, 1, 1]
])
y = [0, 1, 2, 1, 2, 0, 1, 0, 2]  # 0 = Low, 1 = Moderate, 2 = High

# Train model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)

# Stress level labels
stress_labels = {0: "Low Stress 😌", 1: "Moderate Stress 😐", 2: "High Stress 😫"}

# === Prediction function ===
def predict_stress():
    try:
        sleep = float(entry_sleep.get())
        work = float(entry_work.get())
        exercise = float(entry_exercise.get())
        social = float(entry_social.get())

        input_data = np.array([[sleep, work, exercise, social]])
        prediction = model.predict(input_data)[0]

        result_label.config(text=f"Predicted Stress Level: {stress_labels[prediction]}", fg="green" if prediction == 0 else "orange" if prediction == 1 else "red")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# === Tkinter UI ===
root = tk.Tk()
root.title("Stress Level Predictor")
root.geometry("400x400")
root.config(padx=20, pady=20)

tk.Label(root, text="Enter Sleep Hours:", font=("Arial", 12)).pack()
entry_sleep = tk.Entry(root)
entry_sleep.pack(pady=5)

tk.Label(root, text="Enter Work Hours:", font=("Arial", 12)).pack()
entry_work = tk.Entry(root)
entry_work.pack(pady=5)

tk.Label(root, text="Exercise per Week (days):", font=("Arial", 12)).pack()
entry_exercise = tk.Entry(root)
entry_exercise.pack(pady=5)

tk.Label(root, text="Social Activity Level (1–10):", font=("Arial", 12)).pack()
entry_social = tk.Entry(root)
entry_social.pack(pady=5)

tk.Button(root, text="Predict Stress Level", command=predict_stress, font=("Arial", 12)).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()
