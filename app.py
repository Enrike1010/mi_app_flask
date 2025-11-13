from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Cargar modelo y scaler
model = load_model("modelo_multienfermedades.keras")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener valores del formulario
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        glucose = float(request.form['glucose'])
        cholesterol = float(request.form['cholesterol'])
        smoking = float(request.form['smoking'])
        exercise = float(request.form['exercise'])
        alcohol = float(request.form['alcohol'])
        family_history = float(request.form['family_history'])

        # Crear array y normalizar
        datos = np.array([[age, sex, bmi, glucose, cholesterol, smoking, exercise, alcohol, family_history]])
        datos_norm = scaler.transform(datos)

        # Predecir
        pred = model.predict(datos_norm)
        pred_bin = (pred > 0.5).astype(int)
        resultado = pd.DataFrame(pred_bin, columns=["diabetes","hypertension","heart_disease","obesity","lung_cancer"])

        return render_template('index.html', prediction_text=resultado.to_html(index=False))
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
