from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('modelo_enfermedades.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = np.array([data])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        return render_template('index.html', prediction_text=f'Resultado: {prediction[0]}')
    except Exception as e:
        return jsonify({'error': str(e)})

# Render detecta la variable 'app', no hace falta app.run()
