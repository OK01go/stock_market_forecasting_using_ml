from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Load models
def load_model_safe(path, name):
    try:
        model = load_model(path)
        logging.info(f"✅ {name} model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading {name} model: {e}")
        return None

lstm_model = load_model_safe("models/lstm_model.h5", "LSTM")
rnn_model = load_model_safe("models/rnn_stock_model.h5", "RNN")
gru_model = load_model_safe("models/gru_stock_model.h5", "GRU")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model')
        input_data = request.form.get('data')

        logging.info("📩 Received prediction request")
        logging.info(f"   - Model Type: {model_type}")
        logging.info(f"   - Data Sample: {input_data[:50]}..." if input_data else "No data")

        if not input_data or not model_type:
            return jsonify({'error': 'Missing model type or input data'}), 400

        try:
            input_list = [float(i) for i in input_data.split(',') if i.strip()]
        except ValueError:
            return jsonify({'error': 'Input data must contain only numeric values.'}), 400

        if len(input_list) < 60:
            return jsonify({'error': 'Please enter at least 60 past prices.'}), 400

        sequence = input_list[-60:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sequence = scaler.fit_transform(np.array(sequence).reshape(-1, 1)).flatten().tolist()

        model = {'lstm': lstm_model, 'rnn': rnn_model, 'gru': gru_model}.get(model_type.lower())

        if model is None:
            return jsonify({'error': f'Model "{model_type}" is not available or failed to load.'}), 500

        predictions = []
        for _ in range(30):
            input_array = np.array(scaled_sequence).reshape((1, 60, 1))
            scaled_pred = model.predict(input_array, verbose=0)[0][0]
            pred_price = scaler.inverse_transform([[scaled_pred]])[0][0]
            predictions.append(round(float(pred_price), 2))
            scaled_sequence.append(scaled_pred)
            scaled_sequence = scaled_sequence[1:]

        return jsonify({'predictions': predictions})

    except Exception as e:
        logging.error("🚨 Error during prediction: " + str(e))
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"🚀 Starting Flask app on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
