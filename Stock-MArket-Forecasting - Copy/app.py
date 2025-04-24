from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
CORS(app)

# Load deep learning models
try:
    lstm_model = load_model("models/lstm_model.h5")
    print("✅ LSTM model loaded successfully")
except Exception as e:
    print("❌ Error loading LSTM model:", e)
    lstm_model = None

try:
    rnn_model = load_model("models/rnn_stock_model.h5")
    print("✅ RNN model loaded successfully")
except Exception as e:
    print("❌ Error loading RNN model:", e)
    rnn_model = None

try:
    gru_model = load_model("models/gru_stock_model.h5")
    print("✅ GRU model loaded successfully")
except Exception as e:
    print("❌ Error loading GRU model:", e)
    gru_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model')
        input_data = request.form.get('data')

        print("📩 Received request:")
        print("   - Model Type:", model_type)
        print("   - Data Sample:", input_data[:50] + "..." if input_data else "No data")

        if not input_data or not model_type:
            return jsonify({'error': 'Missing model type or input data'}), 400

        # Convert input string to float list
        input_list = [float(i) for i in input_data.split(',') if i.strip()]
        if len(input_list) < 60:
            return jsonify({'error': 'Please enter at least 60 past prices.'}), 400

        # Take last 60 entries
        sequence = input_list[-60:]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sequence = scaler.fit_transform(np.array(sequence).reshape(-1, 1)).flatten().tolist()

        predictions = []

        # Select model
        model = None
        if model_type == 'lstm':
            model = lstm_model
        elif model_type == 'rnn':
            model = rnn_model
        elif model_type == 'gru':
            model = gru_model

        if model is None:
            return jsonify({'error': f'Model "{model_type}" is not available or failed to load.'}), 500

        # Predict 30 days
        for _ in range(30):
            input_array = np.array(scaled_sequence).reshape((1, 60, 1))
            scaled_pred = model.predict(input_array, verbose=0)[0][0]
            pred_price = scaler.inverse_transform([[scaled_pred]])[0][0]
            predictions.append(round(float(pred_price), 2))
            scaled_sequence.append(scaled_pred)
            scaled_sequence = scaled_sequence[1:]

        return jsonify({'predictions': predictions})

    except Exception as e:
        print("🚨 Error during prediction:", str(e))
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask app on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
