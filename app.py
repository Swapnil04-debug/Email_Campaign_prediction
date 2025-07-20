from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

# Load model + preprocessing
model = tf.keras.models.load_model('final_ann_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    print("🟡 Raw features received from frontend:", features)

    scaled = scaler.transform(features)

    print("🔵 Scaled features (sent to model):", scaled)

    proba = model.predict(scaled)
    predicted_class = int(np.argmax(proba, axis=1)[0])

    return jsonify({
        "predicted_class": predicted_class,
        "class_probabilities": proba.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True,port=5050)
