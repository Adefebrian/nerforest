from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)

# load model
model_path = "model_1.pkl"
model = tf.keras.models.load_model(model_path)

# load tokenizer
tokenizer_path = "tokenizer.json"
with open(tokenizer_path, 'r') as f:
    tokenizer = json.load(f)

# define label dictionary
label_dict = {0: "bukan_kebakaran", 1: "kebakaran", 2: "penanganan"}

@app.route('/predict', methods=['POST'])
def predict():
    # get text input from request
    text = request.json['text']

    # preprocess input text
    text = tokenizer.texts_to_sequences([text])
    text = tf.keras.preprocessing.sequence.pad_sequences(text, padding='post', maxlen=64)

    # make prediction
    label = model.predict(text)
    label = np.argmax(label, axis=1)[0]
    label = label_dict[label]

    # return prediction as JSON
    prediction = {"text": text[0].tolist(), "label": label}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
