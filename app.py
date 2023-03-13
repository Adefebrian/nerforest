from flask import Flask, jsonify, request
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the pickled model
with open('model_1.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the tokenizer
tokenizer = Tokenizer()
with open('data.txt', 'r') as file:
    data = file.readlines()
tokenizer.fit_on_texts(data)

# Define the Flask app
app = Flask(__name__)

# Define the route for the model deployment information
@app.route('/', methods=['GET'])
def home():
    return "Model untuk memprediksi label teks 0, 1, atau 2 telah berhasil di-deploy menggunakan Flask!"

# Define the route for text prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'Text is missing or empty'})
    
    # Preprocess the text
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=50)
    
    # Predict the label for the text
    try:
        label_id = np.argmax(model.predict(padded_sequence))
    except Exception as e:
        return jsonify({'error': 'Failed to predict label', 'details': str(e)})
    
    # Manipulate the label
    if label_id == 0:
        label = 0
    elif label_id == 1:
        label = 1
    else:
        label = 2
    
    # Return the prediction as a JSON response
    return jsonify({'text': text, 'label': label})

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
