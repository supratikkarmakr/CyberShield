import pickle

import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load the trained model


model = tf.keras.models.load_model('model.h5')



# Load the Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Max length used during training
MAX_LEN = 100  # Update based on your original notebook (default guess)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    if request.method == 'POST':
        text = request.form['text']
        if text:
            # Preprocess
            sequence = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

            # Predict
            pred = model.predict(padded)
            label = 'Cyberbullying' if pred.argmax() == 1 else 'Not Cyberbullying'
            prediction = f"Prediction: {label}"
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
