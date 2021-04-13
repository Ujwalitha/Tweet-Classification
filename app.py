from flask import Flask,render_template,url_for,request
import keras
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		test_sequences = tokenizer.texts_to_sequences(data)
		test_padded = pad_sequences(test_sequences, padding='post', truncating='post', maxlen=50)
		model = load_model("my_model.h5",compile=False)
		my_prediction = model.predict(test_padded)
		print(my_prediction)
		pred = 1 if my_prediction >= 0.5 else 0
		print(pred)
		#my_prediction=np.round(my_prediction).astype(int).reshape(3263)
	return render_template('home.html',prediction = pred)

if __name__ == '__main__':
	app.run(debug=True)