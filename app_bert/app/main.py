from flask import Flask 
import pandas as pd
from fast_bert.prediction import BertClassificationPredictor
from flask import Flask, jsonify, request
import re
  
app = Flask(__name__) 
app.config.from_object(__name__)

MODEL_PATH = 'model/'

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path='', 
				multi_label=True,
                use_fast_tokenizer=False,
				model_type='bert',
				do_lower_case=False)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            texto = request.data
            s = re.sub('\W+',' ', texto.decode('ASCII'))
            respuesta = predictor.predict(s.split('bertmedicalstring ')[1])
            return jsonify({'Clase1': respuesta[0][0],'Puntaje1':respuesta[0][1],
            'Clase2': respuesta[1][0],'Puntaje2':respuesta[1][1]})

@app.route("/") 
def home_view(): 
        return "<h1>Medical Bert</h1>"

