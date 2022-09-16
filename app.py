# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:35:14 2022

@author: Guille
"""

import numpy as np
from keras.models import load_model
from tensorflow import keras
from flask import Flask, jsonify, request
import time,os
import sklearn.externals
import json
import joblib
import numpy as np
import sklearn
from flask_cors import CORS, cross_origin


# Load the model
model = load_model('modelo_entrenado.h5')

def pre_process(arr):
    return np.argmax(model.predict(arr), axis=-1)[0]


app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

sintomas = [ ]

# Create a method for /
@app.route("/")
def home():
    return "<h1>Servicio iniciado RNA2!</h1>"

@app.route('/predict',methods=['POST'])
def predict_():
    
    reqsintomas=request.get_json(force=True)
    print(reqsintomas)
    sintomas=reqsintomas['Sintomas']
    print(sintomas)

 
    digit_class = pre_process(sintomas)  

  
    res ={
        "pred":int(digit_class))
    }
    
   
  
    return jsonify(res)


if __name__ == '__main__':
    app.run()