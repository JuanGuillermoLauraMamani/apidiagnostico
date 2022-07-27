# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:27:56 2022

@author: Guille
"""

from flask import Flask, jsonify, request
import sklearn.externals
import json
import joblib
import numpy as np
import sklearn
from flask_cors import CORS, cross_origin



#curl -d "{\"Sintomas\":[[30,1,5,1,1,2,1,0,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0]]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predecir

app= Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


sintomas = [ ]

@app.route("/")
@cross_origin()
def home():
    
    
    return jsonify(sintomas)

@app.route("/predecir", methods=["POST"])
@cross_origin()
def predecir():
    reqsintomas=request.get_json(force=True)
    print(reqsintomas)
    sintomas=reqsintomas['Sintomas']
    print(sintomas)
    clf=joblib.load('modelo_entrenado.pkl')
    prediccion=clf.predict(sintomas)
    print(str(prediccion[0]))
    
    res ={
        "descr":sintomas,
        "pred":str(prediccion[0])
        }
    
    #return "prediccion "+str(prediccion[0])
    return jsonify(res)


if __name__ == '__main__':
    app.run()