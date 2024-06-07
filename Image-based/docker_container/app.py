import pandas as pd
from flask import Flask, jsonify,request
import tensorflow as tf
import pre_process
import post_process


app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')

@app.route('/predict',methods=['POST','GET'])
def predict():
    req = request.json.get('instances')
    
    input_data = req[0]['content']

    #preprocessing
    input_data = pre_process.preprocess(input_data)

    #predict
    prediction = model.predict(input_data)

    #postprocessing
    value = post_process.postprocess(prediction) 
    output = {'predictions':[
        {
           'prediction' : value
        }
        ]
        }
    return jsonify(output)

@app.route('/healthz')
def healthz():
    return "OK"


if __name__=='__main__':
    app.run(host='0.0.0.0')