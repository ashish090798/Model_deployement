from sklearn.externals import joblib
import sys
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)
clf = joblib.load(r'C:\Users\vish\Documents\Data\Deployment\model 2\model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        json_ = request.get_json(force =True)
        #print(json_)
        #print(type(json_))
        query = pd.DataFrame(json_)
        
        prediction = list(clf.predict(query))
        print('Prediction : '+ str(prediction))
        return jsonify({'prediction': str(prediction)})
    else:
        print('no model here')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')