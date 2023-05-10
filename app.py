from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled linear regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def prediction():
    return render_template('index.html')

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.form
    prediction_features = np.array([data['feature_1'], data['feature_2'], data['feature_3'],data['feature_4'],data['feature_5']]).astype(float)
    prediction = model.predict(prediction_features.reshape(1,-1))[0]
    return render_template('Prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
