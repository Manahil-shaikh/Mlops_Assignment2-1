#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[23]:


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




