import json
import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the model and dataset
model = pickle.load(open('house_price_pred_model.pickle', 'rb'))
with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]

# Define the Flask application
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('ap.html', locations=__locations)

# Define the prediction function
def predict_price(location, sqft, bath, bhk_size):
    loc_index = __data_columns.index(location.lower())
    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk_size
    if loc_index >= 0:
        x[loc_index] = 1
    
    return model.predict([x])[0]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the user
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = float(request.form['bath'])
    bhk_size = int(request.form['bhk_size'])
    
    # Make a prediction using the model
    prediction = predict_price(location, sqft, bath, bhk_size)
    
    # Display the predicted price on the webpage
    return render_template('ap.html', prediction_text='The predicted price is ₹{} lakhs.'.format(round(prediction, 2)))

if __name__ == '__main__':
    app.run(debug=True)
