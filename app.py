# Importing libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pickled model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    """
    This function renders and displays the html template.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    This function takes the input from the form and predicts using the model.pkl file and displays the prediction.
    """
    # Get user input
    brand = request.form['brand']
    transmission = request.form['transmission']
    fuelType = request.form['fuelType']
    mileage = float(request.form['mileage'])
    engineSize = float(request.form['engineSize'])
    year = int(request.form['year'])

    # Creating dataframe using the input
    new_data = pd.DataFrame({
        'brand': [brand],
        'transmission': [transmission],
        'fuelType': [fuelType],
        'mileage': [mileage],
        'engineSize': [engineSize],
        'year': [year]
    })

    # List of columns in the training data
    columns_lst = ['year', 'tax', 'mileage', 'engineSize', 'brand_audi', 'brand_bmw',
                   'brand_ford', 'brand_hyundi', 'brand_merc', 'brand_skoda',
                   'brand_toyota', 'brand_vauxhall', 'brand_vw', 'transmission_Automatic',
                   'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto',
                   'fuelType_Diesel', 'fuelType_Electric', 'fuelType_Hybrid',
                   'fuelType_Other', 'fuelType_Petrol']

    # Encoding categorical variables in the new data
    new_data_encoded = pd.get_dummies(
        new_data, columns=['brand', 'transmission', 'fuelType'])

    # Filling missing variables
    for i in columns_lst:
        if i not in new_data_encoded.columns:
            new_data_encoded[i] = 0

    # Model prediction
    predicted_price = model.predict(new_data_encoded[columns_lst])
    predicted_price[0] = round(predicted_price[0], 2)

    return render_template('index.html', prediction=predicted_price)

# Condition to run the application
if __name__ == '__main__':
    app.run(debug=True)
