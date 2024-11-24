from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load model, scaler, and label encoders
regression_model = joblib.load('regression_model.pkl')
scaler = joblib.load('scaler_regression.pkl')
label_encoder_soil = joblib.load('label_encoder_soil.pkl')
label_encoder_crop = joblib.load('label_encoder_crop.pkl')

# Get the mapping of soil and crop types for dropdown options
soil_classes = dict(enumerate(label_encoder_soil.classes_))
crop_classes = dict(enumerate(label_encoder_crop.classes_))

@app.route('/', methods=['GET', 'POST'])
def index():
    # Reset values on GET request
    moisture = None
    actual_moisture = None
    difference = None

    if request.method == 'POST':
        try:
            # Get user inputs from the form
            soil_type = int(request.form['Soil Type'])
            crop_type = int(request.form['Crop Type'])
            temperature = float(request.form['Temparature'])
            humidity = float(request.form['Humidity '])
            nitrogen = float(request.form['Nitrogen'])
            phosphorous = float(request.form['Phosphorous'])
            potassium = float(request.form['Potassium'])

            # Prepare data for prediction
            input_data = np.array([soil_type, crop_type, temperature, humidity, nitrogen, phosphorous, potassium]).reshape(1, -1)
            input_data_scaled = scaler.transform(input_data)

            # Predict moisture level
            moisture = regression_model.predict(input_data_scaled)[0]

            # Calculate the difference if actual moisture is provided
            if 'actual_moisture' in request.form and request.form['actual_moisture']:
                actual_moisture = float(request.form['actual_moisture'])
                difference = abs(moisture - actual_moisture)  # Absolute difference for clarity
        except Exception as e:
            # Handle errors (optional)
            print(f"Error during prediction: {e}")

    return render_template('index.html', moisture=moisture, actual_moisture=actual_moisture, difference=difference, soil_classes=soil_classes, crop_classes=crop_classes)

if __name__ == '__main__':
    app.run(port="1000", debug=True)






