from flask import Flask, request, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('decision_tree_model.pkl')
encoder = joblib.load('encoder.pkl')

# Route to display the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction when the form is submitted
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {
        'Account length': request.form['account_length'],
        'Area code': request.form['area_code'],
        'Number vmail messages': request.form['number_vmail_messages'],
        'Total day minutes': request.form['total_day_minutes'],
        'Total day calls': request.form['total_day_calls'],
        'Total day charge': request.form['total_day_charge'],
        'Total eve minutes': request.form['total_eve_minutes'],
        'Total eve calls': request.form['total_eve_calls'],
        'Total eve charge': request.form['total_eve_charge'],
        'Total night minutes': request.form['total_night_minutes'],
        'Total night calls': request.form['total_night_calls'],
        'Total night charge': request.form['total_night_charge'],
        'Total intl minutes': request.form['total_intl_minutes'],
        'Total intl calls': request.form['total_intl_calls'],
        'Total intl charge': request.form['total_intl_charge'],
        'Customer service calls': request.form['custserv_calls'],
        'State': request.form['state'],
        'International plan': request.form['international_plan'],
        'Voice mail plan': 'No'  # Assuming a default value for simplicity
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode the categorical variables using the loaded encoder
    categorical_cols = ['State', 'International plan', 'Voice mail plan']
    encoded_data = encoder.transform(input_df[categorical_cols])

    # Convert encoded data to DataFrame and merge with input data
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
    input_df = input_df.drop(columns=categorical_cols)
    final_input = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Make the prediction using the trained model
    prediction = model.predict(final_input)
    result = "Yes" if prediction[0] == 1 else "No"

    return render_template('index.html', prediction_text=f'Churn Prediction: {result}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
