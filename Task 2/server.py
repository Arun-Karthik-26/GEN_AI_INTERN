from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input'][0]
        print(data)

        # Prepare the input for the model
        input_data = {
            'Age': int(data['Age']),
            'Gender': int(data['Gender']),
            'MaritalStatus': int(data['MaritalStatus']),
            'Education': int(data['Education']),
            'Department': int(data['Department']),
            'JobRole': int(data['JobRole']),
            'YearsAtCompany': int(data['YearsAtCompany']),
            'YearsInCurrentRole': int(data['YearsInCurrentRole']),
            'MonthlyIncome': float(data['MonthlyIncome']),
            'JobSatisfaction': int(data['JobSatisfaction']),
            'WorkLifeBalance': int(data['WorkLifeBalance']),
            'TrainingTimesLastYear': int(data['TrainingTimesLastYear']),
            'OverTime': int(data['OverTime']),
            'DistanceFromHome': float(data['DistanceFromHome'])  # Change to float
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)
        if(prediction[0]==1):
            res='employee will leave the company'
        else:
            res='employee will not leave the company'

        # Convert prediction to standard Python type
        response = {
            'predictions': res  # Convert to int
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")  # Log the error message
        return jsonify({'error': str(e)}), 500  # Return the error as a JSON response

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
