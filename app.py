from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

app = Flask(__name__)

# Load the trained model
model = None  # Initialize the model variable

# Load the dataset for preprocessing
df = pd.read_csv('new_kidney_dis_det_dataset.csv')


# Split the data into features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the route for the main page
@app.route('/')
def home():
    return render_template('form.html')

# Define the route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model  # Use the global model variable

        # Check if the model is loaded, if not, load it
        if model is None:
            # Initialize the model (modify hyperparameters as needed)
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Save the trained model to a file
            with open('model.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)

        # Get data from the form (replace with actual form fields)
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = float(request.form['rbc'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        wbcc = float(request.form['wbcc'])
        rbcc = float(request.form['rbcc'])
        htn = float(request.form['htn'])

        # Create a NumPy array with the input data
        input_data = np.array([[bp, sg, al, su, rbc, bu, sc, sod, pot, hemo, wbcc, rbcc, htn]])

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)

        # Evaluate model performance on the testing set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        if prediction == 1:
            message = "High likelihood of Chronic Kidney Disease. Consult a healthcare professional."
        else:
            message = "Low likelihood of Chronic Kidney Disease. Regular health monitoring is advised."
        # Return the prediction result, accuracy, and AUC score
        return render_template('result.html', message=message, accuracy=accuracy, auc_score=auc_score)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
