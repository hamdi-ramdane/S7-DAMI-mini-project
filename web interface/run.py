from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scaler (ensure these are saved in advance)
scaler = pickle.load(open('models/scaler.pkl', 'rb'))  # Replace with your scaler
dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))  # Replace with your model
nb_model = pickle.load(open('models/nb_model.pkl', 'rb'))  # Replace with your model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get data from form
            age = int(request.form['age'])
            operation_year = int(request.form['operation_year'])
            nbr_axillary_nodes = int(request.form['nbr_axillary_nodes'])

            # Prepare input data
            input_data = pd.DataFrame([[age, operation_year, nbr_axillary_nodes]], 
                                      columns=['age', 'operation_year', 'nbr_axillary_nodes'])

            # Scale input data
            input_data_scaled = pd.DataFrame(scaler.transform(input_data), 
                                             columns=['age', 'operation_year', 'nbr_axillary_nodes'])

            # Make predictions
            dt_prediction = dt_model.predict(input_data_scaled)[0]
            nb_prediction = nb_model.predict(input_data_scaled)[0]

            # Render the results on the same page
            return render_template('index.html', 
                                   dt_prediction=dt_prediction, 
                                   nb_prediction=nb_prediction)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


# 23, 55, 0 -> survived, survived

# 50, 50, 5 -> died, survived

# 50, 55, 30 -> died, died 