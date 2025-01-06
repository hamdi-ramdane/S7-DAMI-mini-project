
import pandas as pd #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.tree import DecisionTreeClassifier #type:ignore
from sklearn.naive_bayes import GaussianNB #type:ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from imblearn.over_sampling import SMOTE #type:ignore
import pickle

data = pd.read_csv("./data/haberman.csv")

# Handling outliers for "nbr_axillary_nodes"
Q1 = data['nbr_axillary_nodes'].quantile(0.25)
Q3 = data['nbr_axillary_nodes'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

original_data = data.copy()
data = data[(data['nbr_axillary_nodes'] >= lower_bound) & (data['nbr_axillary_nodes'] <= upper_bound)]

# Standarization    
scaler = StandardScaler()
data[['age', 'operation_year', 'nbr_axillary_nodes']] = scaler.fit_transform(data[['age', 'operation_year', 'nbr_axillary_nodes']])

# Classification
X = data[['age', 'operation_year', 'nbr_axillary_nodes']]
y = data['survival_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Decision Tree 
dt_model = DecisionTreeClassifier(random_state=69)      # Initialize model 
dt_model.fit(X_train, y_train)                          # Train model
y_pred_dt = dt_model.predict(X_test)                    
dt_accuracy = accuracy_score(y_test, y_pred_dt)         # Evaluate model
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')
dt_precision = precision_score(y_test, y_pred_dt, average='weighted')
dt_recall = recall_score(y_test, y_pred_dt, average='weighted')

# NB 
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')
nb_precision = precision_score(y_test, y_pred_nb, average='weighted')
nb_recall = recall_score(y_test, y_pred_nb, average='weighted')


# Results 
results = {
    'Model': ['Decision Tree', 'Naive Bayes'],
    'Accuracy': [dt_accuracy, nb_accuracy],
    'F1-Score': [dt_f1, nb_f1]
}

# Using SMOTE to oversample the minority class
smote = SMOTE(random_state=69)
X_train_smote, y_train_smote = smote.fit_resample(X, y)


# Train the Decision Tree and Naive Bayes models with SMOTE data
dt_smote = DecisionTreeClassifier(random_state=69)
nb_smote = GaussianNB()
dt_smote.fit(X_train_smote, y_train_smote)
nb_smote.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_dt = dt_smote.predict(X_test)
y_pred_nb = nb_smote.predict(X_test)

# Evaluate the models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate with SMOTE data
dt_smote_accuracy, dt_smote_precision, dt_smote_recall, dt_smote_f1= evaluate_model(y_test, y_pred_dt)
nb_smote_accuracy, nb_smote_precision, nb_smote_recall, nb_smote_f1= evaluate_model(y_test, y_pred_nb)

# Print the results
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Naive Bayes', 'Decision Tree (SMOTE)', 'Naive Bayes (SMOTE)'],
    'Accuracy': [dt_accuracy, nb_accuracy, dt_smote_accuracy, nb_smote_accuracy],
    'F1 Score': [dt_f1, nb_f1, dt_smote_f1, nb_smote_f1],
    'Precision': [dt_precision,nb_precision, dt_smote_precision, nb_smote_precision],
    'Recall': [dt_recall, nb_recall, dt_smote_recall, nb_smote_recall]
})
print(results)

model = dt_smote  # Replace with your trained model, e.g., DecisionTreeClassifier()
filename = 'dt_model.pkl'  # Name of the file to save the model

with open(filename, 'wb') as file:
    pickle.dump(model, file)

model = nb_smote  # Replace with your trained model, e.g., DecisionTreeClassifier()
filename = 'nb_model.pkl'  # Name of the file to save the model

with open(filename, 'wb') as file:
    pickle.dump(model, file)

model = scaler# Replace with your trained model, e.g., DecisionTreeClassifier()
filename = 'scaler.pkl'  # Name of the file to save the model

with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Input prediction
def predict_survival():
    print("\n--- Predict Survival Status ---")
    age = int(input("Enter age: "))
    operation_year = int(input("Enter operation year (year - 1900): "))
    nbr_axillary_nodes = int(input("Enter number of axillary nodes: "))

    # Create a DataFrame for the input with the same column names as the training data
    input_data = pd.DataFrame([[age, operation_year, nbr_axillary_nodes]], 
                              columns=['age', 'operation_year', 'nbr_axillary_nodes'])
    
    # Standardize input using the scaler
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=['age', 'operation_year', 'nbr_axillary_nodes'])


    # Predictions
    dt_prediction = dt_model.predict(input_data_scaled)[0]
    nb_prediction = nb_model.predict(input_data_scaled)[0]

    print("\nPredictions:")
    print(f"Decision Tree Prediction: {dt_prediction}")
    print(f"Naive Bayes Prediction: {nb_prediction}")

# Call the prediction function
# predict_survival()
