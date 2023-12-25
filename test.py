import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from neural_network import NeuralNetwork

i = 0
def on_progress_cal(output):
    i+output

datasets = ['processed_diabetes_012_health_indicators_BRFSS2021.csv', 'processed_diabetes_risk_prediction_dataset.csv', 'winequality-white.csv', 'IRIS.csv']
activation_functions = ['Sigmoid', 'Tanh', 'ReLU']
epochs_list = [5, 10, 20]

# Loop through datasets
for dataset_info in datasets:
    # Load the dataset from a CSV file
    file_path = dataset_info
    print('************************')
    print('Dataset: '+file_path)
    dataset = pd.read_csv("./datasets/"+file_path)

    # Assuming the last column is the target variable
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Convert string labels to integers using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Loop through output activation functions
    for activation_function in activation_functions:
        # Loop through epochs
        for epochs in epochs_list:

            # Initialize and train the neural network
            nn = NeuralNetwork(
                hidden_size=8,
                activation_function=activation_function,
                output_function='Softmax',
                epochs=epochs,
                learning_rate=0.01,
                goal=0.99,
                progress_cal=on_progress_cal
            )
            nn.train(X_train, y_train)

            # Predict on the test set
            predictions = nn.predict(X_test)

            # Calculate and print accuracy
            accuracy = accuracy_score(y_test, predictions)
            print(f'\nDataset: {dataset_info}, Activation: {activation_function}, Epochs: {epochs}, Test Accuracy: {accuracy:.2%}')