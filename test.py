"""
@author: Wajed Afaneh
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from neural_network import NeuralNetwork

# Load the dataset from a CSV file
file_path = 'datasets/IRIS.csv'
dataset = pd.read_csv(file_path)

# Assuming the last column is the target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Normalize the data using StandardScaler
scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Convert string labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 'Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Softmax', 'Linear'

# Initialize and train the neural network
nn = NeuralNetwork(
    hidden_size=8,
    activation_function='Sigmoid',
    output_function='Softmax',
    epochs=10,
    learning_rate=0.01,
    goal=0.99
)
nn.train(X_train, y_train)

# Predict on the test set
predictions = nn.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, predictions)

print('*******************************')
print(f'Test Accuracy: {accuracy:.2%}')
