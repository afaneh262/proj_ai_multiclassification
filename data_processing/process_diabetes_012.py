"""
@author: Wajed Afaneh
"""
import pandas as pd

# Load the dataset from the CSV file
file_path = '../datasets/diabetes_012_health_indicators_BRFSS2021.csv'
df = pd.read_csv(file_path)

# Reorder the columns (assuming 'Diabetes_012' is the first column)
new_order = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'Diabetes_012'
]

df = df[new_order]

# Write the updated dataset to a new CSV file
new_file_path = '../datasets/processed_diabetes_012_health_indicators_BRFSS2021.csv'
df.to_csv(new_file_path, index=False)
print(f'Reordered data saved to: {new_file_path}')
