"""
@author: Wajed Afaneh
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset from the CSV file
file_path = '../datasets/diabetes_risk_prediction_dataset.csv'
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Polyuria'] = label_encoder.fit_transform(df['Polyuria'])
df['Polydipsia'] = label_encoder.fit_transform(df['Polydipsia'])
df['sudden weight loss'] = label_encoder.fit_transform(df['sudden weight loss'])
df['weakness'] = label_encoder.fit_transform(df['weakness'])
df['Polyphagia'] = label_encoder.fit_transform(df['Polyphagia'])
df['Genital thrush'] = label_encoder.fit_transform(df['Genital thrush'])
df['visual blurring'] = label_encoder.fit_transform(df['visual blurring'])
df['Itching'] = label_encoder.fit_transform(df['Itching'])
df['Irritability'] = label_encoder.fit_transform(df['Irritability'])
df['delayed healing'] = label_encoder.fit_transform(df['delayed healing'])
df['partial paresis'] = label_encoder.fit_transform(df['partial paresis'])
df['muscle stiffness'] = label_encoder.fit_transform(df['muscle stiffness'])
df['Alopecia'] = label_encoder.fit_transform(df['Alopecia'])
df['Obesity'] = label_encoder.fit_transform(df['Obesity'])
df['Polyphagia'] = label_encoder.fit_transform(df['Polyphagia'])
# Write the transformed dataset back to a CSV file
transformed_file_path = '../datasets/processed_diabetes_risk_prediction_dataset.csv'
df.to_csv(transformed_file_path, index=False)
print(f'Transformed data saved to: {transformed_file_path}')