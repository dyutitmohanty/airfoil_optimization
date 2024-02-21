import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

main_directory = r'<insert path to Data folder here>'
csv_pattern = '*.csv'
dataframes = []

for root, _, files in os.walk(main_directory):
    print(root)
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            # Read the CSV file and skip the first row (header)
            df = pd.read_csv(file_path)
            print(df.shape)
            dataframes.append(df)

# Concatenate DataFrames
combined_dataframe = pd.concat(dataframes, ignore_index=True)

# Remove rows with NaN values
combined_dataframe.dropna(inplace=True)
cl_condition = (combined_dataframe['Cl'] >= -2) & (combined_dataframe['Cl'] <= 2)
cd_condition = (combined_dataframe['Cd'] >= 0.0005) & (combined_dataframe['Cd'] <= 0.09)
combined_dataframe = combined_dataframe[cl_condition & cd_condition]
combined_dataframe = combined_dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

# scaler = StandardScaler()
# combined_dataframe.iloc[:, 0] = scaler.fit_transform(combined_dataframe.iloc[:, 0].values.reshape(-1, 1))


# Save the cleaned DataFrame as a CSV file
output_csv_file = 'COMPILED_AIRFOIL_DATA.csv'
combined_dataframe.to_csv(output_csv_file, index=False)

print(f"Cleaned data saved to '{output_csv_file}'")
print("Cleaned Data Shape: ", combined_dataframe.shape)

