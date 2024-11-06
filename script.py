# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:52:40 2024

@author: podevyn Bert
"""
# Import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the CSV file
data = pd.read_csv("./DSFI_Assignment/EPC/Dataset Realo.csv")

# Create a subset with only houses
housesdata = data[data["housingType"] == "HOUSE"]

# Summary statistics
housesdata.describe(include = 'all')


# Define the function to categorize EPC labels
def categorize_epc_label(energyConsumption):
    if pd.isna(energyConsumption):  # Check for missing values
        return np.nan
    elif 0 <= energyConsumption <= 100:
        return "A"
    elif 100 < energyConsumption <= 200:
        return "B"
    elif 200 < energyConsumption <= 300:
        return "C"
    elif 300 < energyConsumption <= 400:
        return "D"
    elif 400 < energyConsumption <= 500:
        return "E"
    else:
        return "F"

# Apply the function to the DataFrame and create the EPC_label column
housesdata['EPC_label'] = housesdata['energyConsumption'].apply(categorize_epc_label)

#visualtization
# Drop NaN values from EPC_label to focus on actual categories
filtered_data = housesdata['EPC_label'].dropna()

# Plot the histogram (counts of each label)
plt.figure(figsize=(8, 6))
filtered_data.value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of EPC Labels")
plt.xlabel("EPC Label")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()


# Create a for-loop to check the number of NaN's per column
for column in data.columns:
    nans = data[column].isna().sum()  # Count NaNs in the column
    total = len(data[column])         # Total observations in the column
    percentage = (nans / total) * 100  # Calculate percentage of NaNs
    print(f"{column}: {percentage:.2f}% missing values")

# It's clear that some of the columns have a lot of missing values
# Let's create a loop which removes all variables which have more
# missing values than a certain threshold 
def filter_columns_by_nan_threshold(df, threshold=0.50):
    # Calculate the percentage of NaN values for each column
    nan_percentage = df.isna().mean()
    
    # Filter columns that have NaN percentage below the threshold
    columns_to_keep = nan_percentage[nan_percentage <= threshold].index
    
    # Create a new DataFrame with only the selected columns
    filtered_df = df[columns_to_keep]
    
    return filtered_df

# Create a new cleaned dataset
housesdata_cleaned = filter_columns_by_nan_threshold(housesdata)

# Now we will remove all the variables which we will not use in the final
# regression
housesdata_cleaned.columns
housesdata_cleaned.drop(columns = ['Unnamed: 0', 'addressId', 'admin1Id',
                                   'admin1', 'admin2Id', 'admin2','localityId',
                                   'locality', 'subLocalityId', 'subLocality',
                                   'districtId','postalCode', 'niscode',
                                   'region', 'addressType','lastListing',
                                   'housingType','numberOfSides','price','way',
                                   'address', 'marker',
                                   ], inplace = True)

# Change initialPrice into the log of initialPrice but keep the original column
housesdata_cleaned['logInitialPrice'] = np.log(housesdata_cleaned['initialPrice'])
housesdata_cleaned.drop(columns = ['initialPrice'], inplace = False)

# Extract only the year from the firstListing variable
housesdata_cleaned['firstListing'] = pd.to_datetime(housesdata_cleaned['firstListing'], format = '%Y-%m-%d')
housesdata_cleaned['firstListing'] = housesdata_cleaned['firstListing'].dt.year
housesdata_cleaned['firstListing'] = housesdata_cleaned['firstListing'].astype(str)

# Create dummy variables for the 'EPC_label' column
epc_dummies = pd.get_dummies(housesdata_cleaned['EPC_label'], prefix='EPC').astype(int)
# Concatenate the dummy variables with the original dataset
housesdata_cleaned = pd.concat([housesdata_cleaned, epc_dummies], axis=1)

#regressions for one year
#control variables collections
control_vars_build_char = ['habitableArea', 'buildYear', 'numberOfBedrooms', 'isNewBuild']
control_vars_qual_char = ['floodProneLocation','greenCoverage']
control_vars_energy_char = ['hasSolarPanels', 'hasCentralHeating', 'hasSolarBoiler']

#regressions for one year
# Select dummy variables and control variables
# List of EPC dummy variables (excluding one to avoid the dummy variable trap)
epc_dummies = ['EPC_A', 'EPC_B', 'EPC_C', 'EPC_D', 'EPC_E']  # Omit 'EPC_F' for example

# Check for NaN values
print(housesdata_cleaned[epc_dummies + control_vars_build_char + ['logInitialPrice']].isna().sum())

# Drop rows with NaN values in the relevant columns
housesdata_cleaned = housesdata_cleaned.dropna(subset=epc_dummies + control_vars_build_char + ['logInitialPrice'])

# Ensure all data types are correct
housesdata_cleaned[epc_dummies + control_vars_build_char] = housesdata_cleaned[epc_dummies + control_vars_build_char].apply(pd.to_numeric, errors='coerce')


# Combine EPC dummy variables and control variables
independent_vars = housesdata_cleaned[epc_dummies + control_vars_build_char]
independent_vars = sm.add_constant(independent_vars)  # Adds an intercept term

# Define the dependent variable
dependent_var = housesdata_cleaned['logInitialPrice']

# Run the regression
model = sm.OLS(dependent_var, independent_vars)
results = model.fit()

# Display the regression results
print(results.summary())

# Export regression results to DataFrame
results_df = pd.DataFrame({
    'Variable': results.params.index,
    'Coefficient': results.params.values,
    'Standard Error': results.bse.values,
    't-value': results.tvalues.values,
    'p-value': results.pvalues.values
})
# Export to Excel
results_df.to_excel('regression_results.xlsx', index=False)
print("Regression results exported to 'regression_results.xlsx'")


#creating aloop for each year
# Create a DataFrame to store all results
all_results = []

# Loop over each unique year in the dataset
for year in housesdata_cleaned['firstListing'].unique():
    print(f"Running regression for year: {year}")
    
    # Filter data for the specific year
    data_year = housesdata_cleaned[housesdata_cleaned['firstListing'] == year]

    # Check for NaN values
    print(data_year[epc_dummies + control_vars_build_char + ['logInitialPrice']].isna().sum())

    # Drop rows with NaN values in the relevant columns
    data_year = data_year.dropna(subset=epc_dummies + control_vars_build_char + ['logInitialPrice'])

    # Ensure all data types are correct
    data_year[epc_dummies + control_vars_build_char] = data_year[epc_dummies + control_vars_build_char].apply(pd.to_numeric, errors='coerce')

    # Combine EPC dummy variables and control variables
    independent_vars = data_year[epc_dummies + control_vars_build_char]
    independent_vars = sm.add_constant(independent_vars)  # Adds an intercept term

    # Define the dependent variable
    dependent_var = data_year['logInitialPrice']

    # Run the regression
    model = sm.OLS(dependent_var, independent_vars)
    results = model.fit()

    # Display the regression results
    print(results.summary())

    # Export regression results to DataFrame
    results_df = pd.DataFrame({
        'firstListing': year,
        'Variable': results.params.index,
        'Coefficient': results.params.values,
        'Standard Error': results.bse.values,
        't-value': results.tvalues.values,
        'p-value': results.pvalues.values
    })

    # Append the results for the current year to the all_results list
    all_results.append(results_df)

# Combine all results into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Export the combined results to Excel
final_results.to_excel('regression_results_by_year.xlsx', index=False)
print("All regression results exported to 'regression_results_by_year.xlsx'")