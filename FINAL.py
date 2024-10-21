# %% [markdown]
# # intro MBF task 1
# 
# 
# #### as usual load the standard modules
# #### note, I am using Python 3.10.4 mostly other version will work as well
# #### -> not every package works on every version! -> creating venv is a best practice

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %% [markdown]
# # load in the data
# 
# #### obviously those filepaths are custom to your own device

# %%
data_MBF = pd.read_csv('./Assignment 1/MBF_FINAL_2.csv')

# Read the CPI data, setting 'Date' as the index and parsing it as datetime
CPI = pd.read_csv('./MBF_PythonBootcamp_Task1_CPI.csv', index_col='Date', parse_dates=True)

# %% [markdown]
# # Best practice 
# 
# ##### take a good and long first look at your data
# ##### experiment find out what might be issues what might be interesting
# ##### -> do not underestimate performing work only to find out issues with the data hours, days, weeks later!
# PLOT Rijhuis

# Step 1: Convert RentalPeriodFrom to datetime if not already done
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Step 2: Extract the year from RentalPeriodFrom and create a new column
data_MBF['Year'] = data_MBF['RentalPeriodFrom'].dt.year

# Step 3: Filter for 'Rijhuis'
rijhuis_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Rijhuis']

# Step 4: Calculate median rental fee and charges by year
rijhuis_median = rijhuis_data.groupby('Year').agg({
    'RentalFeeMonthly': 'mean',
    'RentalCharges': 'mean'
}).reset_index()

# Step 5: Filter data for years from 2017 to 2024
rijhuis_median = rijhuis_median[rijhuis_median['Year'].isin(range(2017, 2025))]

# Step 6: Store the 2017 values for comparison
rental_fee_2017_rijhuis = rijhuis_median.loc[rijhuis_median['Year'] == 2017, 'RentalFeeMonthly'].values[0]
rental_charges_2017_rijhuis = rijhuis_median.loc[rijhuis_median['Year'] == 2017, 'RentalCharges'].values[0]

# Calculate % change for years 2018 to 2024 based on 2017
rijhuis_median['RentalFeeChange'] = ((rijhuis_median['RentalFeeMonthly'] - rental_fee_2017_rijhuis) / rental_fee_2017_rijhuis) * 100
rijhuis_median['RentalChargesChange'] = ((rijhuis_median['RentalCharges'] - rental_charges_2017_rijhuis) / rental_charges_2017_rijhuis) * 100

# Step 7: Create the plot for 'Rijhuis'
plt.figure(figsize=(14, 7))
plt.plot(rijhuis_median['Year'], rijhuis_median['RentalFeeChange'], marker='o', label='Median Rental Fee Change (%)', color='blue')
plt.plot(rijhuis_median['Year'], rijhuis_median['RentalChargesChange'], marker='o', label='Median Rental Charges Change (%)', color='orange')
plt.title('Yearly Percentage Change in Median Rental Fee and Charges for Rijhuis (Base Year: 2016)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.axhline(0, color='gray', linestyle='--')  # Horizontal line at 0 for reference
plt.xticks(rijhuis_median['Year'], rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#PLOT Appartement
# Step 1: Filter for 'Appartement'
appartement_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Step 2: Calculate mean rental fee and charges by year
appartement_median = appartement_data.groupby('Year').agg({
    'RentalFeeMonthly': 'mean',
    'RentalCharges': 'mean'
}).reset_index()

# Step 3: Filter data for years from 2017 to 2024
appartement_median = appartement_median[appartement_median['Year'].isin(range(2017, 2025))]

# Step 4: Store the 2017 values for comparison
rental_fee_2017_appartement = appartement_median.loc[appartement_median['Year'] == 2017, 'RentalFeeMonthly'].values[0]
rental_charges_2017_appartement = appartement_median.loc[appartement_median['Year'] == 2017, 'RentalCharges'].values[0]

# Calculate % change for years 2018 to 2024 based on 2017
appartement_median['RentalFeeChange'] = ((appartement_median['RentalFeeMonthly'] - rental_fee_2017_appartement) / rental_fee_2017_appartement) * 100
appartement_median['RentalChargesChange'] = ((appartement_median['RentalCharges'] - rental_charges_2017_appartement) / rental_charges_2017_appartement) * 100

# Step 5: Create the plot for 'Appartement'
plt.figure(figsize=(14, 7))
plt.plot(appartement_median['Year'], appartement_median['RentalFeeChange'], marker='o', label='Mean Rental Fee Change (%)', color='blue')
plt.plot(appartement_median['Year'], appartement_median['RentalChargesChange'], marker='o', label='Mean Rental Charges Change (%)', color='orange')
plt.title('Yearly Percentage Change in Median Rental Fee and Charges for Appartement (Base Year: 2016)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.axhline(0, color='gray', linestyle='--')  # Horizontal line at 0 for reference
plt.xticks(appartement_median['Year'], rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#plot CPI evolution

# Step 1: Reset the index if 'Date' is the index
CPI = CPI.reset_index()

# Step 2: Ensure the 'Date' column is in datetime format
CPI['Date'] = pd.to_datetime(CPI['Date'], errors='coerce')

# Step 3: Extract the year from the 'Date' column
CPI['Year'] = CPI['Date'].dt.year

# Step 4: Filter the DataFrame for years between 2017 and 2024
cpi_filtered = CPI[CPI['Year'].isin(range(2017, 2025))]

# Step 5: Calculate yearly median CPI value
cpi_yearly = cpi_filtered.groupby('Year')['Value'].median().reset_index()

# Step 6: Calculate percentage change relative to 2017
baseline_2017 = cpi_yearly[cpi_yearly['Year'] == 2017]['Value'].values[0]  # Get the median CPI for 2017
cpi_yearly['CPI_Percentage_Change'] = ((cpi_yearly['Value'] - baseline_2017) / baseline_2017) * 100

# Step 7: Create a plot to visualize ONLY the percentage change
plt.figure(figsize=(10, 6))

# Plot only the CPI Percentage Change
plt.plot(cpi_yearly['Year'], cpi_yearly['CPI_Percentage_Change'], marker='o', color='orange', label='CPI Percentage Change')
plt.title('CPI Percentage Change from 2017 to 2024 (Relative to 2017)')
plt.xlabel('Year')
plt.ylabel('CPI Percentage Change (%)')

# Adding grid and legend
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#combined plot of the apartment and the CPI index

# Ensure 'RentalPeriodFrom' is in datetime format
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'], errors='coerce')

# Create a new 'Year' column by extracting the year from 'RentalPeriodFrom'
data_MBF['Year'] = data_MBF['RentalPeriodFrom'].dt.year



# PLOT 1: Appartement
# Step 1: Filter for 'Appartement'
appartement_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Step 2: Calculate mean rental fee and charges by year
appartement_median = appartement_data.groupby('Year').agg({
    'RentalFeeMonthly': 'mean',
    'RentalCharges': 'mean'
}).reset_index()

# Step 3: Filter data for years from 2017 to 2024
appartement_median = appartement_median[appartement_median['Year'].isin(range(2017, 2025))]

# Step 4: Store the 2017 values for comparison
rental_fee_2017_appartement = appartement_median.loc[appartement_median['Year'] == 2017, 'RentalFeeMonthly'].values[0]
rental_charges_2017_appartement = appartement_median.loc[appartement_median['Year'] == 2017, 'RentalCharges'].values[0]

# Step 5: Calculate % change for years 2018 to 2024 based on 2017
appartement_median['RentalFeeChange'] = ((appartement_median['RentalFeeMonthly'] - rental_fee_2017_appartement) / rental_fee_2017_appartement) * 100
appartement_median['RentalChargesChange'] = ((appartement_median['RentalCharges'] - rental_charges_2017_appartement) / rental_charges_2017_appartement) * 100

# PLOT 2: CPI Evolution
# Step 1: Reset the index if 'Date' is the index
CPI = CPI.reset_index()

# Step 2: Ensure the 'Date' column is in datetime format
CPI['Date'] = pd.to_datetime(CPI['Date'], errors='coerce')

# Step 3: Extract the year from the 'Date' column
CPI['Year'] = CPI['Date'].dt.year

# Step 4: Filter the DataFrame for years between 2017 and 2024
cpi_filtered = CPI[CPI['Year'].isin(range(2017, 2025))]

# Step 5: Calculate yearly median CPI value
cpi_yearly = cpi_filtered.groupby('Year')['Value'].median().reset_index()

# Step 6: Calculate percentage change relative to 2017
baseline_2017 = cpi_yearly[cpi_yearly['Year'] == 2017]['Value'].values[0]
cpi_yearly['CPI_Percentage_Change'] = ((cpi_yearly['Value'] - baseline_2017) / baseline_2017) * 100

# Combine Plots into a single figure
plt.figure(figsize=(14, 7))

# Plot the Rental Fee and Rental Charges Change for Appartement
plt.plot(appartement_median['Year'], appartement_median['RentalFeeChange'], marker='o', label='Mean Rental Fee Change (%) (Appartement)', color='blue')
plt.plot(appartement_median['Year'], appartement_median['RentalChargesChange'], marker='o', label='Mean Rental Charges Change (%) (Appartement)', color='green')

# Plot the CPI Percentage Change
plt.plot(cpi_yearly['Year'], cpi_yearly['CPI_Percentage_Change'], marker='o', label='CPI Percentage Change', color='orange')

# Add plot title and labels
plt.title('Percentage Change in Median Rental Fee, Charges for Appartement & CPI (Base Year: 2017)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')

# Add a horizontal line at 0% for reference
plt.axhline(0, color='gray', linestyle='--')

# Rotate the x-axis labels for readability
plt.xticks(appartement_median['Year'], rotation=45)

# Add legend and grid
plt.legend()
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

#PLOT NEW
# Ensure the 'Age' column is already created, as shown earlier
data_MBF['Age'] = 2024 - data_MBF['HuurderDateOfBirth']  # Calculate age if not already done

# Create a new DataFrame excluding rows where real estate type is 'Kamer' and age is <= 25
data_MBF_Without_Koten = data_MBF[~((data_MBF['MappedRealEstateType'] == 'Kamer') & (data_MBF['Age'] <= 25))]

# Ensure that the 'RentalPeriodFrom' column is in datetime format
data_MBF_Without_Koten['RentalPeriodFrom'] = pd.to_datetime(data_MBF_Without_Koten['RentalPeriodFrom'])

# Create the 'Year' column by extracting the year from 'RentalPeriodFrom'
data_MBF_Without_Koten['Year'] = data_MBF_Without_Koten['RentalPeriodFrom'].dt.year

# Step 1: Create age intervals for the tenants
# Define the age intervals
bins = [0, 24, 34, 44, 54, 64, 100]  # Define the bin edges
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']  # Define the corresponding labels

# Create the Age Group column using pd.cut()
data_MBF_Without_Koten['Age Group'] = pd.cut(data_MBF_Without_Koten['Age'], 
                                              bins=bins, 
                                              labels=labels, 
                                              right=True)

# Step 2: Filter data for the years 2018 and 2023
data_2018 = data_MBF_Without_Koten[data_MBF_Without_Koten['Year'] == 2018]
data_2023 = data_MBF_Without_Koten[data_MBF_Without_Koten['Year'] == 2023]

# Step 1: Group the data by Age Group and Real Estate Type for both years
grouped_2018 = data_2018.groupby(['Age Group', 'MappedRealEstateType']).size().reset_index(name='Count')
grouped_2023 = data_2023.groupby(['Age Group', 'MappedRealEstateType']).size().reset_index(name='Count')

# Step 2: Calculate total counts per Age Group for both years
total_counts_2018 = grouped_2018.groupby('Age Group')['Count'].sum().reset_index(name='Total')
total_counts_2023 = grouped_2023.groupby('Age Group')['Count'].sum().reset_index(name='Total')

# Step 3: Merge total counts into the grouped data
merged_2018 = pd.merge(grouped_2018, total_counts_2018, on='Age Group')
merged_2023 = pd.merge(grouped_2023, total_counts_2023, on='Age Group')

# Step 4: Calculate percentages
merged_2018['Percentage'] = (merged_2018['Count'] / merged_2018['Total']) * 100
merged_2023['Percentage'] = (merged_2023['Count'] / merged_2023['Total']) * 100

# Combine the two years for easy analysis or visualization
merged_2018['Year'] = 2018
merged_2023['Year'] = 2023
final_data = pd.concat([merged_2018, merged_2023], ignore_index=True)

# Step 1: Filter out rows where 'MappedRealEstateType' is 'Unknown'
final_data_cleaned = final_data[final_data['MappedRealEstateType'] != 'Unknown']

# percentage of realestatetype per age group
# Step 1: Get unique real estate types
real_estate_types = final_data_cleaned['MappedRealEstateType'].unique()

# Step 2: Set up the number of rows and columns for subplots
num_types = len(real_estate_types)
cols = 2  # Number of columns in the subplot
rows = (num_types + cols - 1) // cols  # Calculate the number of rows needed

# Step 3: Create a figure with subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Step 4: Loop through each real estate type and create a bar chart
for i, real_estate in enumerate(real_estate_types):
    # Filter data for the current real estate type
    type_data = final_data_cleaned[final_data_cleaned['MappedRealEstateType'] == real_estate]
    
    # Check if there is data for this type in both years
    if not type_data.empty:
        # Create a pivot table for percentages
        pivot_data = type_data.pivot_table(
            index='Age Group', 
            columns='Year', 
            values='Percentage', 
            fill_value=0
        )

        # Plot bars for both years
        bar_width = 0.35
        x = range(len(pivot_data))

        # Plot for 2018
        axs[i].bar(x, pivot_data[2018], width=bar_width, label='2018', color='skyblue')
        # Plot for 2023
        axs[i].bar([p + bar_width for p in x], pivot_data[2023], width=bar_width, label='2023', color='lightgreen')

        # Formatting the subplot
        axs[i].set_xlabel('Age Group', fontsize=12)
        axs[i].set_ylabel('Percentage (%)', fontsize=12)
        axs[i].set_title(f'Percentage of {real_estate} by Age Group (2018 vs 2023)', fontsize=14)
        axs[i].set_xticks([p + bar_width / 2 for p in x])
        axs[i].set_xticklabels(pivot_data.index, rotation=45)
        axs[i].legend(title='Year')

# Remove any empty subplots if the number of real estate types is odd
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Step 5: Show the combined plot
plt.tight_layout()
plt.show()

#percentage of agegroup per realestate type

# Step 1: Get unique age groups
age_groups = final_data_cleaned['Age Group'].unique()

# Step 2: Set up the number of rows and columns for subplots
num_groups = len(age_groups)
cols = 2  # Number of columns in the subplot
rows = (num_groups + cols - 1) // cols  # Calculate the number of rows needed

# Step 3: Create a figure with subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Step 4: Loop through each age group and create a bar chart
for i, age_group in enumerate(age_groups):
    # Filter data for the current age group
    age_data = final_data_cleaned[final_data_cleaned['Age Group'] == age_group]
    
    # Check if there is data for this age group in both years
    if not age_data.empty:
        # Create a pivot table for percentages
        pivot_data = age_data.pivot_table(
            index='MappedRealEstateType', 
            columns='Year', 
            values='Percentage', 
            fill_value=0
        )

        # Plot bars for both years
        bar_width = 0.35
        x = range(len(pivot_data))

        # Plot for 2018
        axs[i].bar(x, pivot_data[2018], width=bar_width, label='2018', color='skyblue')
        # Plot for 2023
        axs[i].bar([p + bar_width for p in x], pivot_data[2023], width=bar_width, label='2023', color='lightgreen')

        # Formatting the subplot
        axs[i].set_xlabel('Real Estate Type', fontsize=12)
        axs[i].set_ylabel('Percentage (%)', fontsize=12)
        axs[i].set_title(f'Percentage of Real Estate Types for Age Group {age_group} (2018 vs 2023)', fontsize=14)
        axs[i].set_xticks([p + bar_width / 2 for p in x])
        axs[i].set_xticklabels(pivot_data.index, rotation=45)
        axs[i].legend(title='Year')

# Remove any empty subplots if the number of age groups is odd
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Step 5: Show the combined plot
plt.tight_layout()
plt.show()

#bars for 2017-2024

# Ensure the 'Age' column is already created
data_MBF['Age'] = 2024 - data_MBF['HuurderDateOfBirth']  # Calculate age if not already done

# Create a new DataFrame excluding rows where real estate type is 'Kamer' and age is <= 25
data_MBF_Without_Koten = data_MBF[~((data_MBF['MappedRealEstateType'] == 'Kamer') & (data_MBF['Age'] <= 25))]

# Ensure that the 'RentalPeriodFrom' column is in datetime format
data_MBF_Without_Koten['RentalPeriodFrom'] = pd.to_datetime(data_MBF_Without_Koten['RentalPeriodFrom'])

# Create the 'Year' column by extracting the year from 'RentalPeriodFrom'
data_MBF_Without_Koten['Year'] = data_MBF_Without_Koten['RentalPeriodFrom'].dt.year

# Step 1: Create age intervals for the tenants
# Define the new age intervals
bins = [18, 30, 44, 60, 100]  # Define the bin edges
labels = ['18-30', '31-44', '45-60', '60+']  # Define the corresponding labels

# Create the Age Group column using pd.cut()
data_MBF_Without_Koten['Age Group'] = pd.cut(data_MBF_Without_Koten['Age'], 
                                              bins=bins, 
                                              labels=labels, 
                                              right=True)

# Step 2: Filter data for the years 2018 to 2024 (excluding 2017)
years_to_include = range(2018, 2025)
data_filtered = data_MBF_Without_Koten[data_MBF_Without_Koten['Year'].isin(years_to_include)]

# Step 3: Group the data by Age Group and Real Estate Type for all years
grouped = data_filtered.groupby(['Age Group', 'MappedRealEstateType', 'Year']).size().reset_index(name='Count')

# Step 4: Calculate total counts per Age Group for each year
total_counts = grouped.groupby(['Age Group', 'Year'])['Count'].sum().reset_index(name='Total')

# Step 5: Merge total counts into the grouped data
merged = pd.merge(grouped, total_counts, on=['Age Group', 'Year'])

# Step 6: Calculate percentages
merged['Percentage'] = (merged['Count'] / merged['Total']) * 100

# Step 7: Filter out rows where 'MappedRealEstateType' is 'Unknown'
final_data_cleaned = merged[merged['MappedRealEstateType'] != 'Unknown']

# Percentage of real estate type per age group
# Step 8: Get unique real estate types
real_estate_types = final_data_cleaned['MappedRealEstateType'].unique()

# Step 9: Set up the number of rows and columns for subplots
num_types = len(real_estate_types)
cols = 2  # Number of columns in the subplot
rows = (num_types + cols - 1) // cols  # Calculate the number of rows needed

# Step 10: Create a figure with subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Step 11: Loop through each real estate type and create a bar chart
for i, real_estate in enumerate(real_estate_types):
    # Filter data for the current real estate type
    type_data = final_data_cleaned[final_data_cleaned['MappedRealEstateType'] == real_estate]
    
    # Check if there is data for this type
    if not type_data.empty:
        # Create a pivot table for percentages
        pivot_data = type_data.pivot_table(
            index='Age Group', 
            columns='Year', 
            values='Percentage', 
            fill_value=0
        )

        # Plot bars for all years
        bar_width = 0.1  # Decrease bar width to fit all years
        x = range(len(pivot_data))

        # Plot for each year
        for j, year in enumerate(years_to_include):
            if year in pivot_data.columns:
                axs[i].bar([p + j * bar_width for p in x], pivot_data[year], width=bar_width, label=str(year))

        # Formatting the subplot
        axs[i].set_xlabel('Age Group', fontsize=12)
        axs[i].set_ylabel('Percentage (%)', fontsize=12)
        axs[i].set_title(f'Percentage of {real_estate} by Age Group (2018-2024)', fontsize=14)
        axs[i].set_xticks([p + (len(years_to_include) - 1) * bar_width / 2 for p in x])
        axs[i].set_xticklabels(pivot_data.index, rotation=45)
        axs[i].legend(title='Year')

# Remove any empty subplots if the number of real estate types is odd
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Step 12: Show the combined plot
plt.tight_layout()
plt.show()

# Percentage of age group per real estate type
# Step 13: Get unique age groups
age_groups = final_data_cleaned['Age Group'].unique()

# Step 14: Set up the number of rows and columns for subplots
num_groups = len(age_groups)
cols = 2  # Number of columns in the subplot
rows = (num_groups + cols - 1) // cols  # Calculate the number of rows needed

# Step 15: Create a figure with subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Step 16: Loop through each age group and create a bar chart
for i, age_group in enumerate(age_groups):
    # Filter data for the current age group
    age_data = final_data_cleaned[final_data_cleaned['Age Group'] == age_group]
    
    # Check if there is data for this age group
    if not age_data.empty:
        # Create a pivot table for percentages
        pivot_data = age_data.pivot_table(
            index='MappedRealEstateType', 
            columns='Year', 
            values='Percentage', 
            fill_value=0
        )

        # Plot bars for all years
        bar_width = 0.1  # Decrease bar width to fit all years
        x = range(len(pivot_data))

        # Plot for each year
        for j, year in enumerate(years_to_include):
            if year in pivot_data.columns:
                axs[i].bar([p + j * bar_width for p in x], pivot_data[year], width=bar_width, label=str(year))

        # Formatting the subplot
        axs[i].set_xlabel('Real Estate Type', fontsize=12)
        axs[i].set_ylabel('Percentage (%)', fontsize=12)
        axs[i].set_title(f'Percentage of Real Estate Types for Age Group {age_group} (2018-2024)', fontsize=14)
        axs[i].set_xticks([p + (len(years_to_include) - 1) * bar_width / 2 for p in x])
        axs[i].set_xticklabels(pivot_data.index, rotation=45)
        axs[i].legend(title='Year')

# Remove any empty subplots if the number of age groups is odd
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Step 17: Show the combined plot
plt.tight_layout()
plt.show()

# gemiddelde leeftijd huurders appartementen


# Stel dat de 'data_MBF' DataFrame al bestaat en correct is voorbereid.
# Zorg ervoor dat de 'RentalPeriodFrom' kolom in datetime-formaat is.
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Maak een nieuwe DataFrame die alleen rijen bevat waar het vastgoedtype 'Appartement' is.
apartments_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Stap 1: Bereken de leeftijd op het moment van contractondertekening
apartments_data['AgeAtSigning'] = apartments_data['RentalPeriodFrom'].dt.year - apartments_data['HuurderDateOfBirth']

# Stap 2: Maak een nieuwe kolom voor het jaar van contractondertekening
apartments_data['Year'] = apartments_data['RentalPeriodFrom'].dt.year

# Stap 3: Groepeer de gegevens per jaar en bereken de gemiddelde leeftijd
average_age_per_year = apartments_data.groupby('Year')['AgeAtSigning'].mean().reset_index()

# Stap 4: Plot de resultaten
plt.figure(figsize=(10, 6))
plt.plot(average_age_per_year['Year'], average_age_per_year['AgeAtSigning'], marker='o', color='skyblue')
plt.xlabel('Jaar', fontsize=12)
plt.ylabel('Gemiddelde Leeftijd (Jaren)', fontsize=12)
plt.title('Evolutie van de Gemiddelde Leeftijd van Huurders van Appartementen bij Contractondertekening', fontsize=14)
plt.xticks(average_age_per_year['Year'])  # Stel x-ticks in op de beschikbare jaren
plt.grid()
plt.tight_layout()
plt.show()

# postalcode gemiddelde leeftijd huurder appartement

# Assuming the 'data_MBF' DataFrame already exists and is prepared.
# Ensure that the 'RentalPeriodFrom' column is in datetime format.
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Create a new DataFrame that includes only rows where the real estate type is 'Appartement'.
apartments_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Step 1: Calculate the age at the moment of contract signing.
apartments_data['AgeAtSigning'] = apartments_data['RentalPeriodFrom'].dt.year - apartments_data['HuurderDateOfBirth']

# Step 2: Create a new column for the year of contract signing.
apartments_data['Year'] = apartments_data['RentalPeriodFrom'].dt.year

# Step 3: Group the data by postal code and year, calculating the average age.
average_age_per_postalcode = apartments_data.groupby(['location_postalCode', 'Year'])['AgeAtSigning'].mean().reset_index()

# Step 4: Plot the results for each postal code.
# Get unique postal codes to create separate plots
postal_codes = average_age_per_postalcode['location_postalCode'].unique()

# Set up the number of rows and columns for subplots
num_codes = len(postal_codes)
cols = 2  # Number of columns in the subplot
rows = (num_codes + cols - 1) // cols  # Calculate the number of rows needed

# Create a figure with subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Loop through each postal code and create a line plot
for i, postal_code in enumerate(postal_codes):
    # Filter data for the current postal code
    postal_code_data = average_age_per_postalcode[average_age_per_postalcode['location_postalCode'] == postal_code]
    
    # Plotting the average age evolution for this postal code
    axs[i].plot(postal_code_data['Year'], postal_code_data['AgeAtSigning'], marker='o', label=f'Postal Code: {postal_code}', color='skyblue')
    axs[i].set_xlabel('Year', fontsize=12)
    axs[i].set_ylabel('Average Age at Signing (Years)', fontsize=12)
    axs[i].set_title(f'Evolution of Average Age of Tenants at Signing for Postal Code: {postal_code}', fontsize=14)
    axs[i].grid()
    axs[i].legend()

# Remove any empty subplots if the number of postal codes is odd
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Show the combined plot
plt.tight_layout()
plt.show()

#median rentalprice increase per postalcode

# Stap 1: Zorg ervoor dat de RentalPeriodFrom kolom in datetime-formaat is
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Stap 2: Filter de DataFrame op data vanaf 2018 en voor Appartementen
filtered_data = data_MBF[
    (data_MBF['RentalPeriodFrom'] >= '2018-01-01') & 
    (data_MBF['MappedRealEstateType'] == 'Appartement')
]

# Stap 3: Extraheer het jaar uit de RentalPeriodFrom kolom
filtered_data['year'] = filtered_data['RentalPeriodFrom'].dt.year

# Stap 4: Groepeer de data op postcode en jaar, en bereken de mediane huurprijs
yearly_data = filtered_data.groupby(['location_postalCode', 'year'])['RentalFeeMonthly'].median().reset_index()

# Stap 5: Bereken de procentuele verandering ten opzichte van 2018 voor elke postcode
def calculate_percentage_change(group):
    # Controleer of er een baseline waarde voor 2018 is
    if 2018 in group['year'].values:
        baseline_value = group.loc[group['year'] == 2018, 'RentalFeeMonthly'].values[0]
        group['RentalFeeMonthlyChange'] = ((group['RentalFeeMonthly'] - baseline_value) / baseline_value) * 100
    else:
        # Als 2018 niet in de groep zit, vul met NaN
        group['RentalFeeMonthlyChange'] = None
    return group

# Toepassen van de functie op de groep
yearly_data = yearly_data.groupby('location_postalCode', as_index=False).apply(calculate_percentage_change)

# Filter voor groepen met een geldige baseline
yearly_data = yearly_data[yearly_data['RentalFeeMonthlyChange'].notnull()]

# Stap 6: Maak de lineplot voor elke postcode
plt.figure(figsize=(14, 8))
for postcode, group in yearly_data.groupby('location_postalCode'):
    plt.plot(group['year'], group['RentalFeeMonthlyChange'], marker='o', label=f'Postcode {postcode}')

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Baseline lijn op 0%
plt.title('Procentuele Evolutie van de Rental Fee Monthly voor Appartementen per Postcode (vanaf 2018)')
plt.xlabel('Jaar')
plt.ylabel('Procentuele Verandering (%)')
plt.grid()
plt.xticks(yearly_data['year'].unique())  # Zorg ervoor dat alle jaren op de x-as worden weergegeven
plt.legend(title='Postcodes', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legenda buiten de grafiek
plt.tight_layout()  # Zorg ervoor dat de plot er goed uitziet

plt.show()

#CPI index
# Stap 1: Zorg ervoor dat de 'Date' index in datetime-formaat is
CPI.index = pd.to_datetime(CPI.index)

# Stap 2: Filter de CPI-data voor de jaren 2018 tot en met 2024
CPI_filtered = CPI[(CPI.index.year >= 2018) & (CPI.index.year <= 2024)]

# Stap 3: Gebruik de waarde van de CPI in 2018 als baseline
cpi_baseline = CPI_filtered.loc[CPI_filtered.index.year == 2018, 'Value'].iloc[0]

# Stap 4: Bereken de procentuele verandering voor elk jaar t.o.v. 2018
CPI_filtered['Year'] = CPI_filtered.index.year  # Voeg een jaar kolom toe
CPI_changes = CPI_filtered.groupby('Year').first().reset_index()  # Neem de eerste waarde per jaar
CPI_changes['CPIChange'] = ((CPI_changes['Value'] - cpi_baseline) / cpi_baseline) * 100

# Stap 5: Maak de lineplot van de procentuele verandering
plt.figure(figsize=(10, 6))
plt.plot(CPI_changes['Year'], CPI_changes['CPIChange'], marker='o', color='blue', label='CPI Verandering (%)')

# Baseline lijn op 0%
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

# Titels en labels
plt.title('Procentuele Verandering van de CPI t.o.v. 2018 (voor elk jaar)')
plt.xlabel('Jaar')
plt.ylabel('Procentuele Verandering (%)')

# Grid en legenda
plt.grid()

# Toon alleen de jaren 2018 tot 2024 op de x-as
plt.xticks(CPI_changes['Year'])

plt.legend(title='CPI', loc='upper left')  # Legenda
plt.tight_layout()  # Zorg ervoor dat de plot er goed uitziet

# Plot tonen
plt.show()

# ---------------- CPI Verandering ----------------

# Stap 1: Zorg ervoor dat de 'Date' index in datetime-formaat is voor CPI
CPI.index = pd.to_datetime(CPI.index)

# Stap 2: Filter de CPI-data voor de jaren 2018 tot en met 2024
CPI_filtered = CPI[(CPI.index.year >= 2018) & (CPI.index.year <= 2024)]

# Stap 3: Gebruik de waarde van de CPI in 2018 als baseline
cpi_baseline = CPI_filtered.loc[CPI_filtered.index.year == 2018, 'Value'].iloc[0]

# Stap 4: Bereken de procentuele verandering voor elk jaar t.o.v. 2018
CPI_filtered['Year'] = CPI_filtered.index.year  # Voeg een jaar kolom toe
CPI_changes = CPI_filtered.groupby('Year').first().reset_index()  # Neem de eerste waarde per jaar
CPI_changes['CPIChange'] = ((CPI_changes['Value'] - cpi_baseline) / cpi_baseline) * 100

# ---------------- Huurprijs Verandering ----------------

# Stap 1: Zorg ervoor dat de RentalPeriodFrom kolom in datetime-formaat is
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Stap 2: Filter de DataFrame op data vanaf 2018 en voor Appartementen
filtered_data = data_MBF[
    (data_MBF['RentalPeriodFrom'] >= '2018-01-01') & 
    (data_MBF['MappedRealEstateType'] == 'Appartement')
]

# Stap 3: Extraheer het jaar uit de RentalPeriodFrom kolom
filtered_data['year'] = filtered_data['RentalPeriodFrom'].dt.year

# Stap 4: Groepeer de data op postcode en jaar, en bereken de mediane huurprijs
yearly_data = filtered_data.groupby(['location_postalCode', 'year'])['RentalFeeMonthly'].median().reset_index()

# Stap 5: Bereken de procentuele verandering ten opzichte van 2018 voor elke postcode
def calculate_percentage_change(group):
    # Controleer of er een baseline waarde voor 2018 is
    if 2018 in group['year'].values:
        baseline_value = group.loc[group['year'] == 2018, 'RentalFeeMonthly'].values[0]
        group['RentalFeeMonthlyChange'] = ((group['RentalFeeMonthly'] - baseline_value) / baseline_value) * 100
    else:
        # Als 2018 niet in de groep zit, vul met NaN
        group['RentalFeeMonthlyChange'] = None
    return group

# Toepassen van de functie op de groep
yearly_data = yearly_data.groupby('location_postalCode', as_index=False).apply(calculate_percentage_change)

# Filter voor groepen met een geldige baseline
yearly_data = yearly_data[yearly_data['RentalFeeMonthlyChange'].notnull()]

# ---------------- Plotten van beide grafieken samen ----------------

plt.figure(figsize=(14, 8))

# Plot de CPI-veranderingen (met dikkere lijn en vetgedrukte lijn)
plt.plot(CPI_changes['Year'], CPI_changes['CPIChange'], marker='o', color='blue', label='CPI Verandering (%)', linewidth=3)

# Plot de huurprijsveranderingen per postcode
for postcode, group in yearly_data.groupby('location_postalCode'):
    plt.plot(group['year'], group['RentalFeeMonthlyChange'], marker='o', label=f'Postcode {postcode}')

# Baseline lijn op 0%
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

# Titels en labels
plt.title('Procentuele Verandering van de CPI en Huurprijzen voor Appartementen (vanaf 2018)')
plt.xlabel('Jaar')
plt.ylabel('Procentuele Verandering (%)')

# X-as met alleen jaren
plt.xticks(CPI_changes['Year'].unique())  # Zorg ervoor dat alleen jaren worden getoond op de x-as

# Grid en legenda
plt.grid()

# Legenda aanpassen om CPI in vetgedrukt te laten zien
plt.legend(title='Postcodes en CPI', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()  # Zorg ervoor dat de plot er goed uitziet

# Plot tonen
plt.show()
