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
CPI = pd.read_csv('./Assignment 1/MBF_PythonBootcamp_Task1_CPI 3.csv', index_col='Date', parse_dates=True)

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
