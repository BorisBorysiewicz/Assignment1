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
