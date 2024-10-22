# %% [markdown]
# # intro MBF task 1
# 
# 
# #### as usual load the standard modules
# #### note, I am using Python 3.10.4 mostly other version will work as well
# #### -> not every package works on every version! -> creating venv is a best practice

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # To load the background image
# %% [markdown]
# # load in the data
# 
# #### obviously those filepaths are custom to your own device

# %%
data_MBF = pd.read_csv('./MBF_FINAL_2.csv')

# Read the CPI data, setting 'Date' as the index and parsing it as datetime
CPI = pd.read_csv('./MBF_PythonBootcamp_Task1_CPI.csv', index_col='Date', parse_dates=True)


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


######## indexed version for all years

# Ensure RentalPeriodTo and RentalPeriodFrom are in datetime format
data_MBF['RentalPeriodTo'] = pd.to_datetime(data_MBF['RentalPeriodTo'], errors='coerce')
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'], errors='coerce')

# Filter for 'Appartement'
apartments = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Only consider contracts with RentalPeriodFrom starting in 2018 or later
apartments = apartments[apartments['RentalPeriodFrom'] >= '2018-01-01']

# CPI DataFrame for 2018 to 2024
cpi_first_day = CPI_filtered.loc[CPI_filtered.index.isin(pd.date_range('2018-01-01', '2024-01-01', freq='YS'))]
cpi_first_day.reset_index(inplace=True)  # Reset index to access the year

# Step 1: Create a list of active contracts for each year (2018 to 2024)
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
active_contracts_per_year = {}

for year in years:
    # Define the end date for filtering
    year_end_date = pd.Timestamp(f'{year}-12-31')
    
    # Filter for contracts that have started before the end of the given year and are still active
    active_contracts = apartments[
        (apartments['RentalPeriodFrom'] <= year_end_date) &  # Contract started before or on Dec 31 of current year
        ((apartments['RentalPeriodTo'].isna()) | (apartments['RentalPeriodTo'] >= f'{year}-01-01'))  # Contract is still active
    ].copy()
    
    active_contracts_per_year[year] = active_contracts

# Step 2: Create a CPI changes DataFrame with percentage change between all year pairs from 2018 onwards
cpi_years = cpi_first_day['Year'].unique()

# Create an empty list to store the CPI changes
cpi_differences = []

# Iterate through all possible year pairs
for start_year in cpi_years:
    for end_year in cpi_years:
        if end_year > start_year:
            # Get the CPI values for the start and end year
            cpi_start = cpi_first_day.loc[cpi_first_day['Year'] == start_year, 'Value'].values[0]
            cpi_end = cpi_first_day.loc[cpi_first_day['Year'] == end_year, 'Value'].values[0]
            
            # Calculate the percentage change between the two years
            cpi_diff = ((cpi_end - cpi_start) / cpi_start) * 100
            
            # Store the result
            cpi_differences.append({
                'StartYear': start_year,
                'EndYear': end_year,
                'CPIChange%': cpi_diff
            })

# Convert the list to a DataFrame
cpi_diff_df = pd.DataFrame(cpi_differences)

# Step 3: Define a function to adjust rental fee based on CPI changes
def adjust_rental_fee(row, current_year):
    start_year = row['RentalPeriodFrom'].year
    
    # If we're adjusting for 2018, the adjusted fee is the same as the original fee
    if current_year == 2018:
        return row['RentalFeeMonthly']
    
    # Find the CPI change between the start year and the current year
    cpi_change = cpi_diff_df[
        (cpi_diff_df['StartYear'] == start_year) & 
        (cpi_diff_df['EndYear'] == current_year)
    ]['CPIChange%']

    # If no CPI change is available, return the original fee
    if cpi_change.empty:
        return row['RentalFeeMonthly']
    
    # Adjust the rental fee based on the CPI change
    adjusted_fee = row['RentalFeeMonthly'] * (1 + (cpi_change.values[0] / 100))
    return adjusted_fee

# Step 4: Create adjusted DataFrames for each year (2018 to 2024)
adjusted_dataframes = {}

for year in years:
    active_contracts = active_contracts_per_year[year].copy()
    
    # Adjust the rental fees based on CPI changes
    active_contracts[f'AdjustedRentalFee_{year}'] = active_contracts.apply(lambda row: adjust_rental_fee(row, year), axis=1)
    
    # Store the adjusted DataFrame for the year
    adjusted_dataframes[year] = active_contracts[['location_postalCode', 'RentalPeriodFrom', 'RentalPeriodTo', 'RentalFeeMonthly', f'AdjustedRentalFee_{year}']]

# Now each year has a corresponding DataFrame in `adjusted_dataframes`

#calculate the medians
# Create a new DataFrame to hold median adjusted rental fees per postal code per year
median_adjusted_rental_fees = {}

# Loop through each year to calculate the median adjusted rental fee per postal code
for year in years:
    # Get the adjusted DataFrame for the current year
    df = adjusted_dataframes[year]
    
    # Calculate the median adjusted rental fee per postal code
    median_per_postal_code = df.groupby('location_postalCode')[f'AdjustedRentalFee_{year}'].median().reset_index()
    
    # Rename the column to indicate the year
    median_per_postal_code.rename(columns={f'AdjustedRentalFee_{year}': f'MedianAdjustedRentalFee_{year}'}, inplace=True)
    
    # Add the results to the main summary dictionary
    median_adjusted_rental_fees[year] = median_per_postal_code

# Merge the yearly median dataframes into a single dataframe
median_overview = median_adjusted_rental_fees[years[0]]

for year in years[1:]:
    median_overview = pd.merge(median_overview, median_adjusted_rental_fees[year], on='location_postalCode', how='outer')

# Remove rows where location_postalCode is 9042 from median_overview
median_overview = median_overview[median_overview['location_postalCode'] != 9042]

##### PLOT THE INDEX DATA

# Calculate percentage change based on 2018 medians
for year in years[1:]:  # Skip 2018 since it's the base year
    median_overview[f'{year}'] = (
        (median_overview[f'MedianAdjustedRentalFee_{year}'] - median_overview['MedianAdjustedRentalFee_2018']) /
        median_overview['MedianAdjustedRentalFee_2018']
    ) * 100

# Add a column for 2018 percentage change, which is 0%
median_overview['2018'] = 0  # 0% change for the base year

# Prepare data for plotting
plot_data = median_overview.set_index('location_postalCode')[[  # Include 2018 in the data
    f'{year}' for year in years
]].T


# Plotting
# Define the distinct colors for the postal codes
colors = ['lime', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'red']  # Excluded brown, gray, black, white

# Load the background image
background_image = mpimg.imread('./Afbeelding3.png')  # Update path as needed

# Calculate dynamic y-limits
y_min = plot_data.min().min()  # Minimum value across all years and postal codes
y_max = plot_data.max().max()  # Maximum value across all years and postal codes
padding = 5  # Padding to give some space above and below

# Mapping postal codes to names d
postal_code_names = {
    '9000': '9000 Gent',
    '9040': '9040 Sint-Amandsberg',
    '9031': '9031 Drongen',
    '9052': '9052 Zwijnaarde',
    '9032': '9032 Wondelgem',
    '9051': '9051 Afsnee, Sint-Denijs-Westrem',
    '9030': '9030 Mariakerke',
    '9050': '9050 Gentbrugge, Ledeberg',
    '9041': '9041 Oostakker'
}

# Plotting
plt.figure(figsize=(12, 6))

# Set the background image with adjusted extent
plt.imshow(background_image, aspect='auto', extent=[-0.5, len(plot_data.index) - 0.5, y_min - padding, y_max + padding])  # Adjust extent based on y-limits

plt.ylim(y_min - padding, y_max + padding)  # Set dynamic y-limits with padding

# Plot each postal code with a distinct color
for idx, postal_code in enumerate(plot_data.columns):
    # Retrieve the postal code name from the dictionary, default to the code if not found
    postal_code_name = postal_code_names.get(str(postal_code), postal_code)
    
    plt.plot(plot_data.index, plot_data[postal_code], marker='o', label=f'{postal_code_name}', 
             color=colors[idx % len(colors)], linewidth=2, markersize=6)  # Cycle through colors

plt.title('Evolution of Median Monthly Rental Fees for Apartments per Postal Code (2018 as Base Year, CPI adjusted)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.axhline(0, color='lightblue', linestyle='--')  # Reference line at 0%

# Set x-ticks to show only years
plt.xticks(ticks=range(len(plot_data.index)), labels=plot_data.index, rotation=45)

# Updated legend with postal code names
plt.legend(title='Postal Codes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


