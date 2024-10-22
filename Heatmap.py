import geopandas as gpd
import pandas as pd
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Load the geographic data
gent = gpd.read_file("./stadswijken-gent.geojson")

# Update postcodes for neighborhoods that are not 9000
gent.loc[gent['wijk'] == "Wondelgem", 'postcode'] = 9032
gent.loc[gent['wijk'] == "Drongen", 'postcode'] = 9031
gent.loc[gent['wijk'] == "Sint-Amandsberg", 'postcode'] = 9040
gent.loc[gent['wijk'] == "Gentse Kanaaldorpen en -zone", 'postcode'] = 9042
gent.loc[gent['wijk'] == "Ledeberg", 'postcode'] = 9050
gent.loc[gent['wijk'] == "Sint-Denijs-Westrem - Afsnee", 'postcode'] = 9051
gent.loc[gent['wijk'] == "Oud Gentbrugge", 'postcode'] = 9050
gent.loc[gent['wijk'] == "Zwijnaarde", 'postcode'] = 9052
gent.loc[gent['wijk'] == "Gentbrugge", 'postcode'] = 9050
gent.loc[gent['wijk'] == "Moscou - Vogelhoek", 'postcode'] = 9050
gent.loc[gent['wijk'] == "Oostakker", 'postcode'] = 9041
gent.loc[gent['wijk'] == "Mariakerke", 'postcode'] = 9030

# Fill remaining postcodes with 9000
gent['postcode'].fillna(9000, inplace=True)

# Make sure 'postcode' is converted to integer
gent['postcode'] = gent['postcode'].astype(int)

# Merge geo data by postcode
gent_merged = gent.dissolve(by='postcode').reset_index()[["postcode", "geometry", "wijk"]]

# Create a mapping of postcode to wijk
postcode_to_wijk = {
    9000: "Gent",
    9042: "Desteldonk, Mendonk, Sint-Kruis-Winkel",
    9050: "Gentbrugge, Ledeberg",
    9051: "Afsnee, Sint-Denijs-Westrem",
}

for postcode, wijk in postcode_to_wijk.items():
    gent_merged.loc[gent_merged['postcode'] == postcode, 'wijk'] = wijk

# Load rental prices data
data_MBF = pd.read_csv('./MBF_FINAL_2.csv')

# Load the CPI data
CPI = pd.read_csv('./MBF_PythonBootcamp_Task1_CPI.csv', index_col='Date', parse_dates=True)


# Ensure RentalPeriodTo and RentalPeriodFrom are in datetime format
data_MBF['RentalPeriodTo'] = pd.to_datetime(data_MBF['RentalPeriodTo'], errors='coerce')
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'], errors='coerce')

# Filter for the year 2018
data_2018 = data_MBF[data_MBF['RentalPeriodFrom'].dt.year == 2018]
# + Filter for housetype 'Appartement' for the year 2018
apartments_2018 = data_2018[data_2018['MappedRealEstateType'] == 'Appartement']



#Active contracts in 2024
active_2024 = data_MBF[
    (data_MBF['RentalPeriodTo'].isna()) | (data_MBF['RentalPeriodTo'] >= '2024-01-01')
]

# Filter for housetype 'Appartement'
active_2024_apartments = active_2024[active_2024['MappedRealEstateType'] == 'Appartement']

#Filter CPI-data 
CPI_filtered = CPI[(CPI.index.year >= 2018) & (CPI.index.year <= 2024)]

#Add Year 
CPI_filtered['Year'] = CPI_filtered.index.year

# Extract CPI values for the first day of each year from 2018 to 2024
cpi_first_day = CPI_filtered.loc[CPI_filtered.index.isin(pd.date_range('2018-01-01', '2024-01-01', freq='YS'))]

# Reset index to work with years more easily
cpi_first_day.reset_index(inplace=True)

# Extract the years
cpi_years = cpi_first_day['Year'].unique()

# Create an empty list to store the results
cpi_differences = []

# Iterate through all possible year pairs
for start_year in cpi_years:
    for end_year in cpi_years:
        if end_year > start_year:
            # Get the CPI values for the first day of the start and end years
            cpi_start = cpi_first_day.loc[cpi_first_day['Year'] == start_year, 'Value'].values[0]
            cpi_end = cpi_first_day.loc[cpi_first_day['Year'] == end_year, 'Value'].values[0]
            
            # Calculate the percentage change between the two years
            cpi_diff = ((cpi_end - cpi_start) / cpi_start) * 100
            
            # Append the result to the list
            cpi_differences.append({
                'StartYear': start_year,
                'EndYear': end_year,
                'CPIChange%': cpi_diff
            })

# Convert the list to a DataFrame
cpi_diff_df = pd.DataFrame(cpi_differences)


# Exclude rows where RentalPeriodFrom is before 2018
active_2024_apartments = active_2024_apartments[active_2024_apartments['RentalPeriodFrom'] >= '2018-01-01']

# Define the adjust_rental_fee function to adjust RentalFeeMonthly based on CPI changes
def adjust_rental_fee(row):
    start_year = row['RentalPeriodFrom'].year
    end_year = 2024  # We're adjusting for 2024

    # Find the CPI change between the start year and 2024
    cpi_change = cpi_diff_df[
        (cpi_diff_df['StartYear'] == start_year) & 
        (cpi_diff_df['EndYear'] == end_year)
    ]['CPIChange%']

    # If there's no CPI change available for the year, keep the rental fee as is
    if cpi_change.empty:
        return row['RentalFeeMonthly']
    
    # Adjust the rental fee based on the CPI change
    adjusted_fee = row['RentalFeeMonthly'] * (1 + (cpi_change.values[0] / 100))
    return adjusted_fee

# Apply the adjustment to RentalFeeMonthly
active_2024_apartments['AdjustedRentalFeeMonthly'] = active_2024_apartments.apply(adjust_rental_fee, axis=1)


# Final DataFrame with only the relevant columns
final_df = active_2024_apartments[['location_postalCode', 'RentalPeriodFrom', 'RentalPeriodTo', 'RentalFeeMonthly', 'AdjustedRentalFeeMonthly']]



# Calculate median rental prices for each year
median_rental_prices_2018 = apartments_2018.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()
median_rental_prices_2024 = final_df.groupby('location_postalCode')['AdjustedRentalFeeMonthly'].median().reset_index()

# Rename the columns for merging
median_rental_prices_2018.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent_2018'}, inplace=True)
median_rental_prices_2024.rename(columns={'location_postalCode': 'postcode', 'AdjustedRentalFeeMonthly': 'median_rent_2024'}, inplace=True)

# Merge the median rental prices with the GeoDataFrame for both years
gent_merged_2018 = gent_merged.merge(median_rental_prices_2018, on='postcode', how='left')
gent_merged_2024 = gent_merged.merge(median_rental_prices_2024, on='postcode', how='left')

# Set the median_rent_2024 to NaN for postcode 9042 to create a grey zone
gent_merged_2024.loc[gent_merged_2024['postcode'] == 9042, 'median_rent_2024'] = float('nan')

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Reverse the colormap
cmap = colormaps['RdYlGn_r']

# Define the color for NaN (grey)
missing_color = to_rgba('lightgrey')

# Plot the heatmap for 2018
gent_merged_2018.plot(column='median_rent_2018', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax1, legend=True, missing_kwds={'color': missing_color})
ax1.set_title("Median Monthly Rental Fees of Apartments in 2018 (Base Year)")
ax1.axis('off')  # Turn off the axis

# Annotate the map with wijk names and median rent for 2018
for x, y, label, rent in zip(gent_merged_2018.geometry.centroid.x, gent_merged_2018.geometry.centroid.y, gent_merged_2018['wijk'], gent_merged_2018['median_rent_2018']):
    if pd.notna(rent):
        ax1.text(x, y, f'{label}\n€{int(rent)}', fontsize=10, ha='center', va='center', color='black')
    else:
        ax1.text(x, y, label, fontsize=10, ha='center', va='center', color='black')  # Show label even for missing data

# Plot the heatmap for 2024
gent_merged_2024.plot(column='median_rent_2024', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax2, legend=True, missing_kwds={'color': missing_color})
ax2.set_title("Median Monthly Rental Fees of Apartments in 2024 (CPI Adjusted)")
ax2.axis('off')  # Turn off the axis

# Annotate the map with wijk names and median rent for 2024
for x, y, label, rent in zip(gent_merged_2024.geometry.centroid.x, gent_merged_2024.geometry.centroid.y, gent_merged_2024['wijk'], gent_merged_2024['median_rent_2024']):
    if pd.notna(rent):
        ax2.text(x, y, f'{label}\n€{int(rent)}', fontsize=10, ha='center', va='center', color='black')
    else:
        ax2.text(x, y, label, fontsize=10, ha='center', va='center', color='black')  # Show label even for missing data

# Add a main title to the figure
fig.suptitle("Comparison of Median Monthly Rental Fees of Apartments in Ghent (2018 vs. 2024)", fontsize=16)

# Display the plots
plt.tight_layout()
plt.show()

