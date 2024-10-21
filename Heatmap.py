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

# Ensure RentalPeriodFrom and RentalPeriodTo are in datetime format
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])
data_MBF['RentalPeriodTo'] = pd.to_datetime(data_MBF['RentalPeriodTo'])

# Filter data for only apartments
appartments_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Calculate active contracts for 2018
active_contracts_2018 = appartments_data[
    (appartments_data['RentalPeriodFrom'] <= pd.Timestamp(year=2018, month=12, day=31)) & 
    (appartments_data['RentalPeriodTo'] >= pd.Timestamp(year=2018, month=1, day=1))
]

# Calculate active contracts for 2024
active_contracts_2024 = appartments_data[
    (appartments_data['RentalPeriodFrom'] <= pd.Timestamp(year=2024, month=12, day=31)) & 
    (appartments_data['RentalPeriodTo'] >= pd.Timestamp(year=2024, month=1, day=1))
]

# Calculate median rental prices based on active contracts
median_rental_prices_2018 = active_contracts_2018.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()
median_rental_prices_2024 = active_contracts_2024.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()

# Rename the columns for merging
median_rental_prices_2018.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent_2018'}, inplace=True)
median_rental_prices_2024.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent_2024'}, inplace=True)

# Merge the median rental prices with the GeoDataFrame for both years
gent_merged_2018 = gent_merged.merge(median_rental_prices_2018, on='postcode', how='left')
gent_merged_2024 = gent_merged.merge(median_rental_prices_2024, on='postcode', how='left')

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Reverse the colormap
cmap = colormaps['RdYlGn_r']

# Define the color for NaN (grey)
missing_color = to_rgba('lightgrey')

# Plot the heatmap for 2018
gent_merged_2018.plot(column='median_rent_2018', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax1, legend=True, missing_kwds={'color': missing_color})
ax1.set_title("Median Rental Prices in 2018")
ax1.axis('off')  # Turn off the axis

# Annotate the map with wijk names and median rent for 2018
for x, y, label, rent in zip(gent_merged_2018.geometry.centroid.x, gent_merged_2018.geometry.centroid.y, gent_merged_2018['wijk'], gent_merged_2018['median_rent_2018']):
    if pd.notna(rent):
        ax1.text(x, y, f'{label}\n€{int(rent)}', fontsize=10, ha='center', va='center', color='black')
    else:
        ax1.text(x, y, label, fontsize=10, ha='center', va='center', color='black')  # Show label even for missing data

# Plot the heatmap for 2024
gent_merged_2024.plot(column='median_rent_2024', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax2, legend=True, missing_kwds={'color': missing_color})
ax2.set_title("Median Rental Prices in 2024")
ax2.axis('off')  # Turn off the axis

# Annotate the map with wijk names and median rent for 2024
for x, y, label, rent in zip(gent_merged_2024.geometry.centroid.x, gent_merged_2024.geometry.centroid.y, gent_merged_2024['wijk'], gent_merged_2024['median_rent_2024']):
    if pd.notna(rent):
        ax2.text(x, y, f'{label}\n€{int(rent)}', fontsize=10, ha='center', va='center', color='black')
    else:
        ax2.text(x, y, label, fontsize=10, ha='center', va='center', color='black')  # Show label even for missing data

# Add a main title to the figure
fig.suptitle("Median Rental Prices Ghent", fontsize=16)

# Display the plots
plt.tight_layout()
plt.show()






