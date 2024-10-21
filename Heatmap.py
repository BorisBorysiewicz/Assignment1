import geopandas as gpd
import pandas as pd
from matplotlib import colormaps
import matplotlib.pyplot as plt

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

# Ensure RentalPeriodFrom is in datetime format
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'], errors='coerce')

# Extract the year from RentalPeriodFrom and create a new column
data_MBF['Year'] = data_MBF['RentalPeriodFrom'].dt.year

# Filter data for only apartments
appartments_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Filter data for the years 2018 and 2024
appartments_data_2018 = appartments_data[appartments_data['Year'] == 2018]
appartments_data_2024 = appartments_data[appartments_data['Year'] == 2024]

# Calculate median rental prices for each year
median_rental_prices_2018 = appartments_data_2018.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()
median_rental_prices_2024 = appartments_data_2024.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()

# Rename the columns for merging
median_rental_prices_2018.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent_2018'}, inplace=True)
median_rental_prices_2024.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent_2024'}, inplace=True)

# Merge the median rental prices with the GeoDataFrame for both years
gent_merged_2018 = gent_merged.merge(median_rental_prices_2018, on='postcode', how='left')
gent_merged_2024 = gent_merged.merge(median_rental_prices_2024, on='postcode', how='left')

# Fill NaN values with 0 or an appropriate value if necessary
gent_merged_2018['median_rent_2018'].fillna(0, inplace=True)  # Use a suitable fill value
gent_merged_2024['median_rent_2024'].fillna(0, inplace=True)  # Use a suitable fill value

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Reverse the colormap
cmap = colormaps['RdYlGn_r'] 

# Plot the heatmap for 2018
gent_merged_2018.plot(column='median_rent_2018', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax1, legend=True)
ax1.set_title("Median Rental Prices in 2018")
ax1.axis('off')  # Turn off the axis

# Annotate the map with wijk names for 2018
for x, y, label in zip(gent_merged_2018.geometry.centroid.x, gent_merged_2018.geometry.centroid.y, gent_merged_2018['wijk']):
    ax1.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

# Plot the heatmap for 2024
gent_merged_2024.plot(column='median_rent_2024', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax2, legend=True)
ax2.set_title("Median Rental Prices in 2024")
ax2.axis('off')  # Turn off the axis

# Annotate the map with wijk names for 2024
for x, y, label in zip(gent_merged_2024.geometry.centroid.x, gent_merged_2024.geometry.centroid.y, gent_merged_2024['wijk']):
    ax2.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

# Display the plots
plt.tight_layout()
plt.show()
