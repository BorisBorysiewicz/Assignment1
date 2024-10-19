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
appartments_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Appartement']

# Calculate median rental prices
median_rental_prices = appartments_data.groupby('location_postalCode')['RentalFeeMonthly'].median().reset_index()

# Rename the columns for merging
median_rental_prices.rename(columns={'location_postalCode': 'postcode', 'RentalFeeMonthly': 'median_rent'}, inplace=True)

# Merge the median rental prices with the GeoDataFrame
gent_merged = gent_merged.merge(median_rental_prices, on='postcode', how='left')

# Fill NaN values with 0 or an appropriate value if necessary
gent_merged['median_rent'].fillna(0, inplace=True)  # Use a suitable fill value

# Create the map with the reversed colormap
fig, ax = plt.subplots(figsize=(12, 8))
cmap = colormaps['RdYlGn_r']  # Reverse the colormap
gent_merged.plot(column='median_rent', cmap=cmap, edgecolor='black', linewidth=0.3, ax=ax, legend=True)

plt.axis('off')  # Turn off the axis

# Annotate the map with wijk names
for x, y, label in zip(gent_merged.geometry.centroid.x, gent_merged.geometry.centroid.y, gent_merged['wijk']):
    ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black')

# Display the plot
plt.show()
