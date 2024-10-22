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
data_MBF = pd.read_csv('./Assignment 1/MBF_FINAL_2.csv')

# Read the CPI data, setting 'Date' as the index and parsing it as datetime
CPI = pd.read_csv('./Assignment 1/MBF_PythonBootcamp_Task1_CPI 3.csv', index_col='Date', parse_dates=True)


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

# Load the background image
background_image = mpimg.imread('./Assignment 1/Project/Assignment1/Afbeelding3.png')  # Update path as needed

# Use a set of distinct colors for postal codes
colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']  # List of contrasting colors

# Create a figure and axes
fig, ax = plt.subplots(figsize=(14, 8))

# Set the image as the background
ax.imshow(background_image, aspect='auto', extent=[CPI_changes['Year'].min(), CPI_changes['Year'].max(), 
                                                    min(yearly_data['RentalFeeMonthlyChange']) - 5, 
                                                    max(yearly_data['RentalFeeMonthlyChange']) + 5], 
          zorder=-1)

# Plot de huurprijsveranderingen per postcode with distinct colors
for i, (postcode, group) in enumerate(yearly_data.groupby('location_postalCode')):
    plt.plot(group['year'], 
             group['RentalFeeMonthlyChange'], 
             marker='s',  # Square markers for differentiation
             color=colors[i % len(colors)],  # Use a distinct color from the list
             label=f'Postcode {postcode}', 
             linewidth=4,  # Increased linewidth for better visibility
             markersize=6,  # Larger markers
             linestyle='-',  # Solid line style for postal codes
             alpha=0.85)  # Slight transparency

# Plot de CPI-veranderingen with a big very flashy neon green line
plt.plot(CPI_changes['Year'], 
         CPI_changes['CPIChange'], 
         marker='o',  # Circle markers for enhanced visibility
         color='lime',  # Flashy neon green color for CPI line
         label='CPI Verandering (%)', 
         linewidth=8,  # Increased line width for maximum visibility
         markersize=10,  # Larger markers
         linestyle='-',  # Solid line style
         alpha=1.0,  # Full opacity
         zorder=2)  # Ensure this line is above all others

# Titles and labels with larger font size
plt.title('Procentuele verandering van de mediane huurpijs voor appartementen in Gent (vanaf 2018)', fontsize=16, fontweight='bold')  # Increased fontsize
plt.xlabel('Jaar', fontsize=14)  # Increased fontsize
plt.ylabel('Procentuele Verandering (%)', fontsize=14)  # Increased fontsize

# X-axis with only years
plt.xticks(CPI_changes['Year'].unique(), fontsize=16)  # Increased fontsize for x-ticks (years)
plt.yticks(fontsize=16)  # Increased fontsize for y-ticks (percentages)

# Grid and legend with larger font size
plt.grid(color='white', linestyle='--', linewidth=0.5)  # White grid lines for better visibility
plt.legend(title='Postcodes en CPI', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)  # Increased fontsize for legend

# Remove whitespace around the plot
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Show the plot
plt.show()

################## Indexed version
# Step 1: Filter active contracts in 2024 (RentalPeriodTo is >= 2024 or NaT)
# Ensure RentalPeriodTo is in datetime format
data_MBF['RentalPeriodTo'] = pd.to_datetime(data_MBF['RentalPeriodTo'], errors='coerce')

# Filter for active contracts in 2024 (either ongoing or ending in 2024 or later)
active_2024 = data_MBF[
    (data_MBF['RentalPeriodTo'].isna()) | (data_MBF['RentalPeriodTo'] >= '2024-01-01')
]
# Filter for 'Appartement'
active_2024_apartments = active_2024[active_2024['MappedRealEstateType'] == 'Appartement']

# Step 2: Create a CPI changes DataFrame with percentage change between all year pairs from 2018 onwards

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

# Ensure RentalPeriodFrom is in datetime format
active_2024_apartments['RentalPeriodFrom'] = pd.to_datetime(active_2024_apartments['RentalPeriodFrom'], errors='coerce')

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

# Display the adjusted DataFrame
active_2024_apartments.head()

# Final DataFrame with only the relevant columns
final_df = active_2024_apartments[['location_postalCode', 'RentalPeriodFrom', 'RentalPeriodTo', 'RentalFeeMonthly', 'AdjustedRentalFeeMonthly']]

