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

# %%
data_MBF.head()

# %%
data_MBF.describe()

# %% [markdown]
# ### what type or real estate data are we dealing with?

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Function to create the plots for the provided dataset
def plot_distribution_and_contracts(data, title_suffix, cap_365_days=False):
    # Create subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ---------------- First Plot (ax1): Distribution of MappedRealEstateType ----------------
    data['MappedRealEstateType'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xlabel('MappedRealEstateType')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of MappedRealEstateType (Normal Scale) - {title_suffix}')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # ---------------- Second Plot (ax2): Count of Active Contracts by Real Estate Type ----------------
    # Generate the date range from 2016-01-01 to 2024-12-31
    date_range = pd.date_range(start='2016-01-01', end='2024-12-31')

    # Initialize DataFrame to store the counts for each day and each subgroup
    real_estate_types = ['Rijhuis', 'Appartement', 'Kamer', 'Studio', 'Halfopen bebouwing']
    daily_counts_cleaned = pd.DataFrame(0, index=date_range, columns=real_estate_types)

    # Calculate the count of active contracts for each day and each subgroup
    for real_estate_type in daily_counts_cleaned.columns:
        subset = data[data['MappedRealEstateType'] == real_estate_type]
        for idx, row in subset.iterrows():
            # Ensure 'RentalPeriodFrom' and 'RentalPeriodTo' are datetime objects
            start_date = pd.to_datetime(row['RentalPeriodFrom'])
            end_date = pd.to_datetime(row['RentalPeriodTo'])

            # Cap the date range to 365 days if cap_365_days is True
            if cap_365_days:
                end_date = min(end_date, start_date + pd.Timedelta(days=365))

            # Increment the counts in the cleaned daily count DataFrame
            daily_counts_cleaned.loc[start_date:end_date, real_estate_type] += 1

    # Plot the data as a stacked bar chart on ax2
    daily_counts_cleaned.plot(kind='bar', stacked=True, ax=ax2, width=1.0, alpha=0.8, legend=True)
    ax2.set_title(f'Count of Active Contracts Per Day by Real Estate Type (2016-2024) - {title_suffix}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Count of Active Contracts')

    # Set x-axis to show the first day of January and July every year
    first_jan_june_dates = pd.date_range(start='2016-01-01', end='2024-12-31', freq='6MS')
    ax2.set_xticks([daily_counts_cleaned.index.get_loc(date) for date in first_jan_june_dates])
    ax2.set_xticklabels(first_jan_june_dates.strftime('%Y-%m-%d'), rotation=45)

    # Add text annotation
    ax2.text(0.5, -0.3, 
             "Note the slow buildup of data \n This is real world data from the start of data gathering" 
             "\n Can we even draw sensible conclusions in 2016?", 
             ha="center", fontsize=10, transform=ax2.transAxes)

    # Adjust layout to ensure everything fits without overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    # Return the cleaned daily counts DataFrame
    return 

# Example usage:
# Call the function with a cap of 365 days
plot_distribution_and_contracts(data_MBF, 'All Data', cap_365_days=True)

plot_distribution_and_contracts(data_MBF[data_MBF.source == "A"], 'source A', cap_365_days=True)

plot_distribution_and_contracts(data_MBF[data_MBF.source == "B"], 'source B', cap_365_days=True)

# %% [markdown]
# ### what is the location of those properties?

# %%
# Get counts of each unique value
counts = data_MBF['Borough'].value_counts()

# Restrict to values that have at least 100 occurrences
counts_filtered = counts[counts >= 50]

# Plot bar chart of counts
plt.figure(figsize=(12, 6))
counts_filtered.plot(kind='bar', edgecolor='black')

plt.xlabel('Borough greater Ghent')
plt.ylabel('Number of Occurrences')
plt.title('Histogram of Occurrence Counts in Boroughs (Counts >= 50)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### we even have data from the land register (Cadaster in Flemish)

# %%
# -------------------- First Plot: Median Square Meters Ground Surface for All Boroughs --------------------
# Get counts of each unique Borough
counts = data_MBF['Borough'].value_counts()

# Restrict to Boroughs that have at least 50 occurrences
valid_Boroughs = counts[counts >= 50].index

# Filter the data to include only the Boroughs with at least 50 occurrences
filtered_data = data_MBF[data_MBF['Borough'].isin(valid_Boroughs)]

# Calculate the median of SquareMetersGroundSurfaceBuildingLot for each valid Borough
median_square_meters = filtered_data.groupby('Borough')['SquareMetersGroundSurfaceBuildingLot'].median()

# Sort the median values in descending order
median_square_meters = median_square_meters.sort_values(ascending=False)

# Plot the median SquareMetersGroundSurfaceBuildingLot for each Borough
plt.figure(figsize=(12, 6))
median_square_meters.plot(kind='bar', edgecolor='black')

plt.xlabel('Borough (Greater Ghent)')
plt.ylabel('Median Square Meters Ground Surface of Building Lot')
plt.title('Median Square Meters Ground Surface by Borough (Occurrences >= 50)')
plt.xticks(rotation=90)
plt.tight_layout()

# Show the plot
plt.show()

#This seems odd, these properties are huge.

# -------------------- Second Plot: Median and Mean for 'Rijhuis' --------------------
# Filter the data for MappedRealEstateType == 'Rijhuis'
rijhuis_data = data_MBF[data_MBF['MappedRealEstateType'] == 'Rijhuis']

# Get counts of each unique Borough for 'Rijhuis'
counts = rijhuis_data['Borough'].value_counts()

# Restrict to Boroughs that have at least 50 occurrences
valid_Boroughs = counts[counts >= 50].index

# Filter the data to include only the Boroughs with at least 50 occurrences
filtered_data = rijhuis_data[rijhuis_data['Borough'].isin(valid_Boroughs)]

# Calculate the median and mean of SquareMetersGroundSurfaceBuildingLot for each valid Borough
median_square_meters = filtered_data.groupby('Borough')['SquareMetersGroundSurfaceBuildingLot'].median()
mean_square_meters = filtered_data.groupby('Borough')['SquareMetersGroundSurfaceBuildingLot'].mean()

# Sort the Boroughs based on the median values in descending order
median_square_meters = median_square_meters.sort_values(ascending=False)
mean_square_meters = mean_square_meters.loc[median_square_meters.index]  # Sort mean in the same order as median

# Plot the median and mean SquareMetersGroundSurfaceBuildingLot side by side
fig, ax = plt.subplots(figsize=(12, 6))

# Plot median
ax.bar(median_square_meters.index, median_square_meters, width=0.4, label='Median', align='center', edgecolor='black')

# Plot mean, offset by a little to appear next to the median
ax.bar(mean_square_meters.index, mean_square_meters, width=0.4, label='Mean', align='edge', edgecolor='black', color='orange')

# Add labels and title
plt.xlabel('Borough (Greater Ghent)')
plt.ylabel('Square Meters Ground Surface of Building Lot')
plt.title("Median and Mean Square Meters Ground Surface for 'Rijhuis' by Borough (Occurrences >= 50)")
plt.xticks(rotation=90)

# Add legend
plt.legend()

# Ensure layout is not overlapping
plt.tight_layout()

# Show the plot
plt.show()

# Write the comment below the second plot
# The message should be clear, make sure you understand each variable! Always reflect on the results you get and ask yourself if this is realistic or if something seems off.

# %% [markdown]
# ### We have price data on time points -> we are dealing with a time series 

# %%
# Ensure the 'RentalPeriodFrom' column is in datetime format
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Create a figure with 1 row and 2 columns of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (all data)
ax1.scatter(data_MBF['RentalPeriodFrom'], data_MBF['RentalFeeMonthly'], s=1)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Set major ticks every 12 months
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format x-axis as 'YYYY-MM'
ax1.set_xlabel('Rental Period From')
ax1.set_ylabel('Rental Fee Monthly')
ax1.set_title('Time Series of Monthly Rental Fee')
ax1.tick_params(axis='x', rotation=45)
ax1.text(0.5, -0.3, "Clearly some outliers distort the data", ha="center", fontsize=10, transform=ax1.transAxes)

# Second subplot (Histogram of the first plot's data)
ax2.hist(data_MBF['RentalFeeMonthly'], bins=100, color='orange', edgecolor='black')  # Set a large number of bins
ax2.set_xlabel('Rental Fee Monthly')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Monthly Rental Fee (Large Bins)')
ax2.text(0.5, -0.3, "Distribution of Rental Fees \nHow common are extreme values?", 
         ha="center", fontsize=10, transform=ax2.transAxes)

# Adjust layout to prevent overlap and improve spacing
plt.tight_layout()

# Show the plots
plt.show()

# let's apply a basic notion of outlier removal.

# %% 
# Ensure the 'RentalPeriodFrom' column is in datetime format
data_MBF['RentalPeriodFrom'] = pd.to_datetime(data_MBF['RentalPeriodFrom'])

# Define the quantile for outlier removal (0.1% and 99.9%)
lower_quantile = data_MBF['RentalFeeMonthly'].quantile(0.001)
upper_quantile = data_MBF['RentalFeeMonthly'].quantile(0.999)

# Filter the data to remove outliers (keeping only the middle 98%)
filtered_data = data_MBF[(data_MBF['RentalFeeMonthly'] >= lower_quantile) & 
                         (data_MBF['RentalFeeMonthly'] <= upper_quantile)]

# Create a figure with 1 row and 2 columns of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (filtered data)
ax1.scatter(filtered_data['RentalPeriodFrom'], filtered_data['RentalFeeMonthly'], s=1)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Set major ticks every 12 months
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format x-axis as 'YYYY-MM'
ax1.set_xlabel('Rental Period From')
ax1.set_ylabel('Rental Fee Monthly')
ax1.set_title('Time Series of Monthly Rental Fee (Outliers Removed)')
ax1.tick_params(axis='x', rotation=45)
ax1.text(0.5, -0.3, "Outliers removed, data looks cleaner", ha="center", fontsize=10, transform=ax1.transAxes)

# Second subplot (Histogram of the filtered data)
ax2.hist(filtered_data['RentalFeeMonthly'], bins=100, color='orange', edgecolor='black')  # Set a large number of bins
ax2.set_xlabel('Rental Fee Monthly')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Monthly Rental Fee (Outliers Removed)')
ax2.text(0.5, -0.3, "Distribution of Rental Fees (After Removing Outliers)", 
         ha="center", fontsize=10, transform=ax2.transAxes)

# Adjust layout to prevent overlap and improve spacing
plt.tight_layout()

# Show the plots
plt.show()

# clearly this is not perfect, yet we have only removed O.2% of the data
# rental fees exhibit a fat tail, this is to be expected
# this should provide with a refelex to not try to fit a normal distribution on every sort of data!

# %% [markdown]
# #### note, we have applied one of the most basic outlier detection and removal strategies
# ##### there exist a whole zoo of methods to deal with outliers
# ##### it is mostly up to you to decide what to use when, and reflect on why a certain method is applicable
# ###### again: don't underestimate having a close deep look at your data
# ###### you can sometimes remove easily identifiable errors without relying on assumptions or statistics 

# %% [markdown]
# ### let's try to make a basic index for rental prices in Gent
# 
# #### note the interpretability between both graphs

# %%
# Filter the rental data to have 'RentalPeriodFrom' between 2017 and 2024
filtered_data.set_index('RentalPeriodFrom', inplace=True)
filtered_data_timespan = filtered_data[(filtered_data.index >= '2017-01-01') & (filtered_data.index < '2024-07-01')]

# Resample the filtered data monthly and calculate the mean, median, and count of RentalFeeMonthly
monthly_stats = filtered_data_timespan['RentalFeeMonthly'].resample('M').agg(['mean', 'median', 'count'])

# Normalize rental fee data by dividing by the first value and multiplying by 100
monthly_stats['mean_normalized'] = (monthly_stats['mean'] / monthly_stats['mean'].iloc[0]) * 100
monthly_stats['median_normalized'] = (monthly_stats['median'] / monthly_stats['median'].iloc[0]) * 100

# Filter the CPI data for the same time period
filtered_CPI = CPI[(CPI.index >= '2017-01-01') & (CPI.index < '2024-07-01')]

# Normalize CPI data by dividing by the first value and multiplying by 100
filtered_CPI['Value_normalized'] = (filtered_CPI['Value'] / filtered_CPI['Value'].iloc[0]) * 100

# Create subplots: 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --------------- First Plot: Two Y-Axes (Rental Fee and CPI) ---------------
# Plot mean and median over time on the primary y-axis
ax1.plot(monthly_stats.index, monthly_stats['mean'], label='Mean Rental Fee', color='blue', linewidth=2)
ax1.plot(monthly_stats.index, monthly_stats['median'], label='Median Rental Fee', color='green', linestyle='--', linewidth=2)

# Set labels for the primary y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Rental Fee Monthly')
ax1.set_title('Mean and Median Rental Fee vs CPI (2017-2025)')
ax1.legend(loc='upper left')

# Rotate x-axis labels for readability
ax1.tick_params(axis='x', rotation=45)

# Create a second y-axis to plot CPI data
ax3 = ax1.twinx()
ax3.plot(filtered_CPI.index, filtered_CPI['Value'], label='CPI', color='orange', linewidth=2)

# Set label for the secondary y-axis
ax3.set_ylabel('CPI Value')
ax3.legend(loc='upper right')

# --------------- Second Plot: Normalized Data and Contract Count (Rental Fee and CPI) ---------------
# Plot normalized mean and median over time
ax2.plot(monthly_stats.index, monthly_stats['mean_normalized'], label='Mean Rental Fee (Normalized)', color='blue', linewidth=2)

# Plot the normalized CPI values
ax2.plot(filtered_CPI.index, filtered_CPI['Value_normalized'], label='CPI (Normalized)', color='orange', linewidth=2)

# Set labels for the second plot
ax2.set_xlabel('Date')
ax2.set_ylabel('Normalized Value (Base = 100)')
ax2.set_title('Normalized Rental Fee vs CPI (2017-2025)')
ax2.legend(loc='upper left')

# Create a second y-axis to plot the contract count
ax4 = ax2.twinx()
ax4.plot(monthly_stats.index, monthly_stats['count'], label='Contract Count', color='red', linestyle=':', linewidth=2)

# Set label for the secondary y-axis (contract count)
ax4.set_ylabel('Number of Contracts')
ax4.legend(loc='upper right')

ax2.text(0.5, -0.3, "Easier to compare the evolution of CPI with that of the Rental rates \n yet the rental index drops as the number of contracts spikes!", 
         ha="center", fontsize=10, transform=ax2.transAxes)

# Rotate x-axis labels for readability
ax2.tick_params(axis='x', rotation=45)

# Adjust layout to ensure the plots don't overlap
plt.tight_layout()

# Show both plots
plt.show()


# %% [markdown]
# # any idea what might explain this phenomena?

# %%
# Define a function to calculate histogram data for 3D plotting
def calc_hist_data(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate bin centers
    return hist, bin_centers, bin_edges

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the properties to loop through
real_estate_types = ['Kamer', 'Appartement', 'Rijhuis']
colors = ['red', 'green', 'blue']  # Different colors for each histogram
offsets = [0, 5, 10]  # Set offsets to stack the histograms in the z-axis

# Define the x-axis limits
x_min, x_max = 200, 2000

# Loop through each property type and create the 3D histograms
for i, (real_estate_type, color, offset) in enumerate(zip(real_estate_types, colors, offsets)):
    # Filter data for the specific real estate type
    data = filtered_data_timespan[filtered_data_timespan.MappedRealEstateType == real_estate_type].RentalFeeMonthly

    # Filter the data to only include values within the x-axis range
    data = data[(data >= x_min) & (data <= x_max)]

    # Calculate histogram data
    hist, bin_centers, bin_edges = calc_hist_data(data, bins=1000)

    # Plot only the bars where hist > 0 (i.e., where data exists)
    for j in range(len(hist)):
        if hist[j] > 0:  # Only plot where data exists
            ax.bar3d(bin_centers[j], offset, 0,  # X, Y, Z bottom
                     bin_edges[j+1] - bin_edges[j],  # Width
                     1,  # Depth (constant for stacking)
                     hist[j], color=color, alpha=0.6)

# Set axis labels
ax.set_xlabel('Rental Fee Monthly')
ax.set_zlabel('Frequency')
ax.set_title('3D Stacked Histograms of Rental Fees by Property Type')
ax.set_xlim([x_min, x_max])

# Customize ticks for real estate types on the Y-axis
ax.set_yticks(offsets)
ax.set_yticklabels(real_estate_types)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### it would seem that the bulk of student housing contracts start in september
# ### as these are significantly cheaper then appartments, or houses, these push down the aggregate index!

# %%
# Filter data for source A and B
filtered_data_A = filtered_data[(filtered_data['source'] == 'A') & (filtered_data['MappedRealEstateType'] == 'Appartement')].resample('M').size()
filtered_data_B = filtered_data[(filtered_data['source'] == 'B') & (filtered_data['MappedRealEstateType'] == 'Appartement')].resample('M').size()

# Merge the two datasets by their index (the month), filling missing months with 0
monthly_counts = pd.DataFrame({
    'Source A': filtered_data_A,
    'Source B': filtered_data_B
}).fillna(0)

# Filter data to start from January 2017
monthly_counts = monthly_counts[monthly_counts.index >= '2017-01']

# Plot the count of contracts per month for both sources in a grouped bar chart
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='bar', width=0.8, edgecolor='black', ax=plt.gca())

# Customize the plot
plt.title('Count of Contracts Per Month for Appartements (Source A vs Source B)')
plt.xlabel('Month')
plt.ylabel('Count of Contracts')

# Set x-ticks to show every 6th label
plt.xticks(ticks=range(0, len(monthly_counts), 6), labels=monthly_counts.index.strftime('%Y-%m')[::6], rotation=45)

plt.tight_layout()

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt

def plot_rental_vs_cpi(real_estate_type, data, cpi_data):
    # Filter the rental data for the given real estate type and time range
    filtered_data = data[data['MappedRealEstateType'] == real_estate_type]
    filtered_data_timespan = filtered_data[(filtered_data.index >= '2018-01-01') & (filtered_data.index < '2024-07-01')]

    # Resample the filtered data monthly and calculate the mean, median, and count of RentalFeeMonthly
    monthly_stats = filtered_data_timespan['RentalFeeMonthly'].resample('M').agg(['mean', 'median', 'count'])

    # Normalize rental fee data by dividing by the first value and multiplying by 100
    monthly_stats['mean_normalized'] = (monthly_stats['mean'] / monthly_stats['mean'].iloc[0]) * 100
    monthly_stats['median_normalized'] = (monthly_stats['median'] / monthly_stats['median'].iloc[0]) * 100

    # Filter the CPI data for the same time period and create a copy to avoid the SettingWithCopyWarning
    filtered_CPI = cpi_data[(cpi_data.index >= '2018-01-01') & (cpi_data.index < '2024-07-01')].copy()

    # Normalize CPI data by dividing by the first value and multiplying by 100
    filtered_CPI['Value_normalized'] = (filtered_CPI['Value'] / filtered_CPI['Value'].iloc[0]) * 100

    # Create subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --------------- First Plot: Two Y-Axes (Rental Fee and CPI) ---------------
    # Plot mean and median over time on the primary y-axis
    ax1.plot(monthly_stats.index, monthly_stats['mean'], label='Mean Rental Fee', color='blue', linewidth=2)
    ax1.plot(monthly_stats.index, monthly_stats['median'], label='Median Rental Fee', color='green', linestyle='--', linewidth=2)

    # Set labels for the primary y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rental Fee Monthly')
    ax1.set_title(f'Mean and Median Rental Fee vs CPI ({real_estate_type}) (2017-2025)')
    ax1.legend(loc='upper left')

    # Rotate x-axis labels for readability
    ax1.tick_params(axis='x', rotation=45)

    # Create a second y-axis to plot CPI data
    ax3 = ax1.twinx()
    ax3.plot(filtered_CPI.index, filtered_CPI['Value'], label='CPI', color='orange', linewidth=2)

    # Set label for the secondary y-axis
    ax3.set_ylabel('CPI Value')
    ax3.legend(loc='upper right')

    # --------------- Second Plot: Normalized Data (Rental Fee and CPI) ---------------
    # Plot normalized mean and median over time
    ax2.plot(monthly_stats.index, monthly_stats['mean_normalized'], label='Mean Rental Fee (Normalized)', color='blue', linewidth=2)

    # Plot the normalized CPI values
    ax2.plot(filtered_CPI.index, filtered_CPI['Value_normalized'], label='CPI (Normalized)', color='orange', linewidth=2)

    # Set labels for the second plot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Value (Base = 100)')
    ax2.set_title(f'Normalized Rental Fee vs CPI ({real_estate_type}) (2017-2025)')
    ax2.legend(loc='upper left')

    # Rotate x-axis labels for readability
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to ensure the plots don't overlap
    plt.tight_layout()

    # Show both plots
    plt.show()

# Example usage
plot_rental_vs_cpi('Appartement', filtered_data_timespan, filtered_CPI)
plot_rental_vs_cpi('Appartement', filtered_data_timespan[filtered_data_timespan['source'] == 'A'], filtered_CPI)
plot_rental_vs_cpi('Appartement', filtered_data_timespan[filtered_data_timespan['source'] == 'B'], filtered_CPI)

# %% [markdown]
# ## This is rather unstable, if you play around with the starting date, this will have a large impact on your analysis
# 
# ##### Say we want to monitor the rent households are paying for Appartments they inhabit,
# ##### yet we must take into account that we only have the rent households pay during the first year 
# ##### (as at the start of a second year the rent can be indexed, or could stay the same)

# %%
# Define the quantile for outlier removal (0.1% and 99.9%)
lower_quantile = data_MBF['RentalFeeMonthly'].quantile(0.001)
upper_quantile = data_MBF['RentalFeeMonthly'].quantile(0.999)

# Filter the data to remove outliers (keeping only the middle 98%)
filtered_data = data_MBF[(data_MBF['RentalFeeMonthly'] >= lower_quantile) & 
                         (data_MBF['RentalFeeMonthly'] <= upper_quantile)]

#-> note this here is pretty bad coding, I should not have to redefine 'filtered_data' !!!

# %%
# Function to calculate active stats with a given offset
def calculate_active_stats(filtered_data, date_range, offset_months):
    mean_values = []
    median_values = []

    # Make sure 'RentalPeriodFrom' & 'RentalPeriodTo' are datetime objects
    filtered_data['RentalPeriodFrom'] = pd.to_datetime(filtered_data['RentalPeriodFrom'])
    filtered_data['RentalPeriodTo'] = pd.to_datetime(filtered_data['RentalPeriodTo'])

    # Iterate over each month in the date range
    for date in date_range:
        # Apply the offset to 'RentalPeriodFrom'
        rental_period_with_offset = filtered_data['RentalPeriodFrom'] + pd.DateOffset(months=offset_months)

        # Create a DataFrame with 'RentalPeriodTo' and the 'RentalPeriodFrom' + offset
        temp_df = pd.DataFrame({
            'RentalPeriodTo': filtered_data['RentalPeriodTo'],
            'RentalPeriodWithOffset': rental_period_with_offset
        })

        # Find the minimum date between 'RentalPeriodTo' and 'RentalPeriodFrom + offset'
        min_dates = temp_df.min(axis=1)

        # Find contracts that are active during the current month but only up to `offset_months` months after RentalPeriodFrom
        active_contracts = filtered_data[
            (filtered_data['RentalPeriodFrom'] <= date) & 
            (min_dates >= date)
        ]
        
        # Calculate the mean and median for active contracts
        if not active_contracts.empty:
            mean_values.append(active_contracts['RentalFeeMonthly'].mean())
            median_values.append(active_contracts['RentalFeeMonthly'].median())
        else:
            mean_values.append(np.nan)  # Append NaN if no contracts are active
            median_values.append(np.nan)

    # Create a DataFrame with the results
    active_stats = pd.DataFrame({
        'Date': date_range,
        'Mean': mean_values,
        'Median': median_values
    })

    # Normalize the mean and median by dividing by the first value and multiplying by 100
    active_stats['Mean_normalized'] = (active_stats['Mean'] / active_stats['Mean'].iloc[0]) * 100
    active_stats['Median_normalized'] = (active_stats['Median'] / active_stats['Median'].iloc[0]) * 100
    
    return active_stats

# Function to generate plots
def generate_plots(filtered_data, cpi_data, title_suffix):
    # Ensure 'RentalPeriodFrom' and 'RentalPeriodTo' are in datetime format
    filtered_data['RentalPeriodFrom'] = pd.to_datetime(filtered_data['RentalPeriodFrom'])
    filtered_data['RentalPeriodTo'] = pd.to_datetime(filtered_data['RentalPeriodTo'])

    # Create a date range for the time period we are interested in
    date_range = pd.date_range(start='2018-01-01', end='2024-06-30', freq='M')

    # Calculate active stats for 12 months offset
    active_stats_12_months = calculate_active_stats(filtered_data, date_range, 12)

    # Calculate active stats for 1 month offset
    active_stats_1_month = calculate_active_stats(filtered_data, date_range, 1)

    # Normalize CPI data
    filtered_CPI = cpi_data[(cpi_data.index >= '2018-01-01') & (cpi_data.index < '2024-07-01')].copy()
    filtered_CPI['Value_normalized'] = (filtered_CPI['Value'] / filtered_CPI['Value'].iloc[0]) * 100

    # Create subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # --------------- First Plot: 12-Month Offset ---------------
    ax1.plot(active_stats_12_months['Date'], active_stats_12_months['Mean_normalized'], label='Mean Rental Fee (Normalized)', color='blue', linewidth=2)
    ax1.plot(active_stats_12_months['Date'], active_stats_12_months['Median_normalized'], label='Median Rental Fee (Normalized)', color='green', linestyle='--', linewidth=2)
    ax1.plot(filtered_CPI.index, filtered_CPI['Value_normalized'], label='CPI (Normalized)', color='orange', linewidth=2)

    # Set labels for the first plot
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Value (Base = 100)')
    ax1.set_title(f'Normalized Rental Fee (12-Month Offset) vs CPI ({title_suffix})')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # --------------- Second Plot: 1-Month Offset ---------------
    ax2.plot(active_stats_1_month['Date'], active_stats_1_month['Mean_normalized'], label='Mean Rental Fee (Normalized)', color='blue', linewidth=2)
    ax2.plot(active_stats_1_month['Date'], active_stats_1_month['Median_normalized'], label='Median Rental Fee (Normalized)', color='green', linestyle='--', linewidth=2)
    ax2.plot(filtered_CPI.index, filtered_CPI['Value_normalized'], label='CPI (Normalized)', color='orange', linewidth=2)

    # Set labels for the second plot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Value (Base = 100)')
    ax2.set_title(f'Normalized Rental Fee (1-Month Offset) vs CPI ({title_suffix})')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to ensure nothing overlaps
    plt.tight_layout()

    # Show the plot
    plt.show()

# Loop through data subsets: All data, source 'A', and source 'B'
for source in ['All Data', 'A', 'B']:
    if source == 'All Data':
        data_subset = filtered_data[filtered_data.MappedRealEstateType=="Appartement"]
    else:
        data_subset = filtered_data[(filtered_data['source'] == source) & (filtered_data['MappedRealEstateType'] == "Appartement")].copy()  # Create a copy for source-based data

    # Generate plots for each data subset
    generate_plots(data_subset, CPI, source)

# %%
data_MBF

# %%
#PLOT 1 HEATMAP
# All unique postal codes
postal_codes = data_MBF['location_postalCode'].unique()
print(postal_codes)
    
#dictionary for the medians    
medians_postalCode = {}

# Iterate over each postal code and calculate the median for 'RentalFeeMonthly'
for code in postal_codes:
    medians_postalCode[code] = data_MBF[data_MBF['location_postalCode'] == code]['RentalFeeMonthly'].median()

# Print the medians for each postal code
print(medians_postalCode)

#import for heatmap
import seaborn as sns

# Convert the dictionary to a DataFrame for better visualization
dataframe_medians_postalCode = pd.DataFrame(list(medians_postalCode.items()), columns=['PostalCode', 'MedianRentalFee'])

# Set the postal code as the index for better heatmap readability
dataframe_medians_postalCode = dataframe_medians_postalCode.set_index('PostalCode')

# Create a figure and heatmap
plt.figure(figsize=(12, 10))

sns.heatmap(dataframe_medians_postalCode, annot=True, fmt='.0f', cmap="YlOrRd", linewidths=.5, annot_kws={"size": 12})

# Add titles and labels
plt.title('Median Rental Fee by Postal Code')
plt.xlabel('Rental Prices')
plt.ylabel('Postal Codes')

# Display the heatmap
plt.show()

#PLOT 2
