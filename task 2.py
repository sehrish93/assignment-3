
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_worldbank(filename: str):
  """
    Reads a file containing world bank data and returns the original dataframe, the dataframe with countries as columns, and the dataframe with year as columns.
    
    Parameters:
    - filename (str): The name of the file to be read, including the file path.
    
    Returns:
    - dataframe (pandas dataframe): The original dataframe containing the data from the file.
    - df_transposed_country (pandas dataframe): Dataframe with countries as columns.
    - df_transposed_year (pandas dataframe): Dataframe with year as columns.
  """
  # Read the file into a pandas dataframe
  dataframe = pd.read_csv(filename)
    
  # Transpose the dataframe
  df_transposed = dataframe.transpose()
    
  # Populate the header of the transposed dataframe with the header information 
   
  # silice the dataframe to get the year as columns
  df_transposed.columns = df_transposed.iloc[1]

  # As year is now columns so we don't need it as rows
  df_transposed_year = df_transposed[0:].drop('year')
    
  # silice the dataframe to get the country as columns
  df_transposed.columns = df_transposed.iloc[0]
    
  # As country is now columns so we don't need it as rows
  df_transposed_country = df_transposed[0:].drop('country')
    
  return dataframe, df_transposed_country, df_transposed_year

# load data from World Bank website or a similar source
df, df_country, df_year = read_worldbank('wb_cc_dataset.csv')

def remove_null_values(feature):
  """
  This function removes null values from a given feature.


  Parameters:
    feature (pandas series): The feature to remove null values from.

  Returns:
    numpy array: The feature with null values removed.
  """
  # drop null values from the feature
  return np.array(feature.dropna())

df.columns

def balance_data(df):
  """
  This function takes a dataframe as input and removes missing values from each column individually.
  It then returns a balanced dataset with the same number of rows for each column.

  Input:

  df (pandas dataframe): a dataframe containing the data to be balanced
  Output:

  balanced_df (pandas dataframe): a dataframe with the same number of rows for each column, after removing missing values from each column individually
  """
  # Making dataframe of all the feature in the avaiable in 
  # dataframe passing it to remove null values function 
  # for dropping the null values 
  greenhouse_gas_emissions = remove_null_values(df[['greenhouse_gas_emissions']])

  argicultural_land = remove_null_values(df[['agricultural_land']])

  co2_emission = remove_null_values(df[['co2_emissions']])

  arable_land = remove_null_values(df[['arable_land']])

  cereal_yield = remove_null_values(df[['cereal_yield']])

  population_growth = remove_null_values(df[['population_growth']])

  urban_population = remove_null_values(df[['urban_population']])

  GDP = remove_null_values(df[['GDP']])

  min_length = min(len(greenhouse_gas_emissions), len(argicultural_land), len(co2_emission),len(arable_land), len(cereal_yield),
                   len(population_growth), len(urban_population), len(GDP))
  # after removing the null values we will create datafram 

  clean_data = pd.DataFrame({ 
                                'country': [df['country'].iloc[x] for x in range(min_length)],
                                'year': [df['year'].iloc[x] for x in range(min_length)],
                                'greenhouse_gas_emissions': [greenhouse_gas_emissions[x][0] for x in range(min_length)],
                                'argicultural_land': [argicultural_land[x][0] for x in range(min_length)],
                                 'co2_emission': [co2_emission[x][0] for x in range(min_length)],
                                 'arable_land': [arable_land[x][0] for x in range(min_length)],
                                 'cereal_yield': [cereal_yield[x][0] for x in range(min_length)],
                                 'population_growth': [population_growth[x][0] for x in range(min_length)],
                                 'urban_population': [urban_population[x][0] for x in range(min_length)],
                                 'GDP': [GDP[x][0] for x in range(min_length)]
                                 })
  return clean_data

# Clean and preprocess the data
df = balance_data(df)

df

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['greenhouse_gas_emissions', 'argicultural_land', 'co2_emission',
                                      'arable_land', 'cereal_yield',
                                      'population_growth', 'urban_population', 'GDP']])

# Use KMeans to find clusters in the data
kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_data)

# Add the cluster assignments as a new column to the dataframe
df['cluster'] = kmeans.labels_

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['argicultural_land'], cluster_data['cereal_yield'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('argicultural_land')
plt.ylabel('cereal_yield')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['urban_population'], cluster_data['arable_land'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('urban_population')
plt.ylabel('arable_land')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

df.country.unique()

us = df[df['country'] == 'United States']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = us[us['cluster'] == i]
    plt.scatter(cluster_data['argicultural_land'], cluster_data['cereal_yield'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('argicultural_land')
plt.ylabel('cereal_yield')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

ger = df[df['country'] == 'Germany']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = ger[ger['cluster'] == i]
    plt.scatter(cluster_data['argicultural_land'], cluster_data['cereal_yield'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('argicultural_land')
plt.ylabel('cereal_yield')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

def linear(x, a, b):
  """ Simple linear function calculating a + b*x. """
  f = a + b*x
  return f

# select a country from cluster 0
c1 = df[(df['cluster'] == 1)]

# Use err_ranges function to estimate lower and upper limits of the confidence range
x = c1['year']
y = c1['cereal_yield']

popt, pcov = curve_fit(linear, x, y)
x_pred = np.linspace(2021, 2040, 20)

# calculate the standard deviation for each parameter
sigma = np.sqrt(np.diag(pcov))
y_pred, y_pred_err = err_ranges(x_pred, linear, popt,sigma)

# Use pyplot to create a plot showing the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, linear(x, *popt), '-', label='fit')
plt.fill_between(x_pred, y_pred, y_pred_err, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('Year')
plt.ylabel('cereal_yield')
plt.show()
