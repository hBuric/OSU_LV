import pandas as pd
import numpy as np
import matplotlib . pyplot as plt

data = pd.read_csv('LV3/data_C02_emission.csv')

print()
print(data)
data = data.drop_duplicates()
data = data.dropna(axis = 0)
data = data.dropna(axis = 1)
data = data.reset_index(drop = True)

#a
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=25)

#b
plt.figure()
colors = {'D': 'red', 'E': 'blue', 'X': 'green', 'Z': 'yellow'}
for fuel_type, color in colors.items():
    subset = data[data['Fuel Type'] == fuel_type]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'], color=color,  s=5, alpha= 0.5)

#c
data.boxplot(column=['Fuel Consumption City (L/100km)'], by='Fuel Type')

#d
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
grouped_data_size = data.groupby('Fuel Type').size()
grouped_data_size.plot(kind='bar', ax=ax1)
ax1.set_xlabel('Fuel Type')
ax1.set_ylabel('Count')
    
#e
mean_co2_by_cylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
mean_co2_by_cylinders.plot(kind='bar', ax=ax2)
ax2.set_xlabel('Cylinders')
ax2.set_ylabel('CO2 Emissions (g/km)')

plt.show()
