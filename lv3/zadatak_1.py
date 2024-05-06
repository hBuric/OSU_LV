import pandas as pd
import numpy as np
import matplotlib . pyplot as plt

data = pd.read_csv('data_C02_emission.csv')


print('\na) zadatak\n')
print(data)
print('Tip stvarne veličine: ', data.dtypes)
print('\nBroj mjernja: ', data.__len__())
print('\nDuplicirane vrijednosti: ', len(data.isnull()))
data.drop_duplicates()
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')


print('\nb) zadatak\n')
print('3 automobila sa najvećom gradskom potrošnjom:', data.sort_values(by=["Fuel Consumption City (L/100km)"]).head(3)[["Make","Model", "Fuel Consumption City (L/100km)"]])
print('\n3 automobila sa najmanjom gradskom potrošnjom:', data.sort_values(by=["Fuel Consumption City (L/100km)"]).tail(3)[["Make","Model", "Fuel Consumption City (L/100km)"]])


print('\nc) zadatak\n')
print('Broj vozila sa veličinom motora između 2.5 i 3.5 L: ', data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)].__len__()) 
print('\nProsječna a C02 emisija plinova: ', data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]["CO2 Emissions (g/km)"].mean())


print('\nd) zadatak\n')
print('Broj vozila proizvodača Audi:', len(data[data["Make"] == "Audi"]))
print('\nProsječna emisija C02 plinova automobila proizvodača Audi koji imaju 4 cilindara: ', data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)]["CO2 Emissions (g/km)"].mean())


print('\ne) zadatak\n')  
print('Broj vozila s obzirom na broj cilindara: ', data.groupby("Cylinders").size())
print('\nProsječna emisija C02 plinova s obzirom na broj cilindara: ', data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())
print('\nMedijan vrijednost emisije C02 plinova s obzirom na broj cilindara: ', data.groupby("Cylinders")["CO2 Emissions (g/km)"].median())


print('\nf) zadatak\n')
print('Prosječna gradska potrošnja u slučaju vozila koja koriste dizel: ', data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean())
print('\nProsječna gradska potrošnja u slučaju vozila koja koriste benzin:', data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean())
print('\nMedian vrijednost potrošnje u slučaju vozila koja koriste dizel:', data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median())
print('\nMedian vrijednost potrošnje u slučaju vozila koja koriste benzin: ', data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median())


print('\ng) zadatak\n')
print('Vozilo s 4 cilindra koje koristi dizelski motor I ima najvecu gradsku potrošnju goriva: ', data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")].sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False).head(1))


print('\nh) zadatak\n')
print('Broj vozila sa ručnim mjenjačem: ', data[data["Transmission"].str.startswith("M")].__len__())


print('\ni) zadatak\n')
print('Korelacija: ', data.corr(numeric_only=True))
