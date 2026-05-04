import pandas as pd

df = pd.read_csv("SuperStore Sales Dataset (1).csv", delimiter=";")

print(df.head())
print("Jumlah baris dan kolom:", df.shape)
print("Nama kolom:")
print(df.columns)
print("Data kosong:")
print(df.isnull().sum())
print("Jumlah duplikat:", df.duplicated().sum())