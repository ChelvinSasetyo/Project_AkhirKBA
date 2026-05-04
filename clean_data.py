import pandas as pd
import numpy as np

# 1. Membaca dataset mentah
df = pd.read_csv("SuperStore Sales Dataset (1).csv", delimiter=";")

print("Jumlah data awal:", df.shape)

# 2. Hapus duplikat penuh
df = df.drop_duplicates()

# 3. Bersihkan nama kolom jika ada spasi tersembunyi
df.columns = df.columns.str.strip()

# 4. Ubah kolom tanggal menjadi format datetime
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=True, errors="coerce")

# 5. Fungsi untuk membersihkan angka
def clean_number(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Menghapus koma, contoh: 2,574 menjadi 2574
    value = value.replace(",", "")

    # Jika ada lebih dari satu titik, contoh: 4.942.824 menjadi 4942.824
    if value.count(".") > 1:
        parts = value.split(".")
        value = "".join(parts[:-1]) + "." + parts[-1]

    return pd.to_numeric(value, errors="coerce")

# 6. Ubah kolom angka menjadi numeric
numeric_columns = ["sales", "quantity", "discount", "profit", "shipping_cost", "year"]

for col in numeric_columns:
    df[col] = df[col].apply(clean_number)

# 7. Hapus baris yang masih kosong setelah cleaning
df = df.dropna()

# 8. Buat kolom tambahan untuk analisis
df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days
df["profit_margin"] = df["profit"] / df["sales"]
df["month"] = df["order_date"].dt.month
df["month_name"] = df["order_date"].dt.month_name()

# 9. Buat target untuk machine learning
df["profit_status"] = np.where(df["profit"] > 0, "Profit", "Loss")

# 10. Simpan hasil cleaning
df.to_csv("cleaned_superstore.csv", index=False)

print("Jumlah data setelah cleaning:", df.shape)
print("Data kosong setelah cleaning:")
print(df.isnull().sum())
print("Jumlah duplikat setelah cleaning:", df.duplicated().sum())
print("File cleaned_superstore.csv berhasil dibuat.")