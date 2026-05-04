import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Membaca data bersih
df = pd.read_csv("cleaned_superstore.csv")

# Mengubah order_date menjadi format tanggal
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# Mengubah data transaksi menjadi data bulanan
monthly_data = (
    df.set_index("order_date")
    .resample("MS")
    .agg({
        "sales": "sum",
        "profit": "sum"
    })
    .reset_index()
)

# Membuat fitur waktu
monthly_data["year"] = monthly_data["order_date"].dt.year
monthly_data["month"] = monthly_data["order_date"].dt.month
monthly_data["time_index"] = np.arange(len(monthly_data))

print("Data bulanan:")
print(monthly_data.head())
print("\nJumlah data bulanan:", monthly_data.shape)


def create_features(data):
    month_dummies = pd.get_dummies(data["month"], prefix="month")
    X = pd.concat([data[["time_index"]], month_dummies], axis=1)
    return X


def forecast_target(target_column):
    print(f"\n===== Forecasting {target_column.upper()} =====")

    # Model diuji dengan cara:
    # Training: 2011-2013
    # Testing : 2014
    train_data = monthly_data[monthly_data["year"] <= 2013]
    test_data = monthly_data[monthly_data["year"] == 2014]

    X_train = create_features(train_data)
    y_train = train_data[target_column]

    X_test = create_features(test_data)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = test_data[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Evaluasi prediksi tahun 2014:")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # Setelah evaluasi, model dilatih ulang memakai semua data 2011-2014
    X_all = create_features(monthly_data)
    y_all = monthly_data[target_column]

    final_model = LinearRegression()
    final_model.fit(X_all, y_all)

    # Membuat data masa depan untuk tahun 2015
    future_dates = pd.date_range(start="2015-01-01", periods=12, freq="MS")

    future_data = pd.DataFrame({
        "order_date": future_dates
    })

    future_data["year"] = future_data["order_date"].dt.year
    future_data["month"] = future_data["order_date"].dt.month
    future_data["time_index"] = np.arange(
        monthly_data["time_index"].max() + 1,
        monthly_data["time_index"].max() + 1 + len(future_data)
    )

    X_future = create_features(future_data)
    X_future = X_future.reindex(columns=X_all.columns, fill_value=0)

    future_data[f"predicted_{target_column}"] = final_model.predict(X_future)

    return future_data[["order_date", "year", "month", f"predicted_{target_column}"]]


# Forecast sales dan profit
sales_forecast = forecast_target("sales")
profit_forecast = forecast_target("profit")

# Menggabungkan hasil prediksi sales dan profit
forecast_2015 = sales_forecast.merge(
    profit_forecast,
    on=["order_date", "year", "month"]
)

# Simpan hasil forecasting
forecast_2015.to_csv("forecast_2015.csv", index=False)

print("\n===== HASIL FORECAST 2015 =====")
print(forecast_2015)

print("\nFile forecast_2015.csv berhasil dibuat.")