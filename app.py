import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Dashboard Superstore", layout="wide")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_superstore.csv")
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["ship_date"] = pd.to_datetime(data["ship_date"], errors="coerce")
    data["year"] = data["year"].astype(int)
    return data


@st.cache_data
def load_forecast():
    forecast = pd.read_csv("forecast_2015.csv")
    forecast["order_date"] = pd.to_datetime(forecast["order_date"], errors="coerce")
    forecast["month_name"] = forecast["order_date"].dt.month_name()
    return forecast


df = load_data()


# =========================
# TRAIN MODEL ML
# =========================
@st.cache_resource
def train_profit_loss_model(data):
    features = [
        "sales",
        "quantity",
        "discount",
        "shipping_cost",
        "shipping_days",
        "segment",
        "market",
        "region",
        "category",
        "sub_category",
        "ship_mode",
        "order_priority",
    ]

    X = data[features]
    y = data["profit_status"]

    numeric_features = [
        "sales",
        "quantity",
        "discount",
        "shipping_cost",
        "shipping_days",
    ]

    categorical_features = [
        "segment",
        "market",
        "region",
        "category",
        "sub_category",
        "ship_mode",
        "order_priority",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    labels = ["Loss", "Profit"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True)

    return accuracy, cm, report


# =========================
# TITLE
# =========================
st.title("Dashboard Analitik Penjualan Superstore")
st.write(
    "Dashboard ini menampilkan analisis penjualan, profit, kategori produk, wilayah, "
    "klasifikasi Profit/Loss, dan forecasting Sales serta Profit tahun 2015."
)


# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter Data")

year_options = sorted(df["year"].unique())
selected_year = st.sidebar.multiselect(
    "Pilih Tahun",
    year_options,
    default=year_options,
)

market_options = sorted(df["market"].dropna().unique())
selected_market = st.sidebar.multiselect(
    "Pilih Market",
    market_options,
    default=market_options,
)

category_options = sorted(df["category"].dropna().unique())
selected_category = st.sidebar.multiselect(
    "Pilih Kategori",
    category_options,
    default=category_options,
)

segment_options = sorted(df["segment"].dropna().unique())
selected_segment = st.sidebar.multiselect(
    "Pilih Segment",
    segment_options,
    default=segment_options,
)

st.sidebar.header("Checkbox")
show_profit = st.sidebar.checkbox("Tampilkan transaksi Profit", value=True)
show_loss = st.sidebar.checkbox("Tampilkan transaksi Loss", value=True)
show_preview = st.sidebar.checkbox("Tampilkan preview data", value=True)


# =========================
# APPLY FILTER
# =========================
filtered_df = df[
    (df["year"].isin(selected_year))
    & (df["market"].isin(selected_market))
    & (df["category"].isin(selected_category))
    & (df["segment"].isin(selected_segment))
]

if show_profit and not show_loss:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Profit"]
elif show_loss and not show_profit:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Loss"]
elif not show_profit and not show_loss:
    filtered_df = filtered_df.iloc[0:0]


# =========================
# KPI
# =========================
total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
total_quantity = filtered_df["quantity"].sum()
total_orders = filtered_df["order_id"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales != 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Total Profit", f"{total_profit:,.2f}")
col3.metric("Total Quantity", f"{total_quantity:,.0f}")
col4.metric("Total Orders", f"{total_orders:,.0f}")
col5.metric("Profit Margin", f"{profit_margin:.2f}%")

st.divider()


# =========================
# VISUALISASI DATA
# =========================
st.header("Visualisasi Data Penjualan")

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
else:
    sales_by_year = filtered_df.groupby("year", as_index=False)["sales"].sum()
    fig_sales_year = px.bar(
        sales_by_year,
        x="year",
        y="sales",
        title="Total Sales Berdasarkan Tahun",
    )
    st.plotly_chart(fig_sales_year, use_container_width=True)

    profit_by_category = filtered_df.groupby("category", as_index=False)["profit"].sum()
    fig_profit_category = px.bar(
        profit_by_category,
        x="category",
        y="profit",
        title="Total Profit Berdasarkan Kategori",
    )
    st.plotly_chart(fig_profit_category, use_container_width=True)

    sales_by_region = filtered_df.groupby("region", as_index=False)["sales"].sum()
    fig_sales_region = px.bar(
        sales_by_region,
        x="region",
        y="sales",
        title="Total Sales Berdasarkan Region",
    )
    st.plotly_chart(fig_sales_region, use_container_width=True)

    fig_discount_profit = px.scatter(
        filtered_df,
        x="discount",
        y="profit",
        color="profit_status",
        title="Hubungan Discount dan Profit",
        opacity=0.5,
    )
    st.plotly_chart(fig_discount_profit, use_container_width=True)

    monthly_trend = (
        filtered_df.groupby(["year", "month"], as_index=False)
        .agg({"sales": "sum", "profit": "sum"})
    )
    monthly_trend["period"] = (
        monthly_trend["year"].astype(str)
        + "-"
        + monthly_trend["month"].astype(str).str.zfill(2)
    )

    fig_monthly = px.line(
        monthly_trend,
        x="period",
        y=["sales", "profit"],
        markers=True,
        title="Tren Sales dan Profit per Bulan",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    if show_preview:
        st.subheader("Preview Data")
        st.dataframe(filtered_df.head(100), use_container_width=True)


# =========================
# MACHINE LEARNING
# =========================
st.divider()
st.header("Machine Learning: Klasifikasi Profit/Loss")
st.write(
    "Model Random Forest digunakan untuk mengklasifikasikan transaksi menjadi Profit atau Loss "
    "berdasarkan pola sales, quantity, discount, shipping cost, kategori produk, wilayah, dan metode pengiriman."
)

accuracy, cm, report = train_profit_loss_model(df)

st.metric("Akurasi Model", f"{accuracy * 100:.2f}%")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Loss", "Actual Profit"],
    columns=["Predicted Loss", "Predicted Profit"],
)

st.subheader("Confusion Matrix")
fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    title="Confusion Matrix Klasifikasi Profit/Loss",
)
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)


# =========================
# FORECASTING 2015
# =========================
st.divider()
st.header("Forecasting Sales dan Profit Tahun 2015")
st.write(
    "Bagian ini menampilkan prediksi Sales dan Profit tahun 2015. "
    "Prediksi dibuat dari pola data historis 2011 sampai 2014 yang sudah diringkas menjadi data bulanan."
)

try:
    forecast_df = load_forecast()

    forecast_total_sales = forecast_df["predicted_sales"].sum()
    forecast_total_profit = forecast_df["predicted_profit"].sum()
    forecast_margin = (
        forecast_total_profit / forecast_total_sales * 100
        if forecast_total_sales != 0
        else 0
    )

    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Prediksi Total Sales 2015", f"{forecast_total_sales:,.2f}")
    fcol2.metric("Prediksi Total Profit 2015", f"{forecast_total_profit:,.2f}")
    fcol3.metric("Prediksi Profit Margin 2015", f"{forecast_margin:.2f}%")

    st.subheader("Tabel Prediksi Tahun 2015")
    st.dataframe(forecast_df, use_container_width=True)

    fig_forecast_sales = px.line(
        forecast_df,
        x="order_date",
        y="predicted_sales",
        markers=True,
        title="Prediksi Sales Tahun 2015",
    )
    st.plotly_chart(fig_forecast_sales, use_container_width=True)

    fig_forecast_profit = px.line(
        forecast_df,
        x="order_date",
        y="predicted_profit",
        markers=True,
        title="Prediksi Profit Tahun 2015",
    )
    st.plotly_chart(fig_forecast_profit, use_container_width=True)

except FileNotFoundError:
    st.warning(
        "File forecast_2015.csv belum ditemukan. Jalankan dulu forecast_model.py dengan perintah: py forecast_model.py"
    )
