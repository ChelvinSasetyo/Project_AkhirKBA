import streamlit as st
import pandas as pd
import plotly.express as px

# Membaca data yang sudah dibersihkan
df = pd.read_csv("cleaned_superstore.csv")

# Mengubah order_date menjadi format tanggal
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

st.set_page_config(page_title="Dashboard Superstore", layout="wide")

st.title("Dashboard Analitik Penjualan Superstore")
st.write("Dashboard ini menampilkan analisis penjualan, profit, kategori produk, dan wilayah berdasarkan data Superstore.")

# Sidebar Filter
st.sidebar.header("Filter Data")

year_options = sorted(df["year"].astype(int).unique())
selected_year = st.sidebar.multiselect(
    "Pilih Tahun",
    year_options,
    default=year_options
)

market_options = sorted(df["market"].unique())
selected_market = st.sidebar.multiselect(
    "Pilih Market",
    market_options,
    default=market_options
)

category_options = sorted(df["category"].unique())
selected_category = st.sidebar.multiselect(
    "Pilih Kategori",
    category_options,
    default=category_options
)

# Checkbox
st.sidebar.header("Checkbox")
show_profit = st.sidebar.checkbox("Tampilkan transaksi Profit", value=True)
show_loss = st.sidebar.checkbox("Tampilkan transaksi Loss", value=True)

# Apply filter
filtered_df = df[
    (df["year"].astype(int).isin(selected_year)) &
    (df["market"].isin(selected_market)) &
    (df["category"].isin(selected_category))
]

if show_profit and not show_loss:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Profit"]
elif show_loss and not show_profit:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Loss"]
elif not show_profit and not show_loss:
    filtered_df = filtered_df.iloc[0:0]

# KPI
total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
total_quantity = filtered_df["quantity"].sum()
total_orders = filtered_df["order_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Total Profit", f"{total_profit:,.2f}")
col3.metric("Total Quantity", f"{total_quantity:,.0f}")
col4.metric("Total Orders", f"{total_orders:,.0f}")

st.divider()

# Grafik 1: Sales per Tahun
sales_by_year = filtered_df.groupby("year", as_index=False)["sales"].sum()

fig_sales_year = px.bar(
    sales_by_year,
    x="year",
    y="sales",
    title="Total Sales Berdasarkan Tahun"
)

st.plotly_chart(fig_sales_year, use_container_width=True)

# Grafik 2: Profit berdasarkan kategori
profit_by_category = filtered_df.groupby("category", as_index=False)["profit"].sum()

fig_profit_category = px.bar(
    profit_by_category,
    x="category",
    y="profit",
    title="Total Profit Berdasarkan Kategori"
)

st.plotly_chart(fig_profit_category, use_container_width=True)

# Grafik 3: Sales berdasarkan region
sales_by_region = filtered_df.groupby("region", as_index=False)["sales"].sum()

fig_sales_region = px.bar(
    sales_by_region,
    x="region",
    y="sales",
    title="Total Sales Berdasarkan Region"
)

st.plotly_chart(fig_sales_region, use_container_width=True)

# Grafik 4: Hubungan Discount dan Profit
fig_discount_profit = px.scatter(
    filtered_df,
    x="discount",
    y="profit",
    color="profit_status",
    title="Hubungan Discount dan Profit",
    opacity=0.5
)

st.plotly_chart(fig_discount_profit, use_container_width=True)

# Tabel Data
st.subheader("Preview Data")
st.dataframe(filtered_df.head(100))

st.divider()
st.header("Machine Learning: Klasifikasi Profit/Loss")

show_ml = st.checkbox("Tampilkan hasil Machine Learning", value=True)

if show_ml:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        "order_priority"
    ]

    X = df[features]
    y = df["profit_status"]

    numeric_features = [
        "sales",
        "quantity",
        "discount",
        "shipping_cost",
        "shipping_days"
    ]

    categorical_features = [
        "segment",
        "market",
        "region",
        "category",
        "sub_category",
        "ship_mode",
        "order_priority"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Akurasi Model", f"{accuracy * 100:.2f}%")

    labels = ["Loss", "Profit"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Loss", "Actual Profit"],
        columns=["Predicted Loss", "Predicted Profit"]
    )

    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix Klasifikasi Profit/Loss"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)