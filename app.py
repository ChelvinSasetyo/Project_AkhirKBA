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
st.set_page_config(page_title="Dashboard Superstore", layout="wide", page_icon="📊")

# =========================
# THEME TOGGLE (SIANG/MALAM)
# =========================
# Tombol pengganti tema diletakkan di sidebar paling atas
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3094/3094926.png", width=100)
theme_mode = st.sidebar.radio("🌗 Tema Visual:", ["🌞 Siang (Light Nature)", "🌙 Malam (Dark Forest)"], horizontal=True)

is_dark = (theme_mode == "🌙 Malam (Dark Forest)")

# Konfigurasi Palet Warna Tema Alam (Hijau)
if is_dark:
    # Nuansa malam di hutan
    app_bg = "#0d1f15"
    sidebar_bg = "#0a170f"
    text_col = "#e8f5e9"
    card_bg = "#153322"
    accent_col = "#4caf50"
    plot_template = "plotly_dark"
    chart_colors = ['#81c784', '#aed581', '#4db6ac', '#81d4fa', '#ffb74d']
    status_color_map = {"Profit": "#66bb6a", "Loss": "#ef5350"}
    cm_color_scale = "Greens"
else:
    # Nuansa siang di taman
    app_bg = "#f4fcf5"
    sidebar_bg = "#e8f5e9"
    text_col = "#1b3320"
    card_bg = "#ffffff"
    accent_col = "#2e7d32"
    plot_template = "plotly_white"
    chart_colors = ['#2e7d32', '#558b2f', '#00695c', '#0277bd', '#e65100']
    status_color_map = {"Profit": "#1b5e20", "Loss": "#c62828"}
    cm_color_scale = "Greens"

# =========================
# INJEKSI CUSTOM CSS TEMA
# =========================
st.markdown(f"""
    <style>
    /* Background utama */
    .stApp {{
        background-color: {app_bg};
        color: {text_col};
    }}
    /* Background sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    /* Header transparant */
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    /* Warna Teks Global */
    .stMarkdown, h1, h2, h3, p, label {{
        color: {text_col} !important;
    }}
    /* Card KPI / Metric */
    div[data-testid="metric-container"] {{
        background-color: {card_bg};
        border: 1px solid {accent_col}40;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 6px solid {accent_col};
    }}
    div[data-testid="metric-container"] label {{
        color: {text_col} !important;
        opacity: 0.8;
    }}
    div[data-testid="metric-container"] div {{
        color: {text_col} !important;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {card_bg};
        border-radius: 8px 8px 0px 0px;
        border: 1px solid {accent_col}40;
        border-bottom: none;
        padding: 10px 20px;
        color: {text_col} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {accent_col}20 !important;
        border-bottom: 3px solid {accent_col} !important;
    }}
    </style>
""", unsafe_allow_html=True)


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
    features = ["sales", "quantity", "discount", "shipping_cost", "shipping_days", 
                "segment", "market", "region", "category", "sub_category", "ship_mode", "order_priority"]
    X = data[features]
    y = data["profit_status"]

    numeric_features = ["sales", "quantity", "discount", "shipping_cost", "shipping_days"]
    categorical_features = ["segment", "market", "region", "category", "sub_category", "ship_mode", "order_priority"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    labels = ["Loss", "Profit"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True)

    return accuracy, cm, report


# =========================
# HEADER DASHBOARD
# =========================
st.title("Dashboard Analitik & Prediksi Superstore")
st.markdown("Sistem ini menampilkan visualisasi penjualan interaktif, **Klasifikasi Machine Learning (Profit/Loss)**, serta **Forecasting 2015**.")
st.divider()


# =========================
# SIDEBAR FILTER
# =========================
with st.sidebar:
    st.divider()
    st.title("Parameter Filter")
    st.markdown("Sesuaikan data yang ingin dianalisis:")

    year_options = sorted(df["year"].unique())
    selected_year = st.multiselect("Pilih Tahun", year_options, default=year_options)

    market_options = sorted(df["market"].dropna().unique())
    selected_market = st.multiselect("Pilih Market", market_options, default=market_options)

    category_options = sorted(df["category"].dropna().unique())
    selected_category = st.multiselect("Pilih Kategori", category_options, default=category_options)

    segment_options = sorted(df["segment"].dropna().unique())
    selected_segment = st.multiselect("Pilih Segment", segment_options, default=segment_options)

    st.markdown("### Status Transaksi")
    show_profit = st.checkbox("Tampilkan transaksi Profit", value=True)
    show_loss = st.checkbox("Tampilkan transaksi Loss", value=True)


# =========================
# APPLY FILTER LOKAL
# =========================
filtered_df = df[
    (df["year"].isin(selected_year)) & 
    (df["market"].isin(selected_market)) & 
    (df["category"].isin(selected_category)) & 
    (df["segment"].isin(selected_segment))
]

if show_profit and not show_loss:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Profit"]
elif show_loss and not show_profit:
    filtered_df = filtered_df[filtered_df["profit_status"] == "Loss"]
elif not show_profit and not show_loss:
    filtered_df = filtered_df.iloc[0:0]


# =========================
# RENDER TABS UTAMA
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Ringkasan Penjualan", 
    "🤖 Klasifikasi ML (Profit/Loss)", 
    "🔮 Forecasting 2015", 
    "🗃️ Raw Data"
])

# ---------------------------------
# TAB 1: RINGKASAN PENJUALAN
# ---------------------------------
with tab1:
    if filtered_df.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih di sidebar.")
    else:
        # Menampilkan KPI
        total_sales = filtered_df["sales"].sum()
        total_profit = filtered_df["profit"].sum()
        total_quantity = filtered_df["quantity"].sum()
        total_orders = filtered_df["order_id"].nunique()
        profit_margin = (total_profit / total_sales * 100) if total_sales != 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Profit", f"${total_profit:,.2f}")
        col3.metric("Total Quantity", f"{total_quantity:,.0f} Pcs")
        col4.metric("Total Orders", f"{total_orders:,.0f}")
        col5.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        st.write("") # Spacer

        # Grafik dalam Grid 2x2
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            sales_by_year = filtered_df.groupby("year", as_index=False)["sales"].sum()
            fig_sales_year = px.bar(sales_by_year, x="year", y="sales", title="Total Sales Berdasarkan Tahun", color_discrete_sequence=[chart_colors[0]])
            fig_sales_year.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_sales_year, use_container_width=True)
            
            sales_by_region = filtered_df.groupby("region", as_index=False)["sales"].sum().sort_values('sales')
            fig_sales_region = px.bar(sales_by_region, x="sales", y="region", orientation='h', title="Total Sales Berdasarkan Region", color_discrete_sequence=[chart_colors[1]])
            fig_sales_region.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_sales_region, use_container_width=True)

        with g_col2:
            profit_by_category = filtered_df.groupby("category", as_index=False)["profit"].sum()
            fig_profit_category = px.pie(profit_by_category, values="profit", names="category", title="Proporsi Profit Berdasarkan Kategori", hole=0.4, color_discrete_sequence=chart_colors)
            fig_profit_category.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_profit_category, use_container_width=True)

            monthly_trend = filtered_df.groupby(["year", "month"], as_index=False).agg({"sales": "sum", "profit": "sum"})
            monthly_trend["period"] = monthly_trend["year"].astype(str) + "-" + monthly_trend["month"].astype(str).str.zfill(2)
            fig_monthly = px.line(monthly_trend, x="period", y=["sales", "profit"], markers=True, title="Tren Sales dan Profit per Bulan", color_discrete_sequence=[chart_colors[2], chart_colors[4]])
            fig_monthly.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_monthly, use_container_width=True)

        # Scatter Plot Full Width
        fig_discount_profit = px.scatter(filtered_df, x="discount", y="profit", color="profit_status", 
                                         title="Hubungan Tingkat Diskon terhadap Status Profit/Loss", 
                                         opacity=0.6, color_discrete_map=status_color_map)
        fig_discount_profit.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
        st.plotly_chart(fig_discount_profit, use_container_width=True)


# ---------------------------------
# TAB 2: KLASIFIKASI ML
# ---------------------------------
with tab2:
    st.subheader("Model Klasifikasi Random Forest")
    st.markdown("Algoritma Machine Learning dilatih menggunakan puluhan ribu data historis untuk **memprediksi apakah sebuah transaksi akan menghasilkan Keuntungan (Profit) atau Kerugian (Loss)** berdasarkan atribut seperti biaya pengiriman, diskon, dan kategori produk.")
    
    accuracy, cm, report = train_profit_loss_model(df)
    
    col_acc, col_space = st.columns([1, 3])
    with col_acc:
        st.metric("Tingkat Akurasi Prediksi", f"{accuracy * 100:.2f}%")

    ml_col1, ml_col2 = st.columns(2)
    
    with ml_col1:
        cm_df = pd.DataFrame(cm, index=["Actual Loss", "Actual Profit"], columns=["Predicted Loss", "Predicted Profit"])
        fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix", color_continuous_scale=cm_color_scale)
        fig_cm.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with ml_col2:
        st.markdown("**Classification Report Details:**")
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True, height=250)


# ---------------------------------
# TAB 3: FORECASTING 2015
# ---------------------------------
with tab3:
    st.subheader("Prediksi Penjualan & Keuntungan Tahun 2015")
    st.markdown("Data pada tab ini merupakan hasil pemodelan Machine Learning (Time Series / Regresi) untuk memproyeksikan target performa bisnis di tahun 2015 berdasarkan tren historis.")
    
    try:
        forecast_df = load_forecast()
        forecast_total_sales = forecast_df["predicted_sales"].sum()
        forecast_total_profit = forecast_df["predicted_profit"].sum()
        forecast_margin = (forecast_total_profit / forecast_total_sales * 100) if forecast_total_sales != 0 else 0

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("Prediksi Total Sales 2015", f"${forecast_total_sales:,.2f}")
        fcol2.metric("Prediksi Total Profit 2015", f"${forecast_total_profit:,.2f}")
        fcol3.metric("Estimasi Profit Margin", f"{forecast_margin:.2f}%")
        fcol4.metric("Status Data", "Berhasil Di-load ✓")

        st.write("")
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            fig_forecast_sales = px.line(forecast_df, x="order_date", y="predicted_sales", markers=True, title="Tren Proyeksi Sales 2015", line_shape="spline", color_discrete_sequence=[chart_colors[0]])
            fig_forecast_sales.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_forecast_sales, use_container_width=True)
        with fc_col2:
            fig_forecast_profit = px.line(forecast_df, x="order_date", y="predicted_profit", markers=True, title="Tren Proyeksi Profit 2015", line_shape="spline", color_discrete_sequence=[status_color_map["Profit"]])
            fig_forecast_profit.update_layout(template=plot_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
            st.plotly_chart(fig_forecast_profit, use_container_width=True)
            
    except FileNotFoundError:
        st.error("❌ **File forecast_2015.csv belum ditemukan!** Pastikan Anda telah menjalankan script `forecast_model.py` terlebih dahulu.")


# ---------------------------------
# TAB 4: RAW DATA
# ---------------------------------
with tab4:
    st.subheader("Database Transaksi")
    st.markdown("Menampilkan tabel data historis yang sudah difilter melalui panel sidebar. Anda bisa mengunduh tabel ini ke dalam format `.csv` dengan mengklik ikon di pojok kanan atas tabel.")
    st.dataframe(filtered_df, use_container_width=True, height=500)