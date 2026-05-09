import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="Dashboard Superstore",
    page_icon="",
    layout="wide"
)


# Styling dashboard
st.markdown("""
<style>
    :root {
        --bg-main: #061b33;
        --bg-panel: #0b2a4a;
        --bg-card: #0d355f;
        --border: rgba(79, 195, 247, 0.35);
        --text-main: #eaf6ff;
        --text-muted: #a9c7df;
        --blue: #38d5ff;
        --green: #3df5a0;
        --orange: #ffbd59;
        --red: #ff5d73;
    }

    .stApp {
        background: linear-gradient(135deg, #061b33 0%, #082a4f 50%, #03101f 100%);
        color: var(--text-main);
    }

    [data-testid="stSidebar"] {
        background: #041526;
        border-right: 1px solid var(--border);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, p, label, span, div {
        color: var(--text-main);
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: var(--text-muted);
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(13, 53, 95, 0.98), rgba(8, 35, 64, 0.98));
        border: 1px solid var(--border);
        border-left: 5px solid var(--blue);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
        min-height: 112px;
    }

    .kpi-label {
        color: var(--text-muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }

    .kpi-note {
        color: var(--text-muted);
        font-size: 0.76rem;
        margin-top: 0.45rem;
    }

    .section-card {
        background: rgba(8, 39, 72, 0.78);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
        margin-bottom: 1rem;
    }

    .small-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(13, 53, 95, 0.98), rgba(8, 35, 64, 0.98));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.20);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(13, 53, 95, 0.85);
        border-radius: 12px;
        padding: 10px 16px;
        border: 1px solid var(--border);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 213, 255, 0.20) !important;
        border: 1px solid rgba(56, 213, 255, 0.80) !important;
    }

    hr {
        border-color: rgba(79, 195, 247, 0.25);
    }
</style>
""", unsafe_allow_html=True)


def prepare_data(data):
    data = data.copy()
    data.columns = data.columns.str.strip()

    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")

    if "ship_date" in data.columns:
        data["ship_date"] = pd.to_datetime(data["ship_date"], errors="coerce")

    if "year" in data.columns:
        data["year"] = data["year"].astype(int)

    if "profit_status" not in data.columns and "profit" in data.columns:
        data["profit_status"] = np.where(data["profit"] > 0, "Profit", "Loss")

    if "month" not in data.columns:
        data["month"] = data["order_date"].dt.month

    if "month_name" not in data.columns:
        data["month_name"] = data["order_date"].dt.month_name()

    if "shipping_days" not in data.columns and {"ship_date", "order_date"}.issubset(data.columns):
        data["shipping_days"] = (data["ship_date"] - data["order_date"]).dt.days

    if "profit_margin" not in data.columns:
        data["profit_margin"] = np.where(data["sales"] != 0, data["profit"] / data["sales"], 0)

    return data


@st.cache_data
def load_default_data():
    data = pd.read_csv("cleaned_superstore.csv")
    return prepare_data(data)


@st.cache_data
def load_forecast():
    forecast = pd.read_csv("forecast_2015.csv")
    forecast["order_date"] = pd.to_datetime(forecast["order_date"], errors="coerce")
    forecast["month_name"] = forecast["order_date"].dt.month_name()
    return forecast


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
        "order_priority"
    ]

    existing_features = [col for col in features if col in data.columns]

    X = data[existing_features]
    y = data["profit_status"]

    numeric_features = [
        col for col in ["sales", "quantity", "discount", "shipping_cost", "shipping_days"]
        if col in existing_features
    ]

    categorical_features = [
        col for col in existing_features
        if col not in numeric_features
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
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )),
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

    labels = ["Loss", "Profit"]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(15)

    return accuracy, cm, report, importance_df


with st.sidebar:
    st.markdown("### Panel Filter")
    st.caption("Gunakan filter berikut untuk menyesuaikan data yang ditampilkan.")

    uploaded_file = st.file_uploader(
        "Upload data bersih CSV/XLSX",
        type=["csv", "xlsx"]
    )


if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df = prepare_data(df)
else:
    df = load_default_data()


st.markdown(
    '<div class="main-title">Dashboard Analitik Penjualan Superstore</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Visualisasi data penjualan, analisis profitabilitas, klasifikasi transaksi, dan prediksi performa tahun 2015.</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.divider()
    st.markdown("### Filter Data")

    year_options = sorted(df["year"].dropna().astype(int).unique())
    selected_year = st.multiselect(
        "Tahun",
        year_options,
        default=year_options
    )

    market_options = sorted(df["market"].dropna().unique())
    selected_market = st.multiselect(
        "Market",
        market_options,
        default=market_options
    )

    region_options = sorted(df["region"].dropna().unique())
    selected_region = st.multiselect(
        "Region",
        region_options,
        default=region_options
    )

    category_options = sorted(df["category"].dropna().unique())
    selected_category = st.multiselect(
        "Kategori",
        category_options,
        default=category_options
    )

    segment_options = sorted(df["segment"].dropna().unique())
    selected_segment = st.multiselect(
        "Segment",
        segment_options,
        default=segment_options
    )

    status_options = ["Profit", "Loss"]
    selected_status = st.multiselect(
        "Status Transaksi",
        status_options,
        default=status_options
    )

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    selected_date = st.date_input(
        "Rentang Tanggal",
        [min_date, max_date]
    )

    top_n = st.slider(
        "Jumlah Sub-Kategori Teratas",
        min_value=5,
        max_value=20,
        value=10
    )


filtered_df = df[
    (df["year"].astype(int).isin(selected_year)) &
    (df["market"].isin(selected_market)) &
    (df["region"].isin(selected_region)) &
    (df["category"].isin(selected_category)) &
    (df["segment"].isin(selected_segment)) &
    (df["profit_status"].isin(selected_status))
]

if isinstance(selected_date, (list, tuple)) and len(selected_date) == 2:
    start_date = pd.to_datetime(selected_date[0])
    end_date = pd.to_datetime(selected_date[1])

    filtered_df = filtered_df[
        (filtered_df["order_date"] >= start_date) &
        (filtered_df["order_date"] <= end_date)
    ]


if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()


total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
total_orders = filtered_df["order_id"].nunique()
total_quantity = filtered_df["quantity"].sum()
avg_discount = filtered_df["discount"].mean()
profit_margin = (total_profit / total_sales * 100) if total_sales != 0 else 0
loss_count = (filtered_df["profit_status"] == "Loss").sum()


def kpi_card(label, value, note, color):
    return f"""
    <div class="kpi-card" style="border-left-color:{color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.markdown(
    kpi_card("Total Penjualan", f"${total_sales:,.0f}", "Berdasarkan data terpilih", "#38d5ff"),
    unsafe_allow_html=True
)

kpi2.markdown(
    kpi_card("Total Profit", f"${total_profit:,.0f}", "Akumulasi keuntungan", "#3df5a0"),
    unsafe_allow_html=True
)

kpi3.markdown(
    kpi_card("Jumlah Order", f"{total_orders:,.0f}", "Order unik", "#ffbd59"),
    unsafe_allow_html=True
)

kpi4.markdown(
    kpi_card("Jumlah Produk", f"{total_quantity:,.0f}", "Total item terjual", "#9d7cff"),
    unsafe_allow_html=True
)

kpi5.markdown(
    kpi_card("Margin Profit", f"{profit_margin:.2f}%", "Profit terhadap penjualan", "#4de3c1"),
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


tab_overview, tab_ml, tab_forecast, tab_data = st.tabs([
    "Ringkasan Data",
    "Model Klasifikasi",
    "Prediksi 2015",
    "Tabel Data"
])


with tab_overview:
    row1_col1, row1_col2, row1_col3 = st.columns([1.2, 2.2, 1.2])

    with row1_col1:
        st.markdown(
            '<div class="section-card"><div class="small-title">Komposisi Transaksi</div>',
            unsafe_allow_html=True
        )

        status_count = filtered_df["profit_status"].value_counts().reset_index()
        status_count.columns = ["profit_status", "count"]

        fig_status = px.pie(
            status_count,
            names="profit_status",
            values="count",
            hole=0.62,
            color="profit_status",
            color_discrete_map={
                "Profit": "#3df5a0",
                "Loss": "#ff5d73"
            },
            title="Transaksi Profit dan Loss"
        )

        fig_status.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            margin=dict(t=45, b=15, l=15, r=15),
            height=310
        )

        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-card"><div class="small-title">Perbandingan Kategori</div>',
            unsafe_allow_html=True
        )

        radar_df = filtered_df.groupby("category", as_index=False).agg({
            "sales": "sum",
            "profit": "sum"
        })

        radar_categories = radar_df["category"].tolist()
        radar_values = radar_df["sales"].tolist()

        if radar_categories:
            radar_categories += [radar_categories[0]]
            radar_values += [radar_values[0]]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill="toself",
            name="Sales",
            line=dict(color="#38d5ff")
        ))

        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True)
            ),
            height=330,
            margin=dict(t=25, b=25, l=25, r=25),
            showlegend=False
        )

        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown(
            '<div class="section-card"><div class="small-title">Tren Penjualan dan Profit</div>',
            unsafe_allow_html=True
        )

        monthly_trend = (
            filtered_df.groupby(["year", "month"], as_index=False)
            .agg({"sales": "sum", "profit": "sum"})
            .sort_values(["year", "month"])
        )

        monthly_trend["period"] = (
            monthly_trend["year"].astype(str) + "-" +
            monthly_trend["month"].astype(str).str.zfill(2)
        )

        fig_trend = px.line(
            monthly_trend,
            x="period",
            y=["sales", "profit"],
            markers=True,
            title="Tren Sales dan Profit per Bulan",
            color_discrete_sequence=["#38d5ff", "#3df5a0"]
        )

        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            height=390,
            margin=dict(t=55, b=35, l=25, r=25),
            legend_title_text="Metrik"
        )

        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

        with gauge_col1:
            fig_margin = go.Figure(go.Indicator(
                mode="gauge+number",
                value=profit_margin,
                number={"suffix": "%", "font": {"color": "#eaf6ff"}},
                title={"text": "Margin Profit", "font": {"color": "#eaf6ff"}},
                gauge={
                    "axis": {
                        "range": [None, max(50, profit_margin + 10)],
                        "tickcolor": "#eaf6ff"
                    },
                    "bar": {"color": "#3df5a0"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(79,195,247,0.35)"
                }
            ))

            fig_margin.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=30, b=15, l=15, r=15)
            )

            st.plotly_chart(fig_margin, use_container_width=True)

        with gauge_col2:
            fig_discount = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_discount * 100,
                number={"suffix": "%", "font": {"color": "#eaf6ff"}},
                title={"text": "Rata-Rata Diskon", "font": {"color": "#eaf6ff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#eaf6ff"},
                    "bar": {"color": "#ffbd59"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(79,195,247,0.35)"
                }
            ))

            fig_discount.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=30, b=15, l=15, r=15)
            )

            st.plotly_chart(fig_discount, use_container_width=True)

        with gauge_col3:
            loss_rate = loss_count / len(filtered_df) * 100 if len(filtered_df) else 0

            fig_loss = go.Figure(go.Indicator(
                mode="gauge+number",
                value=loss_rate,
                number={"suffix": "%", "font": {"color": "#eaf6ff"}},
                title={"text": "Rasio Loss", "font": {"color": "#eaf6ff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#eaf6ff"},
                    "bar": {"color": "#ff5d73"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(79,195,247,0.35)"
                }
            ))

            fig_loss.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=30, b=15, l=15, r=15)
            )

            st.plotly_chart(fig_loss, use_container_width=True)

    with row1_col3:
        st.markdown(
            '<div class="section-card"><div class="small-title">Penjualan Berdasarkan Region</div>',
            unsafe_allow_html=True
        )

        region_sales = (
            filtered_df.groupby("region", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=True)
        )

        fig_region = px.bar(
            region_sales,
            x="sales",
            y="region",
            orientation="h",
            color="sales",
            color_continuous_scale="Blues",
            title="Total Sales per Region"
        )

        fig_region.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            height=370,
            margin=dict(t=50, b=25, l=15, r=15),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_region, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-card"><div class="small-title">Sub-Kategori Teratas</div>',
            unsafe_allow_html=True
        )

        top_sub = (
            filtered_df.groupby("sub_category", as_index=False)["profit"]
            .sum()
            .sort_values("profit", ascending=False)
            .head(top_n)
        )

        fig_top_sub = px.bar(
            top_sub,
            x="profit",
            y="sub_category",
            orientation="h",
            color="profit",
            color_continuous_scale="Teal",
            title=f"{top_n} Sub-Kategori dengan Profit Tertinggi"
        )

        fig_top_sub.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            height=350,
            margin=dict(t=50, b=25, l=15, r=15),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_top_sub, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-card"><div class="small-title">Hubungan Diskon terhadap Profit</div>',
        unsafe_allow_html=True
    )

    fig_scatter = px.scatter(
        filtered_df,
        x="discount",
        y="profit",
        color="profit_status",
        size="sales",
        hover_data=["category", "sub_category", "market", "region"],
        title="Pola Diskon dan Profit",
        opacity=0.62,
        color_discrete_map={
            "Profit": "#3df5a0",
            "Loss": "#ff5d73"
        }
    )

    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eaf6ff",
        height=430,
        margin=dict(t=55, b=30, l=30, r=30)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab_ml:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Klasifikasi Transaksi Profit dan Loss")

    st.write(
        "Model Random Forest digunakan untuk mengklasifikasikan transaksi ke dalam kategori Profit atau Loss. "
        "Fitur yang digunakan meliputi sales, quantity, discount, shipping cost, kategori produk, wilayah, dan metode pengiriman."
    )

    accuracy, cm, report, importance_df = train_profit_loss_model(df)

    ml_kpi1, ml_kpi2, ml_kpi3 = st.columns(3)

    ml_kpi1.metric("Akurasi", f"{accuracy * 100:.2f}%")
    ml_kpi2.metric("Jumlah Data", f"{len(df):,}")
    ml_kpi3.metric("Kelas Target", "Profit / Loss")

    cm_col, report_col = st.columns([1, 1.2])

    with cm_col:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Loss", "Actual Profit"],
            columns=["Predicted Loss", "Predicted Profit"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )

        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            height=430
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    with report_col:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True, height=420)

    st.divider()

    st.subheader("Faktor yang Paling Berpengaruh")

    st.write(
        "Bagian ini menunjukkan variabel yang paling banyak digunakan model dalam membedakan transaksi Profit dan Loss. "
        "Nilai importance yang lebih tinggi menunjukkan kontribusi variabel yang lebih besar terhadap hasil klasifikasi."
    )

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="15 Faktor Teratas dalam Klasifikasi Profit/Loss",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig_importance.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eaf6ff",
        height=520,
        yaxis=dict(autorange="reversed"),
        margin=dict(t=55, b=30, l=30, r=30),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


with tab_forecast:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Prediksi Sales dan Profit Tahun 2015")

    st.write(
        "Prediksi tahun 2015 dibuat menggunakan model regresi sederhana berbasis fitur waktu. "
        "Data historis tahun 2011 sampai 2014 diringkas menjadi data bulanan sebelum digunakan dalam pemodelan."
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

        fc1, fc2, fc3 = st.columns(3)

        fc1.metric("Estimasi Sales 2015", f"${forecast_total_sales:,.2f}")
        fc2.metric("Estimasi Profit 2015", f"${forecast_total_profit:,.2f}")
        fc3.metric("Estimasi Margin", f"{forecast_margin:.2f}%")

        forecast_col1, forecast_col2 = st.columns(2)

        with forecast_col1:
            fig_fc_sales = px.line(
                forecast_df,
                x="order_date",
                y="predicted_sales",
                markers=True,
                title="Prediksi Sales Bulanan 2015",
                color_discrete_sequence=["#38d5ff"]
            )

            fig_fc_sales.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
                height=400
            )

            st.plotly_chart(fig_fc_sales, use_container_width=True)

        with forecast_col2:
            fig_fc_profit = px.line(
                forecast_df,
                x="order_date",
                y="predicted_profit",
                markers=True,
                title="Prediksi Profit Bulanan 2015",
                color_discrete_sequence=["#3df5a0"]
            )

            fig_fc_profit.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
                height=400
            )

            st.plotly_chart(fig_fc_profit, use_container_width=True)

        st.subheader("Tabel Prediksi 2015")
        st.dataframe(forecast_df, use_container_width=True)

    except FileNotFoundError:
        st.error("File forecast_2015.csv belum ditemukan. Jalankan dulu file forecast_model.py.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_data:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Data Transaksi")
    st.write("Tabel berikut menampilkan data transaksi sesuai filter yang dipilih.")

    st.dataframe(filtered_df, use_container_width=True, height=520)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Data Hasil Filter",
        data=csv,
        file_name="filtered_superstore_data.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)