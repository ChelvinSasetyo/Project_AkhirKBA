# Project Akhir KBA - Dashboard Analitik Superstore

Project ini merupakan sistem dashboard analitik penjualan Superstore berbasis Python dan Streamlit. Sistem ini digunakan untuk menampilkan visualisasi penjualan, analisis profit, klasifikasi transaksi Profit/Loss menggunakan machine learning, serta prediksi sales dan profit tahun 2015.

## Fitur Sistem
- Import dan cleaning data Superstore
- Dashboard interaktif dengan filter tahun, market, region, kategori, segment, dan status transaksi
- Visualisasi total sales, profit, order, quantity, dan margin profit
- Grafik tren sales dan profit
- Analisis hubungan discount terhadap profit
- Machine learning klasifikasi Profit/Loss menggunakan Random Forest
- Forecasting sales dan profit tahun 2015
- Download data hasil filter

## Teknologi
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## Cara Menjalankan Project
1. Install library:
   pip install -r requirements.txt

2. Jalankan data cleaning:
   py clean_data.py

3. Jalankan forecasting:
   py forecast_model.py

4. Jalankan dashboard:
   py -m streamlit run app.py