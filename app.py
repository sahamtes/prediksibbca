import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf

# Sidebar untuk memilih sumber data
with st.sidebar:
    st.image("image.png")
    st.title("Prediksi Harga Saham")
    
    # Menambahkan opsi untuk upload file atau ambil data dari Yahoo Finance
    data_source = st.radio("Pilih sumber data:", ["Upload File CSV", "Ambil Data dari Yahoo Finance"])
    
    if data_source == "Upload File CSV":
        file = st.file_uploader("Upload file CSV Anda", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            csv_data = df.to_csv().encode('utf-8')
    else:
        ticker = st.text_input("Masukkan kode saham (misal: TLKM.JK, PTBA.JK, BBRI.JK): ").upper()
        start_date = st.date_input("Pilih tanggal mulai:")
        end_date = st.date_input("Pilih tanggal akhir:")

        if ticker and start_date and end_date:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.error(f"Tidak ada data yang tersedia untuk {ticker} dalam rentang tanggal yang dipilih.")
            else:
                df['Date'] = pd.to_datetime(df.index)
                df.set_index('Date', inplace=True)
                csv_data = df.to_csv().encode('utf-8')

    choice = st.radio("Daftar Menu", ["Data Detail", "Data Model", "Prediksi"])
    st.info("Aplikasi ini digunakan untuk memprediksi harga saham dengan time frame daily")

# Jika data berhasil diambil atau diunggah
if 'df' in locals() and not df.empty:
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Menu "Data Detail" untuk melihat data yang digunakan
    if choice == "Data Detail":
        st.title("Data Detail")
        st.info("Data Head")
        st.dataframe(df.head())
        st.info("Data Tail")
        st.dataframe(df.tail())
        st.info("Data Describe")
        st.write(df.describe())
        
        # Download data yang digunakan
        st.download_button(
            label="Download sebagai CSV",
            data=csv_data,
            file_name='stock_data.csv',
            mime='text/csv',
        )

    # Menu "Data Model" untuk melatih model dan melihat hasil perbandingan
    elif choice == "Data Model":
        st.title("Model Prediksi Harga Saham Triple Exponential Smoothing (TES)")

        # Model Triple Exponential Smoothing (TES)
        tes_model = ExponentialSmoothing(train_df['Close'], trend='add', seasonal='add', seasonal_periods=155).fit()
        tes_pred = tes_model.forecast(len(test_df))
        tes_pred = pd.DataFrame(tes_pred, columns=['Predicted Close TES'])
        tes_pred.index = test_df.index
        
        # Display tabel perbandingan data aktual dan prediksi
        st.subheader("Perbandingan Data Actual dan Prediksi TES")
        
        # Plot untuk Triple Exponential Smoothing
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        train_df['Close'].plot(style='--', color='gray', legend=True, label='Training Data', ax=ax1)
        test_df['Close'].plot(style='--', color='red', legend=True, label='Test Data', ax=ax1)
        tes_pred['Predicted Close TES'].plot(color='blue', legend=True, label='TES Prediction', ax=ax1)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price")
        ax1.set_title("Prediksi Harga Saham - TES")
        st.pyplot(fig1)
        
        # Gabungkan data actual dan prediksi TES ke dalam satu tabel
        comparison_df = pd.concat([test_df['Close'], tes_pred], axis=1)
        comparison_df.columns = ['Actual Close', 'Predicted Close TES']     
        st.dataframe(comparison_df)
        
        # MAPE untuk TES
        st.info("Evaluasi Model Data Actual 80% (Train) dan Data Predict 20% (Test)")
        mape = [
            {"MAPE": "<10%", "Kategori": "Performa model prediksi akurat"},
            {"MAPE": ">10%-20%", "Kategori": "Performa model prediksi baik"},
            {"MAPE": ">20%-50%", "Kategori": "Performa model prediksi layak"},
            {"MAPE": ">50%", "Kategori": "Performa model prediksi tidak akurat"}
        ]
        st.table(mape)
        tes_mape = mean_absolute_percentage_error(test_df['Close'], tes_pred['Predicted Close TES'])
        st.write('TES Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(tes_mape * 100))

    # Menu "Prediksi" untuk melihat prediksi masa depan
    elif choice == "Prediksi":
        st.title("Prediksi Harga Saham")
        
        # Pilih jumlah hari untuk prediksi
        date = st.slider("Pilih jumlah hari prediksi", 1, 150, step=1)

        if st.button("Predict"):
            # Prediksi dengan Triple Exponential Smoothing (TES)
            tes_model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=155).fit()
            tes_pred = tes_model.forecast(date)
            tes_pred_df = pd.DataFrame(tes_pred, columns=['Predicted Close TES'])
            tes_pred_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=date, freq='D')

            # Tampilan hasil prediksi TES
            st.subheader("Hasil Prediksi Menggunakan Triple Exponential Smoothing (TES)")
            fig1, ax1 = plt.subplots()
            df['Close'].plot(style='--', color='gray', legend=True, label='Data Aktual', ax=ax1)
            tes_pred_df['Predicted Close TES'].plot(color='blue', legend=True, label='Prediksi TES', ax=ax1)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Close Price")
            st.pyplot(fig1)
            
            st.dataframe(tes_pred_df)
            
            # Evaluasi model berdasarkan jumlah data aktual dan jumlah data prediksi
            st.info("Evaluasi Model Berdasarkan Jumlah Data Aktual dan Jumlah Data Prediksi")
            mape = [
            {"MAPE": "<10%", "Kategori": "Performa model prediksi akurat"},
            {"MAPE": ">10%-20%", "Kategori": "Performa model prediksi baik"},
            {"MAPE": ">20%-50%", "Kategori": "Performa model prediksi layak"},
            {"MAPE": ">50%", "Kategori": "Performa model prediksi tidak akurat"}
            ]
            st.table(mape)
            if date <= len(df):  # Jika periode prediksi dalam rentang data yang tersedia
                df_test = df['Close'][-date:]  # Mengambil data aktual pada periode prediksi
                tes_pred_test = tes_pred[:len(df_test)]  # Sesuaikan prediksi dengan periode yang sama
                test_mape = mean_absolute_percentage_error(df_test, tes_pred_test)
                st.write('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(test_mape * 100))
            else:
                st.warning("Tidak dapat menghitung MAPE karena prediksi melebihi data yang tersedia")
            
else:
    st.write("Silahkan upload file CSV atau masukkan kode saham dan tanggal untuk memulai analisis.")
