import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf

# Sidebar content
with st.sidebar:
    st.image("image.png")
    st.title("Prediksi Harga Saham")
    
    # Menambahkan opsi untuk upload file atau ambil data dari Yahoo Finance
    data_source = st.radio("Pilih sumber data:", ["Upload File CSV", "Ambil Data dari Yahoo Finance"])
    
    if data_source == "Upload File CSV":
        # Upload file jika memilih opsi ini
        file = st.file_uploader("Upload your input csv file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            # Menyimpan file CSV yang di-upload
            csv_data = df.to_csv().encode('utf-8')
    else:
        # Ambil data dari Yahoo Finance jika memilih opsi ini
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
                # Menyimpan data yang diambil dari Yahoo Finance sebagai CSV
                csv_data = df.to_csv().encode('utf-8')

    # Pilihan menu untuk analisis
    choice = st.radio("Daftar Menu", ["Data Detail", "Data Model", "Prediksi"])
    st.info("Aplikasi ini digunakan untuk memprediksi harga saham dengan time frame daily")

# Pastikan dataframe ada sebelum melanjutkan ke menu lain
if 'df' in locals() and not df.empty:
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Data Detail Menu
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
            data=csv_data,  # Gunakan data CSV yang disimpan
            file_name='stock_data.csv',
            mime='text/csv',
        )

    # Data Model Menu
    elif choice == "Data Model":
        st.title("Triple Exponential Smoothing Model")
        # Membangun model Triple Exponential Smoothing
        model = ExponentialSmoothing(train_df['Close'], trend='add', seasonal='add', seasonal_periods=155).fit()

        # Generate predictions for the test set
        date = len(test_df)  # Number of days to forecast based on the length of the test set
        pred = model.forecast(date)
        pred = pd.DataFrame(pred, columns=['Predicted Close'])
        pred.index = test_df.index

        combined = pd.concat([test_df['Close'], pred], axis=1)

        st.info("Data Actual vs Prediction")

        # Plot the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        train_df['Close'].plot(style='--', color='gray', legend=True, label='Training Data', ax=ax)
        test_df['Close'].plot(style='--', color='r', legend=True, label='Test Data', ax=ax)
        pred['Predicted Close'].plot(color='b', legend=True, label='Prediction', ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_title("Stock Price Prediction")
        st.pyplot(fig)

        # Display the table
        st.dataframe(combined)

        # Calculate and display MAPE
        st.info("Evaluasi Model Data Actual 80% (Train) dan Data Predict 20% (Test)")
        mape = [
            {"MAPE": "<10%", "Kategori": "Performa model prediksi akurat"},
            {"MAPE": ">10%-20%", "Kategori": "Performa model prediksi baik"},
            {"MAPE": ">20%-50%", "Kategori": "Performa model prediksi layak"},
            {"MAPE": ">50%", "Kategori": "Performa model prediksi tidak akurat"}
        ]

        st.table(mape)
        test_mape = mean_absolute_percentage_error(test_df['Close'], pred['Predicted Close'])
        st.write('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(test_mape * 100))

    # Prediksi Menu
    elif choice == "Prediksi":
        st.title("Prediksi Harga Saham")
        date = st.slider("Pilih jumlah hari prediksi", 1, 150, step=1)

        # Forecast
        if st.button("Predict"):
            # Train a Triple Exponential Smoothing model on the entire data
            model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=155).fit()

            # Generate predictions
            pred = model.forecast(date)
            pred = pd.DataFrame(pred, columns=['Close'])
            pred.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=date, freq='D')

            st.info("Hasil Prediksi")

            # Plot the chart
            fig, ax = plt.subplots()
            df['Close'].plot(style='--', color='gray', legend=True, label='Known', ax=ax)
            pred['Close'].plot(color='b', legend=True, label='Prediksi', ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            st.pyplot(fig)

            # Display the table
            st.dataframe(pred)
            
            # Evaluasi model berdasarkan jumlah data aktual dan jumlah periode data prediksi
            if date <= len(df):  # Jika periode prediksi dalam rentang data yang tersedia
                st.info("Evaluasi Model")
                df_test = df['Close'][-date:]  # Mengambil data aktual pada periode prediksi
                pred_test = pred['Close'][:len(df_test)]  # Sesuaikan prediksi dengan periode yang sama
                test_mape = mean_absolute_percentage_error(df_test, pred_test)
                st.write('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(test_mape * 100))
            else:
                st.warning("Tidak dapat menghitung MAPE karena prediksi melebihi data yang tersedia")
else:
    st.write("Silahkan upload file CSV atau masukkan kode saham dan tanggal untuk memulai analisis.")
