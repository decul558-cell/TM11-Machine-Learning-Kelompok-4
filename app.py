
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Computer Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = "computer-prices-2025/computer_prices_all.csv"
MODEL_PATH = "models/best_model.pkl"
RESULT_PATH = "hasil/perbandingan_model.csv"
METADATA_PATH = "models/model_metadata.json"


# =====================================================
# 2. MEMUAT MODEL DAN DATASET
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULT_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


# Memeriksa ketersediaan file
required_files = [
    DATASET_PATH,
    MODEL_PATH,
    RESULT_PATH,
    METADATA_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error("File aplikasi belum lengkap:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

try:
    model = load_model()
    df = load_dataset()
    results = load_results()
    metadata = load_metadata()

except Exception as error:
    st.error(f"Gagal memuat aplikasi: {error}")
    st.stop()


# =====================================================
# 3. MENYIAPKAN DATA
# =====================================================

X = df.drop(
    columns=["ID", "price"],
    errors="ignore"
)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

best_model_name = metadata.get(
    "best_model",
    "Gradient Boosting"
)


# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("💻 Computer AI")

    st.caption("Machine Learning Price Prediction")

    st.divider()

    page = st.radio(
        "MENU UTAMA",
        [
            "Dashboard",
            "Prediksi Harga",
            "Evaluasi Model",
            "Tentang Aplikasi"
        ]
    )

    st.divider()

    st.success("Model: " + best_model_name)

    st.caption(
        "TM11 Machine Learning | Kelompok 4"
    )


# =====================================================
# 5. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("💻 Computer Price Predictor")

    st.write(
        "Aplikasi prediksi harga komputer menggunakan "
        "algoritma Machine Learning."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Asli",
        f"{len(df):,}".replace(",", ".")
    )

    col2.metric(
        "Jumlah Fitur",
        len(X.columns)
    )

    col3.metric(
        "Model Terbaik",
        best_model_name
    )

    best_result = results[
        results["Model"] == best_model_name
    ].iloc[0]

    col4.metric(
        "R² Score",
        f"{best_result['R2']:.4f}"
    )

    st.divider()

    st.subheader("Perbandingan Performa Model")

    chart_data = results.set_index("Model")[
        ["MAE", "RMSE"]
    ]

    st.bar_chart(chart_data)

    st.subheader("Ringkasan Dataset")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Contoh data komputer")
        st.dataframe(
            df.drop(columns=["ID"], errors="ignore").head(10),
            use_container_width=True
        )

    with c2:
        st.write("Informasi dataset")
        st.write(f"Jumlah data asli: {len(df):,}")
        st.write(f"Jumlah fitur: {len(X.columns)}")
        st.write(
            f"Jumlah fitur numerik: {len(numeric_features)}"
        )
        st.write(
            f"Jumlah fitur kategorikal: {len(categorical_features)}"
        )
        st.write(
            f"Missing values: {df.isnull().sum().sum()}"
        )


# =====================================================
# 6. PREDIKSI HARGA
# =====================================================

elif page == "Prediksi Harga":

    st.title("🔍 Prediksi Harga Komputer")

    st.write(
        "Masukkan spesifikasi komputer untuk mendapatkan "
        "estimasi harga berdasarkan model Gradient Boosting."
    )

    st.info(
        "Silakan isi spesifikasi komputer pada formulir berikut."
    )

    with st.form("prediction_form"):

        user_input = {}

        # ---------------- NUMERIK ----------------

        st.subheader("Spesifikasi Hardware")

        numeric_columns = st.columns(2)

        for index, col in enumerate(numeric_features):

            series = df[col].dropna()

            if series.empty:
                default_value = 0.0
                minimum = 0.0
                maximum = 1000.0
            else:
                default_value = float(series.median())
                minimum = float(series.min())
                maximum = float(series.max())

            with numeric_columns[index % 2]:

                if pd.api.types.is_integer_dtype(df[col]):
                    user_input[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default_value),
                        step=1
                    )
                else:
                    user_input[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default_value,
                        step=0.1,
                        format="%.2f"
                    )

        # ---------------- KATEGORIKAL ----------------

        st.divider()

        st.subheader("Spesifikasi Kategori")

        categorical_columns = st.columns(2)

        for index, col in enumerate(categorical_features):

            options = sorted(
                df[col].dropna().unique().tolist(),
                key=str
            )

            if len(options) == 0:
                options = ["Unknown"]

            with categorical_columns[index % 2]:

                user_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=options,
                    format_func=lambda value: str(value)
                )

        st.divider()

        submitted = st.form_submit_button(
            "🔮 PREDIKSI HARGA SEKARANG",
            use_container_width=True
        )

    # ---------------- HASIL PREDIKSI ----------------

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=X.columns
        )

        try:

            prediction = model.predict(input_df)[0]

            st.divider()

            st.success("Prediksi harga berhasil dilakukan!")

            st.subheader("Hasil Prediksi Harga")

            st.metric(
                "Estimasi Harga Komputer",
                f"{prediction:,.2f}"
            )

            st.caption(
                "Nilai mengikuti satuan harga pada dataset."
            )

            st.info(
                f"Model yang digunakan: {best_model_name}"
            )

            st.subheader("Spesifikasi Komputer")

            st.dataframe(
                input_df.T.rename(
                    columns={0: "Spesifikasi"}
                ),
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Terjadi kesalahan saat prediksi: {error}"
            )


# =====================================================
# 7. EVALUASI MODEL
# =====================================================

elif page == "Evaluasi Model":

    st.title("📊 Evaluasi dan Perbandingan Model")

    st.write(
        "Perbandingan hasil pengujian algoritma "
        "AdaBoost dan Gradient Boosting."
    )

    st.divider()

    st.subheader("Tabel Evaluasi")

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    metric = st.selectbox(
        "Pilih metrik evaluasi",
        ["MAE", "MAPE", "MSE", "RMSE", "R2"]
    )

    st.subheader(f"Perbandingan {metric}")

    chart = results.set_index("Model")[[metric]]

    st.bar_chart(chart)

    st.divider()

    st.subheader("Interpretasi Hasil")

    st.write(
        "**MAE, MAPE, MSE, dan RMSE:** "
        "semakin kecil nilainya semakin baik."
    )

    st.write(
        "**R²:** semakin mendekati 1 semakin baik."
    )

    st.success(
        f"Model terbaik berdasarkan RMSE adalah "
        f"{best_model_name}."
    )

    st.write(
        f"Nilai RMSE: {best_result['RMSE']:.4f}"
    )


# =====================================================
# 8. TENTANG APLIKASI
# =====================================================

elif page == "Tentang Aplikasi":

    st.title("ℹ️ Tentang Aplikasi")

    st.write(
        """
        Computer Price Predictor merupakan aplikasi
        Machine Learning untuk memperkirakan harga
        komputer berdasarkan spesifikasi hardware
        dan karakteristik perangkat.
        """
    )

    st.subheader("Algoritma yang Digunakan")

    st.markdown(
        """
        **1. AdaBoost (Adaptive Boosting)**

        Algoritma ensemble yang menggabungkan beberapa
        weak learner untuk meningkatkan kemampuan prediksi.

        **2. Gradient Boosting**

        Algoritma ensemble yang membangun model secara
        bertahap dengan memperbaiki kesalahan prediksi
        dari model sebelumnya.
        """
    )

    st.subheader("Informasi Dataset")

    st.write("Dataset: Computer Prices 2025")
    st.write("Jumlah data asli: 100.000")
    st.write("Jumlah fitur: 32")
    st.write("Target prediksi: price")

    st.write(
        f"Data eksperimen: {metadata['sample_size']:,} baris"
    )

    st.write(
        f"Data training: {metadata['training_size']:,} baris"
    )

    st.write(
        f"Data testing: {metadata['testing_size']:,} baris"
    )

    st.divider()

    st.caption(
        "Tugas TM11 Machine Learning | Kelompok 4"
    )


# =====================================================
# 9. FOOTER
# =====================================================

st.divider()

st.caption(
    "Computer Price Predictor | Machine Learning TM11 | Kelompok 4"
)
