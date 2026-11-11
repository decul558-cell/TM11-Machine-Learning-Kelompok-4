
# ============================================================
# TUGAS TM11 MACHINE LEARNING
# PREDIKSI HARGA KOMPUTER
# METODE:
# 1. ADABOOST (ADAPTIVE BOOSTING)
# 2. GRADIENT BOOSTING
# KELOMPOK 4
# ============================================================

import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PENGATURAN PROJECT
# ============================================================

DATASET_PATH = (
    "computer-prices-2025/computer_prices_all.csv"
)

SAMPLE_SIZE = 25000
RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("hasil", exist_ok=True)

print("\n" + "=" * 65)
print("       PROGRAM PREDIKSI HARGA KOMPUTER")
print("       ADABOOST DAN GRADIENT BOOSTING")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

print("\n[1] MEMBACA DATASET")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("Dataset berhasil dibaca!")
print("Jumlah data asli :", len(df))
print("Jumlah kolom     :", len(df.columns))
print("Missing values   :", df.isnull().sum().sum())


# ============================================================
# 3. PENGAMBILAN SAMPEL DATA
# ============================================================

print("\n[2] PENGAMBILAN SAMPEL DATA")

jumlah_sampel = min(SAMPLE_SIZE, len(df))

df = df.sample(
    n=jumlah_sampel,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("Jumlah sampel yang digunakan:", len(df))


# ============================================================
# 4. MENENTUKAN FITUR DAN TARGET
# ============================================================

print("\n[3] MENENTUKAN FITUR DAN TARGET")

if "price" not in df.columns:
    raise ValueError("Kolom target 'price' tidak ditemukan!")

# Menghapus kolom ID karena hanya identitas data
X = df.drop(columns=["ID", "price"], errors="ignore")

# Target prediksi
y = df["price"]

print("Jumlah fitur:", X.shape[1])
print("Target      : price")

# Memisahkan fitur numerik dan kategorikal
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Fitur numerik     :", len(numeric_features))
print("Fitur kategorikal :", len(categorical_features))


# ============================================================
# 5. PREPROCESSING DATA
# ============================================================

print("\n[4] PREPROCESSING DATA")

# Preprocessing data numerik
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# Preprocessing data kategorikal
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)

# Menggabungkan preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("Preprocessing berhasil disiapkan!")


# ============================================================
# 6. MEMBAGI DATA TRAINING DAN TESTING
# ============================================================

print("\n[5] PEMBAGIAN DATA TRAINING DAN TESTING")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Total data    :", len(df))
print("Data training :", len(X_train))
print("Data testing  :", len(X_test))


# ============================================================
# 7. MEMBUAT MODEL MACHINE LEARNING
# ============================================================

print("\n[6] MEMBUAT MODEL MACHINE LEARNING")

models = {

    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=2,
            random_state=RANDOM_STATE
        ),
        n_estimators=30,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=RANDOM_STATE
    )

}

print("Model AdaBoost berhasil dibuat!")
print("Model Gradient Boosting berhasil dibuat!")


# ============================================================
# 8. PROSES TRAINING, TESTING, DAN EVALUASI
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 65)
    print("MEMPROSES MODEL:", name)
    print("=" * 65)

    # Membuat pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # ---------------- TRAINING ----------------

    print("\nMemulai proses training...")

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print("Training selesai!")
    print(
        f"Waktu training: {training_time:.2f} detik"
    )

    # ---------------- TESTING ----------------

    print("\nMemulai proses testing...")

    y_pred = pipeline.predict(X_test)

    print("Testing selesai!")

    # ---------------- EVALUASI ----------------

    print("\nMenghitung evaluasi model...")

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    ) * 100

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nHASIL EVALUASI", name)
    print("-" * 45)
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.4f}%")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # Menyimpan hasil evaluasi
    results.append({
        "Model": name,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Waktu_Training": training_time
    })

    trained_models[name] = pipeline

    # Menyimpan model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        pipeline,
        f"models/{filename}.pkl"
    )

    print("\nModel berhasil disimpan!")


    # Menyimpan hasil prediksi
    prediction_df = pd.DataFrame({

        "Harga_Aktual": y_test.values,

        "Harga_Prediksi": y_pred,

        "Error": y_test.values - y_pred

    })

    prediction_df.to_csv(
        f"hasil/prediksi_{filename}.csv",
        index=False
    )

    print("Hasil prediksi berhasil disimpan!")


# ============================================================
# 9. PERBANDINGAN KEDUA MODEL
# ============================================================

print("\n" + "=" * 65)
print("       PERBANDINGAN HASIL EVALUASI MODEL")
print("=" * 65)

results_df = pd.DataFrame(results)

print(
    results_df.round(4).to_string(index=False)
)

# Menyimpan tabel evaluasi
results_df.to_csv(
    "hasil/perbandingan_model.csv",
    index=False
)


# ============================================================
# 10. MENENTUKAN MODEL TERBAIK
# ============================================================

print("\n" + "=" * 65)
print("       PENENTUAN MODEL TERBAIK")
print("=" * 65)

# RMSE terendah menjadi kriteria utama
best_index = results_df["RMSE"].idxmin()

best_model_name = results_df.loc[
    best_index, "Model"
]

best_model = trained_models[best_model_name]

# Menyimpan model terbaik
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("Model terbaik:", best_model_name)

print(
    "Nilai RMSE   :",
    round(results_df.loc[best_index, "RMSE"], 4)
)

print(
    "Nilai MAE    :",
    round(results_df.loc[best_index, "MAE"], 4)
)

print(
    "Nilai MAPE   :",
    round(results_df.loc[best_index, "MAPE"], 4), "%"
)

print(
    "Nilai R2     :",
    round(results_df.loc[best_index, "R2"], 4)
)


# ============================================================
# 11. MENYIMPAN INFORMASI MODEL
# ============================================================

metadata = {
    "best_model": best_model_name,
    "dataset": DATASET_PATH,
    "sample_size": len(df),
    "training_size": len(X_train),
    "testing_size": len(X_test),
    "target": "price",
    "features": X.columns.tolist(),
    "evaluation": results_df.to_dict(
        orient="records"
    )
}

with open(
    "models/model_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 12. PROGRAM SELESAI
# ============================================================

print("\n" + "=" * 65)
print("       PROGRAM MACHINE LEARNING SELESAI")
print("=" * 65)

print("\nFile yang berhasil dibuat:")

print("1. models/adaboost.pkl")
print("2. models/gradient_boosting.pkl")
print("3. models/best_model.pkl")
print("4. models/model_metadata.json")
print("5. hasil/perbandingan_model.csv")
print("6. hasil/prediksi_adaboost.csv")
print("7. hasil/prediksi_gradient_boosting.csv")

print("\nModel terbaik:", best_model_name)

print("\nTerima kasih!")
print("=" * 65)
