import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Dasbor Klasifikasi Kualitas Apel")

st.title("🍎 Analisis Prediktif: Klasifikasi Kualitas Apel")

st.write(
    """
    Dasbor ini menyediakan eksplorasi interaktif klasifikasi kualitas apel.
    Kita akan melalui pemahaman data, persiapan, dan terakhir, evaluasi model.
    """
)

# Memuat Data
st.header("1. Pemuatan Data")
st.write("Memuat dataset Kualitas Apel...")

try:
    apel = pd.read_csv('Dashboard/apple_quality.csv')
    st.success("Dataset berhasil dimuat!")
    st.dataframe(apel.head())
except FileNotFoundError:
    st.error("`apple_quality.csv` tidak ditemukan. Pastikan file berada di direktori yang sama dengan file app.py.")
    st.stop()


# EDA
st.header("2. Pemahaman Data (Analisis Data Eksplorasi)")

st.subheader("Informasi Dataset")
st.write(f"Dataset berisi **{apel.shape[0]} baris** dan **{apel.shape[1]} kolom**.")
st.dataframe(apel.head())

st.write("#### Tipe Data dan Nilai Hilang")
st.write(apel.info(verbose=True, show_counts=True))

st.markdown("""
- Dataset memiliki 7 kolom bertipe `float64` dan 2 kolom bertipe `object` (`Acidity` dan `Quality`).
- `A_id` akan dihilangkan karena ini adalah kolom ID.
- `Acidity` (saat ini `object`) perlu dikonversi ke `float64`.
""")

st.subheader("Statistik Deskriptif")
st.dataframe(apel.describe())
st.write("""
- `count` menunjukkan 4000 entri valid per kolom numerik, mengindikasikan beberapa nilai hilang dari total 4001 baris.
- Nilai `mean` di sekitar nol menunjukkan data mungkin sudah distandardisasi.
- `std`, `min`, `max`, dan kuartil memberikan wawasan tentang distribusi data.
""")

st.subheader("Pemeriksaan Nilai Hilang")
st.dataframe(apel.isnull().sum().to_frame(name='Nilai Hilang'))
st.write("Setiap kolom, kecuali 'Acidity', memiliki 1 nilai hilang.")