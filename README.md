# Eksperimen SML - Ridwan Awaludin

Eksperimen ini menggunakan Breast Cancer Wisconsin (Diagnostic) Dataset dari
UCI Machine Learning Repository untuk klasifikasi biner diagnosis malignant
dan benign. Dataset dimuat melalui scikit-learn. Notebook mengikuti Template
Eksperimen MSML dan mencakup pemeriksaan data, EDA, pembagian data, imputasi
median, dan standardisasi. Eksperimen ini bersifat edukatif dan bukan alat
diagnosis medis.

## Isi repository

- `Template_Eksperimen_MSML_Ridwan_Awaludin.ipynb`: notebook eksperimen.
- `automate_Ridwan-Awaludin.py`: skrip untuk mengulang preprocessing dan
  mengekspor dataset.
- `data/raw/breast_cancer_raw.csv`: dataset sebelum preprocessing.
- `data/processed/breast_cancer_train_preprocessed.csv` dan
  `data/processed/breast_cancer_test_preprocessed.csv`: data train dan test setelah
  imputasi serta standardisasi.

Dataset terdiri dari 569 sampel dan 30 fitur numerik, bersumber dari UCI
Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>.
Imputer dan scaler hanya di-fit pada data train untuk menghindari kebocoran
informasi dari data test.

Notebook menyimpan dataset raw dan hasil preprocessing langsung dari alur
eksperimen menggunakan `to_csv()`: `data/raw/breast_cancer_raw.csv` serta CSV
train dan test di `data/processed/`. Folder `data/` dibuat relatif terhadap
working directory saat notebook dijalankan.

## Menjalankan otomasi

Gunakan Python 3.10 atau lebih baru:

```powershell
python -m pip install -r requirements.txt
python automate_Ridwan-Awaludin.py
```

Skrip membuat folder `data/raw` dan `data/processed`, lalu menulis ulang CSV
hasil eksperimen dengan pembagian stratified 80:20 dan `random_state=42`.
