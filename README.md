# Eksperimen SML - Ridwan Awaludin

Eksperimen ini menggunakan dataset Iris dari scikit-learn untuk menyiapkan
data klasifikasi tiga spesies bunga. Notebook mengikuti Template Eksperimen
MSML dan mencakup pemeriksaan data, EDA, pembagian data, imputasi median, dan
standardisasi.

## Isi repository

- `Template_Eksperimen_MSML_Ridwan_Awaludin.ipynb`: notebook eksperimen.
- `automate_Ridwan-Awaludin.py`: skrip untuk mengulang preprocessing dan
  mengekspor dataset.
- `data/raw/iris_raw.csv`: dataset Iris sebelum preprocessing.
- `data/processed/iris_train_preprocessed.csv` dan
  `data/processed/iris_test_preprocessed.csv`: data train dan test setelah
  imputasi serta standardisasi.

Dataset raw diekspor dari dataset Iris bawaan scikit-learn. Imputer dan scaler
hanya di-fit pada data train untuk menghindari kebocoran informasi dari data
test.

## Menjalankan otomasi

Gunakan Python 3.10 atau lebih baru:

```powershell
python -m pip install -r requirements.txt
python automate_Ridwan-Awaludin.py
```

Skrip membuat folder `data/raw` dan `data/processed`, lalu menulis ulang CSV
hasil eksperimen dengan pembagian stratified 80:20 dan `random_state=42`.
