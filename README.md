# Machine Learning - Loan Approval Prediction Pipeline

Repositori ini memuat implementasi *end-to-end* pipeline Machine Learning untuk memprediksi persetujuan kelayakan pengajuan pinjaman (*Loan Approval*) berdasarkan profil keuangan dan skor kredit calon nasabah.

## 📌 Permasalahan & Metodologi Kerja

Proyek ini bertujuan untuk menyelesaikan masalah klasifikasi kelayakan kredit secara otomatis menggunakan algoritma *supervised learning*. Alur kerja yang diterapkan mencakup:

1. **Data Collection**: Mengolah dataset `loan_approval.csv` yang berisi 2.000 data nasabah dengan fitur seperti `income`, `credit_score`, `loan_amount`, `years_employed`, dan `points`.
2. **Data Preprocessing**:
   - Menghapus kolom identitas non-prediktif (`name`, `city`).
   - Mengubah target kategorikal `loan_approved` menjadi nilai biner (`1` untuk Approved, `0` untuk Rejected).
   - Menormalisasi variabel numerik menggunakan `StandardScaler`.
3. **Data Splitting**: Membagi dataset menjadi **80% Data Latih (Training)** dan **20% Data Uji (Testing)** menggunakan metode *stratified sampling*.
4. **Model Training & Evaluation**: Melatih dan membandingkan performa 3 algoritma klasifikasi:
   - *Logistic Regression*
   - *Decision Tree Classifier*
   - *Random Forest Classifier*
   
   Evaluasi dilakukan menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, serta visualisasi *Confusion Matrix*.
5. **Inference Pipeline**: Menggunakan model *Random Forest* terbaik untuk memprediksi data nasabah baru secara *real-time* beserta kalkulasi persentase probabilitas persetujuannya.

## 📁 Struktur Repositori
- `Machine_Learning.ipynb` : Jupyter Notebook utama berisi seluruh proses preprocessing, pelatihan model, evaluasi, hingga inferensi.
- `loan_approval.csv` : Dataset tabular yang digunakan dalam eksperimen.

## 🛠️ Library & Tools
- **Python 3**
- `pandas` & `numpy` (Manajemen & manipulasi data)
- `scikit-learn` (Preprocessing, algoritma Machine Learning, & metrik evaluasi)
- `matplotlib` & `seaborn` (Visualisasi data & confusion matrix)

## Cara Menjalankan
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/Aggrakurnia/Machine_Learning.git](https://github.com/Aggrakurnia/Machine_Learning.git)
