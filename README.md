# PREDIKSI RISIKO BANJIR BERDASARKAN CURAH HUJAN DAN KETINGGIAN WILAYAH UNTUK MITIGASI BENCANA KOTA

Repositori ini berisi dokumentasi proyek, formulir pengajuan ide, laporan pengerjaan, serta eksperimen model Machine Learning dan Deep Learning untuk memprediksi tingkat risiko banjir di suatu wilayah berdasarkan faktor meteorologi dan karakteristik topografi.

Faktor utama yang digunakan dalam penelitian ini adalah curah hujan dan ketinggian wilayah. Proyek ini bertujuan untuk mengembangkan model klasifikasi yang mampu mengidentifikasi tingkat risiko banjir ke dalam tiga kategori, yaitu **Rendah, Sedang, dan Tinggi**.

Pengembangan proyek dilakukan secara terstruktur dengan mengikuti tahapan **ML Lifecycle (L-M-O-P)**, mulai dari pengolahan data hingga evaluasi dan pengembangan model.

## INDEKS DOKUMEN PROYEK

Seluruh dokumen administrasi, laporan, dan dokumentasi pengerjaan proyek tersimpan di dalam folder `docs/` dengan rincian sebagai berikut:

- [docs/09_LAMPIRAN_B.md](./docs/09_LAMPIRAN_B.md) – Formulir pengajuan ide proyek.
- [docs/09_minggu9.md](./docs/09_minggu9.md) – Dokumentasi pengerjaan proyek minggu ke-9.
- [docs/Laporan_Proyek_ML_LMOP.md](./docs/Laporan_Proyek_ML_LMOP.md) – Laporan keseluruhan proyek berdasarkan tahapan ML Lifecycle.

## GAMBARAN UMUM PROYEK

Proyek ini dikembangkan sebagai bentuk kontribusi terhadap **SDG 11 (Kota dan Permukiman yang Berkelanjutan)** dan **SDG 13 (Penanganan Perubahan Iklim)**, khususnya dalam mendukung upaya mitigasi dan kesiapsiagaan terhadap bencana banjir.

Melalui pendekatan Machine Learning dan Deep Learning, proyek ini membandingkan tiga pendekatan pemodelan untuk mengetahui model yang paling baik dalam mengklasifikasikan tingkat risiko banjir.

### 1. Model Baseline

**Logistic Regression dan Decision Tree** digunakan sebagai model pembanding awal untuk mengetahui kemampuan dasar algoritma dalam mengklasifikasikan tingkat risiko banjir berdasarkan fitur yang tersedia.

### 2. Model Advanced ML

**Random Forest dan XGBoost** digunakan untuk mempelajari hubungan non-linear serta interaksi antarvariabel, khususnya antara curah hujan dan kondisi topografi wilayah, sehingga diharapkan mampu menghasilkan prediksi yang lebih baik.

### 3. Model Deep Learning

**Multilayer Perceptron (MLP)** digunakan untuk mempelajari pola hubungan yang lebih kompleks pada data tabular. Model ini dirancang menggunakan beberapa hidden layers serta teknik Dropout dan Early Stopping untuk membantu mengurangi risiko overfitting.

## TUJUAN PROYEK

1. Mengembangkan model Machine Learning dan Deep Learning untuk memprediksi tingkat risiko banjir.
2. Menganalisis hubungan antara curah hujan, ketinggian wilayah, dan faktor lingkungan lainnya terhadap risiko banjir.
3. Membandingkan performa model Baseline, Advanced ML, dan Deep Learning menggunakan metrik evaluasi yang sesuai.
4. Menentukan model dengan performa terbaik untuk mendukung penyediaan informasi awal dalam upaya mitigasi bencana banjir.

## METODE EVALUASI

Evaluasi model dilakukan menggunakan beberapa metrik, yaitu:

- **Accuracy:** Mengukur persentase prediksi yang benar dari seluruh data.
- **Precision:** Mengukur ketepatan model dalam memprediksi suatu kelas risiko.
- **Recall:** Mengukur kemampuan model dalam mengenali seluruh data pada suatu kelas, terutama kelas risiko tinggi.
- **F1-Score (Macro):** Mengukur keseimbangan antara Precision dan Recall pada seluruh kelas klasifikasi.

Perhatian utama diberikan pada nilai Recall untuk kelas risiko tinggi guna meminimalkan kesalahan prediksi wilayah berisiko tinggi sebagai wilayah berisiko rendah.

## TEKNOLOGI YANG DIGUNAKAN

- **Python** – Bahasa pemrograman utama.
- **Pandas & NumPy** – Pengolahan dan manipulasi data.
- **Scikit-learn** – Pengembangan model Machine Learning dan preprocessing.
- **XGBoost** – Pengembangan model Advanced Machine Learning.
- **TensorFlow / Keras** – Pengembangan model Deep Learning.
- **Matplotlib & Seaborn** – Visualisasi dan analisis data.

## CATATAN

Hasil prediksi dari proyek ini merupakan informasi pendukung untuk membantu memahami tingkat risiko banjir berdasarkan data yang digunakan. Model yang dikembangkan bukan pengganti sistem peringatan dini resmi maupun keputusan dari instansi penanggulangan bencana.