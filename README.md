PREDIKSI RISIKO BANJIR BERDASARKAN CURAH HUJAN DAN KETINGGIAN WILAYAH UNTUK MITIGASI BENCANA KOTA

Repositori ini berisi seluruh dokumentasi proyek, formulir pengajuan ide, serta eksperimen model Machine Learning dan Deep Learning untuk mengklasifikasikan tingkat risiko banjir daerah berdasarkan variabel meteorologi (curah hujan) dan karakteristik topografi (ketinggian tanah). Pengembangan proyek dirancang terstruktur mengikuti alur kerja ML Lifecycle (L-M-O-P).

INDEKS DOKUMEN PROYEK

Seluruh berkas administrasi dan laporan proyek tersimpan di dalam folder docs/:

docs/09_LAMPIRAN_B.md — Formulir Pengajuan Ide (Checkpoint 1) yang memuat formulasi learning problem, rincian dataset Kaggle, mitigasi etika, dan referensi awal.

docs/09_minggu9.md — Panduan tahapan dan checklist pengerjaan minggu pertama (Learning Problem).

docs/Laporan_Proyek_ML_LMOP.md — Laporan utama proyek mencakup Bab 1 hingga Bab 11 (Project Overview, Data Preparation, Modeling, hingga Evaluation).

GAMBARAN UMUM PROYEK

Proyek ini dikembangkan sebagai bentuk kontribusi terhadap SDG 11 (Kota dan Permukiman yang Berkelanjutan) dan SDG 13 (Penanganan Perubahan Iklim) melalui penyediaan sistem peringatan dini berbasis risiko bencana. Terdapat tiga klasifikasi pendekatan pemodelan yang diterapkan:

Model Baseline

Logistic Regression / Decision Tree sebagai landasan evaluasi sederhana untuk mengukur keterpisahan linear antar kelas risiko.

Model Advanced ML

Random Forest dan XGBoost untuk menangkap interaksi variabel non-linear yang kompleks antara curah hujan dan slope topografi.

Model Deep Learning

Multilayer Perceptron (MLP) berarsitektur multi-layer yang dilengkapi dengan Dropout dan Early Stopping untuk mencegah overfitting pada data tabular.
