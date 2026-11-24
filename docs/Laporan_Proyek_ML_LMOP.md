
**Judul Proyek: Prediksi Tingkat Risiko Banjir Berdasarkan Curah Hujan dan Ketinggian Wilayah Menggunakan Pendekatan Machine Learning dan Deep Learning**  


**Nama Mahasiswa:** [Takeru Putra Nojima - Satrio yunan wicaksono]  
**NIM:** [244311029 - 244311027]  
**Program Studi:** Teknologi Rekayasa Perangkat Lunak  
**Mata Kuliah:** Data Science  
**Dosen Pengampu:** Gus Nanang Syaifuddiin S.Kom, M.Kom  
**Tahun Akademik:** 2026/Semester 5  
**Link GitHub Repository:** [https://github.com/TakeruNojima/dataScience]  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** SDG - 11 - Kota dan Permukiman yang Berkelanjutan dan SDG - 13 - Penanganan Perubahan Iklim
**Target SDG yang Relevan:** Target 11.5 (Mengurangi dampak buruk bencana alam) dan Target 13.1 (Memperkuat ketahanan dan kapasitas adaptasi terhadap bahaya terkait iklim dan bencana alam)

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```
.

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang
Bencana banjir merupakan salah satu masalah hidrometeorologi yang paling sering terjadi dan menimbulkan kerugian besar terhadap infrastruktur, ekonomi, serta keselamatan jiwa manusia. Di banyak wilayah, banjir sering datang secara mendadak akibat kombinasi curah hujan ekstrem dan kondisi topografi atau ketinggian tanah yang rendah. Oleh karena itu, pembangunan sistem peringatan dini berbasis teknologi sangat penting untuk mendeteksi potensi bahaya lebih awal sehingga mitigasi dan evakuasi dapat dilakukan secara terarah. Proyek ini bermanfaat bagi masyarakat luas, instansi penanggulangan bencana, serta peneliti dalam merancang strategi pencegahan dampak banjir secara proaktif.

Penelitian terkait pemodelan bencana dan hidrologi didukung oleh berbagai literatur ilmiah, salah satunya mengenai pemanfaatan pendekatan komputasi cerdas dalam mengenali pola data lingkungan:

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Permasalahan ini selaras langsung dengan SDG 11 (Kota dan Permukiman yang Berkelanjutan) dan SDG 13 (Penanganan Perubahan Iklim), di mana pemanfaatan teknologi analitik data iklim dan topografi dapat meningkatkan ketahanan kota terhadap bencana alam.

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? |  SDG 11: Kota dan Permukiman yang Berkelanjutan (Sustainable Cities and Communities) & SDG 13: Penanganan Perubahan Iklim  |
| **Target SDG** | Target (misal 3.4, 13.1) mana yang paling relevan? |  Target 11.5 (Mengurangi dampak buruk bencana alam) dan Target 13.1 (Memperkuat kapasitas adaptasi terhadap bencana iklim)  |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? |  Tingginya risiko banjir akibat curah hujan tinggi di wilayah bertopografi rendah yang mengancam keselamatan warga dan merusak fasilitas umum.  |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? |  ML/DL mampu memproses pola hubungan non-linear yang kompleks antara curah hujan dan ketinggian wilayah untuk mengotomatisasi klasifikasi tingkat risiko banjir secara akurat.  |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG (langsung/tidak langsung)? |  Memberikan peringatan dini berbasis data bagi warga dan pihak berwenang untuk mengambil tindakan mitigasi sebelum bencana terjadi.  |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? |  Risiko False Negative (wilayah rawan diprediksi aman) sangat berbahaya bagi keselamatan; serta potensi bias data jika sampel wilayah kurang seimbang. Tidak melibatkan data privat penduduk.  |

**Sumber ide:** Diangkat dari isu nyata tingginya banjir musiman di berbagai wilayah Indonesia (seperti Jakarta dan Semarang) akibat curah hujan ekstrem, penurunan muka tanah, serta alih fungsi lahan, yang menuntut sistem mitigasi bencana berbasis data iklim.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini dimulai dari tahap **Learning Problem (L)** dengan merumuskan tujuan prediksi risiko kejadian banjir dan memahami struktur dataset tabular `flood_risk_dataset_india.csv` dari Kaggle (10.000 baris dan 14 kolom). Selanjutnya, data dibersihkan, fitur kategorikal di-*encode*, dan dilakukan *feature scaling* pada tahap Data Preparation, disusul dengan tahap **Model Design (M)** dan **Optimization (O)** untuk melatih model *baseline*, *advanced* (Random Forest/XGBoost), hingga *deep learning* (MLP). Tahap terakhir adalah **Predict & Evaluate (P)** untuk menguji performa model pada *test set*, melakukan analisis kesalahan (*error analysis*), serta melakukan iterasi perbaikan jika diperlukan.

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements

1. Tingginya variasi faktor hidrometeorologi dan lingkungan seperti intensitas curah hujan (`Rainfall`), debit sungai (`River Discharge`), tinggi muka air (`Water Level`), dan ketinggian wilayah (`Elevation`) menuntut model untuk mampu memprediksi risiko kejadian banjir dengan akurasi dan ketepatan yang tinggi.
2. Dataset tabular mentah mengandung kombinasi fitur numerik dengan rentang skala yang sangat berbeda (misalnya kepadatan penduduk hingga ribuan jiwa dibandingkan tinggi muka air dalam satuan meter) serta fitur kategorikal (`Land Cover` dan `Soil Type`), sehingga memerlukan proses *encoding* dan *feature scaling* yang tepat tanpa kebocoran data (*data leakage*).
3. Hubungan antara kondisi cuaca, topografi, dan kerentanan banjir bersifat non-linear, sehingga diperlukan perbandingan performa antara model *machine learning* konvensional dan *deep learning* untuk menemukan arsitektur yang paling optimal.

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, dll.)? | Memprediksi kejadian risiko banjir pada kolom **`Flood Occurred`** menggunakan tugas **Klasifikasi Biner** (`0` = Tidak Terjadi Banjir, `1` = Terjadi Banjir). |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis? Metrik keberhasilan utama? | Meminimalkan **Binary Cross-Entropy Loss** (*log-loss*), dengan metrik evaluasi utama berupa **F1-Score**, **Recall**, dan **ROC-AUC** untuk meminimalkan keterjadian *False Negative*. |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data? | Dataset tabular Kaggle (`flood_risk_dataset_india.csv`), berisi **10.000 baris** dan **14 kolom** (berukuran **1,77 MB**, format CSV) yang diasumsikan *independent and identically distributed* (i.i.d.). |

### 3.3 Goals

1. Membangun model pemelajaran mesin untuk memprediksi tingkat risiko kejadian banjir (`Flood Occurred`) dengan metrik F1-Score dan ROC-AUC di atas 80%.
2. Mengukur dan membandingkan performa tiga pendekatan pemodelan (Baseline: Logistic Regression, Advanced: Random Forest/XGBoost, Deep Learning: Multilayer Perceptron).
3. Menentukan model terbaik berbasis penanganan komprehensif terhadap fitur hidrometeorologi dan topografi.
4. Menghasilkan alur kerja pemrosesan data dan pemodelan yang dapat direproduksi penuh (100% *reproducible*).
5. Memberikan kontribusi nyata terhadap target SDG 11.5 dan SDG 13.1 melalui penyediaan landasan sistem peringatan dini bencana berbasis data.

### 3.4 Solution Approach  **[M]**

#### **Model 1 – Baseline Model**
- **Model yang dipilih:** **Logistic Regression**
- **Alasan Pemilihan:** Logistic Regression dipilih sebagai model acuan dasar (*baseline*) karena kesederhanaannya, kecepatan komputasi yang tinggi, dan sifatnya yang mudah diinterpretasikan secara matematis untuk masalah klasifikasi biner.

#### **Model 2 – Advanced / ML Model**
- **Model yang dipilih:** **Random Forest Classifier / XGBoost**
- **Alasan Pemilihan:** Random Forest / XGBoost merupakan algoritma *ensemble* berbasis pohon keputusan yang mampu memodelkan hubungan non-linear kompleks dan interaksi antar fitur (seperti kombinasi curah hujan tinggi dan elevasi rendah) tanpa memerlukan asumsi distribusi data linear.

#### **Model 3 – Deep Learning Model (WAJIB)**
- **Model yang dipilih:** **Multilayer Perceptron (MLP)**
- **Alasan Pemilihan:** Arsitektur neural network MLP dengan minimal 2 *hidden layers*, fungsi aktivasi ReLU, serta *Dropout layer* sangat sesuai untuk mengekstrak representasi fitur berdimensi tinggi pada data tabular dan meminimalkan *Binary Cross-Entropy Loss*.

---

## 4. DATA UNDERSTANDING  **[L]**
### 4.1 Informasi Dataset
**Sumber Dataset:** Kaggle – Flood Risk Dataset in India (file `flood_risk_dataset_india.csv`).

**Deskripsi Dataset:**
- Jumlah baris (rows): **10.000**
- Jumlah kolom (columns/features): **14**
- Tipe data: **Tabular** (campuran fitur numerik kontinu, biner, dan kategorikal)
- Ukuran dataset: **1,77 MB** (1.810,85 KB)
- Format file: **CSV**

### 4.2 Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi |
|---|---|---|
| `Latitude` | Float (`float64`) | Koordinat garis lintang lokasi pengamatan di wilayah India (rentang: $8,00^\circ$ – $36,99^\circ$). |
| `Longitude` | Float (`float64`) | Koordinat garis bujur lokasi pengamatan di wilayah India (rentang: $68,00^\circ$ – $97,00^\circ$). |
| `Rainfall (mm)` | Float (`float64`) | Intensitas curah hujan dalam milimeter (rentang: $0,01$ – $299,97\text{ mm}$, rata-rata $150,02\text{ mm}$). |
| `Temperature (°C)` | Float (`float64`) | Suhu udara permukaan dalam derajat Celsius (rentang: $15,00^\circ\text{C}$ – $44,99^\circ\text{C}$, rata-rata $29,96^\circ\text{C}$). |
| `Humidity (%)` | Float (`float64`) | Tingkat kelembapan udara relatif dalam persentase (rentang: $20,00\%$ – $100,00\%$, rata-rata $59,75\%$). |
| `River Discharge (m³/s)` | Float (`float64`) | Debit aliran air sungai dalam meter kubik per detik (rentang: $0,04$ – $4.999,70\text{ m}^3/\text{s}$, rata-rata $2.515,72\text{ m}^3/\text{s}$). |
| `Water Level (m)` | Float (`float64`) | Ketinggian permukaan air dalam meter (rentang: $0,003$ – $10,00\text{ m}$, rata-rata $5,02\text{ m}$). |
| `Elevation (m)` | Float (`float64`) | Ketinggian wilayah/topografi di atas permukaan laut dalam meter (rentang: $1,15$ – $8.846,89\text{ m}$, rata-rata $4.417,14\text{ m}$). |
| `Land Cover` | Object (`object`) | Jenis tutupan lahan di wilayah pengamatan, terdiri dari 5 kategori: `Water Body`, `Desert`, `Forest`, `Agricultural`, dan `Urban`. |
| `Soil Type` | Object (`object`) | Jenis tanah di wilayah pengamatan yang memengaruhi daya serap air, terdiri dari 5 kategori: `Peat`, `Silt`, `Clay`, `Loam`, dan `Sandy`. |
| `Population Density` | Float (`float64`) | Kepadatan penduduk di area tersebut (rentang: $2,29$ – $9.999,17$, rata-rata $5.021,47$). |
| `Infrastructure` | Integer (`int64`) | Indikator ketersediaan infrastruktur pengendali banjir/drainase (`0` = Tidak memadai/tidak ada, `1` = Tersedia/memadai). |
| `Historical Floods` | Integer (`int64`) | Riwayat kejadian banjir sebelumnya pada lokasi tersebut (`0` = Tidak ada riwayat, `1` = Pernah mengalami banjir). |
| `Flood Occurred` | Integer (`int64`) | **Variabel target** klasifikasi biner yang menunjukkan terjadinya banjir (`0` = Tidak Terjadi Banjir, `1` = Terjadi Banjir). |

### 4.3 Kondisi Data

Berdasarkan pemeriksaan yang dijalankan pada notebook **`01_learning_problem_eda.ipynb`**, kondisi dataset adalah sebagai berikut:

- **Missing Values:** Tidak ditemukan nilai kosong. Hasil pengecekan `df.isnull().sum().sum()` adalah **0** di seluruh 14 kolom.
- **Duplicate Data:** Tidak ditemukan baris duplikat. Hasil pengecekan `df.duplicated().sum()` adalah **0**.
- **Outliers:** Berdasarkan pemeriksaan statistik metode *Interquartile Range* (IQR) pada seluruh fitur numerik, ditemukan **0 outlier** di luar batas $Q_1 - 1,5 \times \text{IQR}$ maupun $Q_3 + 1,5 \times \text{IQR}$.
- **Imbalanced Data:** Distribusi kelas pada variabel target `Flood Occurred` tergolong **seimbang (*balanced*)** dengan rincian:

| Kelas Target (`Flood Occurred`) | Keterangan | Jumlah Data | Persentase |
|---|---|---:|---:|
| `0` | Tidak Terjadi Banjir | 4.943 | 49,43% |
| `1` | Terjadi Banjir | 5.057 | 50,57% |
| **Total** | | **10.000** | **100,00%** |

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

Seluruh visualisasi eksplorasi data berikut diimplementasikan pada notebook pertama (**`01_learning_problem_eda.ipynb`**):

#### Visualisasi 1: Distribusi Kejadian Banjir (`Flood Occurred`)
Visualisasi dilakukan menggunakan `sns.countplot()` pada kolom `Flood Occurred` (dengan label `Tidak Banjir (0)` dan `Terjadi Banjir (1)`).
- **Insight:** Jumlah sampel untuk kelas `1` (Terjadi Banjir) sebanyak **5.057 baris (50,57%)** dan kelas `0` (Tidak Terjadi Banjir) sebanyak **4.943 baris (49,43%)**. Selisih antar kelas hanya 1,14%, mengonfirmasi bahwa dataset seimbang dan tidak memerlukan teknik re-sampling.

#### Visualisasi 2: Kejadian Banjir Berdasarkan Tutupan Lahan (`Land Cover`)
Visualisasi dilakukan menggunakan `sns.countplot(x='Land Cover', hue='Flood Occurred')` untuk membandingkan proporsi banjir pada 5 jenis tutupan lahan (`Water Body`: 2.046, `Desert`: 2.033, `Forest`: 2.005, `Agricultural`: 1.978, `Urban`: 1.938).
- **Insight:** Setiap jenis tutupan lahan terdistribusi secara seragam di kisaran 19,38%–20,46% dari keseluruhan data, dengan rasio kejadian banjir yang proporsional pada masing-masing kategori.

#### Visualisasi 3: Distribusi Curah Hujan (`Rainfall (mm)`)
Visualisasi dilakukan menggunakan `sns.histplot()` dengan kurva KDE pada kolom `Rainfall (mm)`.
- **Insight:** Curah hujan tersebar merata dari $0,01\text{ mm}$ hingga $299,97\text{ mm}$ dengan rata-rata $150,02\text{ mm}$ dan standar deviasi $86,03\text{ mm}$, menunjukkan sebaran kontinu tanpa tingkat kemencengan (*skewness*) ekstrem.


## 5. DATA PREPARATION  **[L]**

Seluruh rangkaian prapemrosesan data dipisahkan secara modular ke dalam notebook kedua (**`02_data_preparation.ipynb`**) agar alur transformasi fitur siap pakai sebelum tahap pemodelan.

### 5.1 Data Cleaning
1. **Pemeriksaan missing values:** Diverifikasi pada `01_learning_problem_eda.ipynb` menggunakan `df.isnull().sum().sum()`. Hasilnya **0**, tidak membutuhkan imputasi data.
2. **Pemeriksaan duplikat:** Menggunakan `df.duplicated().sum()`. Hasilnya **0**, seluruh 10.000 sampel data bersifat unik.
3. **Pemeriksaan tipe data:** Terdapat 12 kolom numerik (`9 float64` dan `3 int64`) serta 2 kolom kategorikal bertipe `object` (`Land Cover` dan `Soil Type`).
4. **Penanganan outlier:** Evaluasi nilai min-max dan IQR menunjukkan seluruh nilai fitur numerik berada pada rentang fisik wajar, sehingga tidak dilakukan pemotongan data.

### 5.2 Feature Engineering
Pada notebook **`02_data_preparation.ipynb`**, dilakukan pemisahan variabel dan pembentukan fitur *dummy*:
1. **Pemisahan Fitur (`X`) dan Target (`y`):** Kolom `Flood Occurred` ditetapkan sebagai variabel target (`y`), sedangkan 13 kolom lainnya menjadi fitur prediktor.
2. **Encoding Fitur Kategorikal:** Fitur `Land Cover` (5 kategori) dan `Soil Type` (5 kategori) diubah menjadi bentuk biner menggunakan **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`). Penggunaan `drop_first=True` menghasilkan 8 kolom indikator baru sehingga total matriks `X` memiliki **19 fitur prediktor**.

```python
# Implementasi pada 02_data_preparation.ipynb
X = pd.get_dummies(
    df.drop(columns=['Flood Occurred']),
    columns=['Land Cover', 'Soil Type'],
    drop_first=True
)
y = df['Flood Occurred']

print("Jumlah fitur setelah One-Hot Encoding:", X.shape[1])
print("\nJumlah data per kategori target (Flood Occurred):")
print(y.value_counts())
```


### 5.3 Data Transformation

1. **One-Hot Encoding pada fitur kategorikal:** Mengonversi `Land Cover` dan `Soil Type` menjadi variabel dummy biner numerik.
2. **Feature Scaling menggunakan `StandardScaler`:** Mengingat perbedaan skala antar variabel (seperti `Elevation` hingga $8.846,89\text{ m}$ dan `Population Density` hingga $9.999,17$ versus `Water Level` $0\text{--}10\text{ m}$), dilakukan pembakuan skala ($z = \frac{x - \mu}{\sigma}$) menggunakan `StandardScaler`.
3. **Pencegahan Data Leakage:** Proses `fit_transform(X_train)` hanya dilakukan pada subset *training*, kemudian parameter standar ($\mu, \sigma$) yang dihasilkan digunakan untuk mentransformasi `transform(X_val)`

### 5.4 Data Splitting
Pada notebook **`02_data_preparation.ipynb`**, dataset dibagi menjadi subset data *training* dan *validation* menggunakan rasio **80:20** dengan `train_test_split` dan penguncian acak (`random_state=42`) serta stratifikasi kelas (`stratify=y`):

```python
# Implementasi pada 02_data_preparation_india.ipynb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data validation: {X_val.shape[0]}")
```

Hasil pembagian data:

| Subset | Proporsi | Jumlah Sampel | Kelas `0` (Tidak Banjir) | Kelas `1` (Banjir) |
|---|---:|---:|---:|---:|
| Training | 80% | 8.000 | 3.954 (49,42%) | 4.046 (50,58%) |
| Validation | 20% | 2.000 | 989 (49,45%) | 1.011 (50,55%) |
| **Total** | **100%** | **10.000** | **4.943 (49,43%)** | **5.057 (50,57%)** |

### 5.5 Data Balancing (jika diperlukan)
Berdasarkan analisis distribusi target pada Bab 4.3 dan 5.4, persentase kelas `0` (49,43%) dan kelas `1` (50,57%) sangat seimbang. Oleh karena itu, **teknik penyeimbangan data seperti SMOTE atau *undersampling* tidak diperlukan**.

### 5.6 Ringkasan Data Preparation
1. **Apa yang dilakukan:**
  - Melakukan validasi kebersihan data pada dataset `flood_risk_dataset.csv` (10.000 baris × 14 kolom) di notebook **`01_learning_problem_eda.ipynb`**, memastikan 0 *missing values*, 0 duplikat, dan 0 *outlier* ekstrem.
  - Mengaplikasikan *One-Hot Encoding* (`drop_first=True`) pada 2 fitur kategorikal (`Land Cover` dan `Soil Type`) di notebook **`02_data_preparation.ipynb`**, memetakan matriks `X` menjadi 19 fitur numerik.
  - Melakukan *stratified splitting* data dengan rasio 80:20 ($8.000$ sampel *train*, $2.000$ sampel *validation*).
  - Menjalankan *StandardScaler* secara terpisah (*fit* pada *train*, *transform* pada *val*) untuk mencegah kebocoran data.
2. **Mengapa penting:**
  - *One-Hot Encoding* memungkinkan algoritma matematika ML/DL membaca data kategorikal tanpa implikasi urutan numerik.
  - *Feature scaling* mencegah fitur berdimensi besar mendominasi fungsi biaya (*cost function*) dan pembaruan gradien.
  - Pembagian data *stratified* menjaga konsistensi proporsi kelas target pada subset uji, sementara pemisahan ke dalam dua notebook menjaga modularitas kode.
3. **Bagaimana implementasinya:**
  - Pemeriksaan awal & EDA (`01_learning_problem_eda.ipynb`): `isnull()`, `duplicated()`, `describe()`, `countplot()`, dan `histplot()`.
  - Transformasi kategorikal (`02_data_preparation.ipynb`): `pd.get_dummies()`.
  - Data splitting (`02_data_preparation.ipynb`): `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
  - Standardisasi (`02_data_preparation.ipynb`): `StandardScaler().fit_transform(X_train)` dan `transform(X_val)`.

---

## 6. MODELING  **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**

**Nama Model:** Logistic Regression
**Teori Singkat:**  
Logistic Regression memodelkan probabilitas kejadian banjir ($P(Y=1 \mid \mathbf{x})$) dengan menghitung kombinasi linear dari fitur-fitur input ke dalam bentuk log-odds (logit) log-odds tersebut kemudian dipetakan menjadi probabilitas pada rentang $[0, 1]$ menggunakan fungsi aktivasi sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$. Model melatih bobot $\mathbf{w}$ dan bias $b$ dengan meminimalkan Binary Cross-Entropy Loss (serta regularisasi L2) sehingga prediksi mendekati label aktual Flood Occurred.
**Alasan Pemilihan:**  
Logistic Regression dipilih sebagai baseline karena proses pelatihannya cepat, koefisien fiturnya mudah diinterpretasikan (menunjukkan arah pengaruh fitur seperti Rainfall (mm), River Discharge (m³/s), atau Elevation (m)), dan menghasilkan keluaran probabilitas yang jelas sebagai tolok ukur awal sebelum mencoba model yang lebih kompleks.

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** Model klasifikasi linear probabilistik (generalized linear model dengan logit link / fungsi sigmoid).
- **Hypothesis space:** Hypothesis space: Himpunan seluruh fungsi batas keputusan linear (hyperplane) pada ruang fitur berdimensi-$d$ (setelah standardisasi fitur numerik dan one-hot encoding fitur kategorikal seperti Land Cover dan Soil Type)
- **Inductive bias / asumsi:** Mengasumsikan hubungan antara fitur input dan log-odds target bersifat linear serta aditif, tidak menangkap interaksi kompleks/non-linear antarfitur secara otomatis tanpa rekayasa fitur tambahan, dan cenderung memilih bobot berukuran kecil karena regularisasi L2.
- **Implikasi terhadap dataset Anda:** Pada flood_risk_dataset_india.csv (10.000 baris, distribusi target Flood Occurred seimbang sekitar 50,57% kelas 1 dan 49,43% kelas 0), asumsi batas keputusan linear kemungkinan kurang cocok / terbatas apabila risiko banjir sebenarnya dipengaruhi oleh ambang batas atau interaksi non-linear antarvariabel (misalnya curah hujan tinggi bersamaan dengan elevasi rendah dan tutupan lahan tertentu). Oleh karena itu, model ini berfungsi sebagai batas bawah (benchmark) untuk melihat seberapa besar peningkatan performa yang bisa dicapai oleh model non-linear pada tahap berikutnya.

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model_baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=100, random_state=42))
])

model_baseline.fit(X_train, y_train)
y_val_pred = model_baseline.predict(X_val)
y_val_prob = model_baseline.predict_proba(X_val)[:, 1]
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**

**Objective/loss yang dioptimalkan:**  Binary Cross-Entropy Loss (Log-Loss) dengan regularisasi L2 (Ridge), yaitu meminimalkan: $$\mathcal{J}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big] + \frac{1}{2C}\Vert{}\mathbf{w}\Vert{}_2^2$$
**Parameter yang digunakan:**
```
penalty (regularization): 'l2' (menekan norma bobot $\mathbf{w}$ agar tidak terlalu besar dan lebih tahan terhadap noise)
C (inverse regularization strength): 1.0
solver: 'lbfgs'
max_iter: 100 (sudah konvergen pada data yang telah distandardisasi; dapat dinaikkan jika diperlukan pada eksperimen lanjutan)
random_state: 42
```

#### 6.1.5 Hasil Awal  **[O → P]**

**[Tuliskan hasil awal pada validation set; evaluasi akhir pada test set dijelaskan di Bab 7]**

Pada evaluasi awal menggunakan validation set (1.500 sampel, threshold probabilitas default 0.5):

    Accuracy: 0.4867

    Precision: 0.4933

    Recall: 0.5349

    F1-Score: 0.5133

    ROC-AUC: 0.4934

    Confusion Matrix (Validation Set): TN = 324, FP = 417, FN = 353, TP = 406

Performa awal model baseline masih berada di sekitar ~0.49–0.53 pada data fitur mentah, sehingga menjadi titik acuan awal (benchmark) untuk dibandingkan dengan model ensemble non-linear (Random Forest pada Subbab 6.2) maupun tahap optimization berikutnya. Evaluasi akhir pada test set akan dibahas pada Bab 7.

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]  
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: ensemble pohon keputusan]
- **Hypothesis space:** [fungsi non-linear piecewise, interaksi fitur otomatis, dll.]
- **Inductive bias / asumsi:** [misal: data tabular dengan struktur ambang/threshold]
- **Perbandingan dengan baseline:** [apa yang bisa dipelajari model ini yang tidak bisa baseline?]

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Search space: [rentang parameter yang dicoba]
- Skema validasi: [k-fold CV / validation set]
- Best parameters: [...]
- Skor validasi sebelum vs sesudah tuning: [...]

#### 6.2.5 Kontrol Overfitting  **[O]**
[Jelaskan regularisasi/pembatasan kompleksitas yang dipakai, misal: max_depth, min_samples_leaf, early stopping pada boosting]

#### 6.2.6 Hasil Model  **[O → P]**

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**

**Inductive bias arsitektur:** [misal: CNN → lokalitas & translation invariance; LSTM → ketergantungan sekuensial; MLP → tanpa asumsi struktur]  
**Hypothesis space:** [kapasitas model: jumlah layer, unit, parameter]

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
- Regularisasi: [Dropout / L2 / BatchNorm / augmentasi]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Exp-2 | [ ] | [ ] | [ ] | [ ] | [ ] |

#### 6.3.5 Implementasi (Ringkas)  **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| [misal: recall kelas minoritas rendah] | [L] data | [misal: class weight / SMOTE] | [ ... ] |
| [misal: gap train-val besar] | [O] optimasi | [misal: tambah dropout, early stopping] | [ ... ] |
| [misal: model terlalu sederhana] | [M] desain | [misal: tambah layer / ganti arsitektur] | [ ... ] |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | [ ... ] | [ ... ] |
| **M** – Model Design | [ ... ] | [ ... ] |
| **O** – Optimization | [ ... ] | [ ... ] |
| **P** – Predict & Evaluate | [ ... ] | [ ... ] |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**
```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2** | **L** – Data | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5) | Notebook data preparation + repo GitHub dibuat |
| **3** | **M** – Model Design | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5) | Model DL berjalan end-to-end (training awal) |
| **5** | **O** – Optimization | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7) | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10) | Laporan final + GitHub + video + presentasi |

**Catatan untuk dosen:**
- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**
- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia) | Contoh arah ide ML (inspirasi) |
|-----|------------------|--------------------------------|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| 3 | Kehidupan Sehat dan Sejahtera | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman yang Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi dan Produksi yang Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ ... ] |
| Judul ide | [ ... ] |
| SDG & target yang dipilih | [ ... ] |
| Masalah nyata yang diangkat | [ ... ] |
| Tugas ML (Target & jenis tugas) | [ ... ] |
| Objective & metrik utama | [ ... ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ ... ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ] |
| Risiko & pertimbangan etika (bias, privasi) | [ ... ] |
| Referensi awal (minimal 2) | [ ... ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
