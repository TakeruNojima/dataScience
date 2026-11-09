## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ Takeru Putra Nojima / 029 - Satrio yunan wicaksono  ] |

| Judul ide | [ Memprediksi Risiko Banjir di Suatu Wilayah Berdasarkan Curah Hujan dan Ketinggian Wilayah Menggunakan Pendekatan Machine Learning dan Deep Learning ] |

| SDG & target yang dipilih | [ SDG 11: Kota dan Permukiman yang Berkelanjutan (Target 11.5: Mengurangi dampak bencana alam dan jumlah korban terdampak)
SDG 13: Penanganan Perubahan Iklim (Target 13.1: Memperkuat ketahanan dan kapasitas adaptasi terhadap bahaya terkait iklim dan bencana alam) ] |

| Masalah nyata yang diangkat | [ Bencana banjir sering timbul secara mendadak akibat kombinasi antara intensitas curah hujan yang tinggi dan kondisi topografi (ketinggian) wilayah yang rendah. Kurangnya sistem peringatan dini berbasis risiko membuat proses mitigasi dan evakuasi terlambat, sehingga menimbulkan kerugian materi serta ancaman keselamatan warga. ] |

| Tugas ML (Target & jenis tugas) | [ Jenis Tugas: Klasifikasi Multikelas / Biner (Tabular Data)
Target Prediksi: Tingkat Risiko Banjir (misal: Rendah, Sedang, Tinggi atau 0: Tidak Banjir, 1: Potensi Banjir) ] |

| Objective & metrik utama | [ Objective: Meminimalkan kesalahan klasifikasi risiko banjir, terutama False Negatives (memprediksi wilayah aman padahal berisiko tinggi).
Metrik Utama: F1-Score (Macro/Weighted), Recall (terutama untuk kelas risiko tinggi), dan Accuracy. ] |

| Dataset (sumber, URL, ukuran, jenis data) | [ 
Sumber: Dataset Publik (Kaggle - Flood Prediction Dataset / BMKG / Open Data Indonesia)

Fitur Utama: Curah hujan (rainfall), ketinggian tanah (elevation/altitude), kelembapan, drainase, dll.

URL: [https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset], [https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset]

Jenis Data: Data Tabular

Ukuran Data: ~10.000+ baris data (format CSV) ] |

| Rencana tiga model (baseline / advanced / deep learning) | [ 
1. Baseline Model: Logistic Regression / Decision Tree (sebagai model pembanding sederhana)
2. Advanced ML: Random Forest / XGBoost (memodelkan hubungan non-linear antara variabel iklim & topografi)
3. Deep Learning: Multilayer Perceptron (MLP) dengan minimal 2-3 hidden layers, Batch Normalization, dan Dropout (untuk mempelajari representasi pola interaksi data tabular yang kompleks) ] |

| Risiko & pertimbangan etika (bias, privasi) | [ 
- Risiko Model (False Negatives): Kesalahan prediksi yang menilai daerah rawan sebagai daerah aman dapat menyebabkan kepanikan atau kegagalan mitigasi awal.
- Bias Data: Ketimpangan jumlah sampel wilayah yang pernah mengalami banjir vs tidak pernah banjir (class imbalance).
- Privasi: Menggunakan data agregat geospasial/cuaca tanpa mengekspos data pribadi penduduk. ] |

| Referensi awal (minimal 2) | [ 
1. World Meteorological Organization (WMO). (2021). Guidance on Flash Flood Forecasting and Early Warning Systems.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. ] |

**Kriteria persetujuan dosen:**
- [X] Ide berasal dari mahasiswa dan tidak duplikat
- [X] Selaras jelas dengan satu SDG dan targetnya
- [X] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [X] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [X] Layak untuk tiga model termasuk deep learning dalam 6 minggu