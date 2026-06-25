## Klasifikasi Email Phising/Spam dengan TF-IDF: Perbandingan SVM, Random Forest, dan LSTM

---

## Daftar Isi
1. [Overview](#1-overview)
2. [Tujuan](#2-tujuan)
3. [Dataset](#3-dataset)
4. [Metodologi / Pipeline](#4-metodologi--pipeline)
5. [Pemodelan](#5-pemodelan)
6. [Hasil Evaluasi](#6-hasil-evaluasi)
7. [Diskusi](#7-diskusi)
8. [Keterbatasan & Catatan Teknis](#8-keterbatasan--catatan-teknis)
9. [Kesimpulan](#9-kesimpulan)
10. [Rekomendasi Pengembangan Lanjutan](#10-rekomendasi-pengembangan-lanjutan)
11. [Lampiran](#11-lampiran)

---

## 1. Overview

Proyek ini membangun dan membandingkan tiga model klasifikasi email spam/ham (legitimate) berbasis representasi fitur **TF-IDF**, dengan tiga algoritma yang berbeda karakter:

| Model | Algoritma | Sifat |
|---|---|---|
| Model 1 | SVM (linear kernel) | Classifier linear klasik |
| Model 2 | Random Forest | Ensemble tree-based |
| Model 3 | LSTM (pseudo-sequence reshape dari vektor TF-IDF) | Deep learning, diuji secara kritis sebagai studi ablasi |

Dataset gabungan dari tiga sumber publik (**Enron**, **Nazario**, **SpamAssassin**) diuji dalam dua skenario: **Case 1** (data besar, distribusi kelas alami/imbalanced) dan **Case 2** (data lebih kecil, kelas diseimbangkan via undersampling).

**Temuan utama:** SVM linear secara konsisten mengungguli Random Forest dan LSTM pada kedua skenario (akurasi 99,69% di Case 1 dan 99,23% di Case 2). Model LSTM — yang dibangun dengan mereshape vektor TF-IDF menjadi pseudo-sequence — tetap kompetitif (97,00% / 90,59%) namun secara konsisten berada di bawah dua model lain, mendukung hipotesis bahwa pendekatan ini tidak benar-benar memanfaatkan kapabilitas pemodelan sekuensial LSTM.

---

## 2. Tujuan

1. Membangun pipeline klasifikasi spam end-to-end: penggabungan data multi-sumber → pembersihan teks → EDA → preprocessing NLP → vektorisasi TF-IDF → pemodelan → evaluasi → analisis error.
2. Membandingkan performa tiga keluarga algoritma classifier (linear, ensemble tree, deep learning sekuensial) di atas representasi fitur yang sama (TF-IDF).
3. Menguji secara kritis apakah arsitektur "TF-IDF + LSTM" — yang umum diklaim di berbagai tutorial/paper — benar-benar memberikan keuntungan dari kapabilitas sekuensial LSTM, atau hanya bekerja sebagai fungsi non-linear biasa.
4. Mengevaluasi dampak strategi penyeimbangan kelas (class balancing) terhadap performa model melalui perbandingan dua skenario dataset.

---

## 3. Dataset

### 3.1 Sumber Data

| Sumber | Jumlah Baris (mentah) | Kolom Asli | Peran dalam Proyek |
|---|---|---|---|
| **Enron** | 517.401 | `file`, `message` | Korpus email korporat yang sah (digunakan sebagai sumber tambahan kelas **ham**) |
| **Nazario** | 1.565 | `sender`, `receiver`, `date`, `subject`, `body`, `urls`, `label` | Korpus phishing publik |
| **SpamAssassin** | 5.796 | `text`, `target` | Korpus campuran spam/ham publik |

Karena ukuran Enron sangat besar dibanding dua sumber lain, dataset ini di-*sample* terlebih dahulu (`random_state=42`) sebelum digabung — inilah yang melahirkan dua skenario (Case 1 & Case 2) di bawah.

### 3.2 Skema Pelabelan

Setiap sumber distandardisasi menjadi tiga kolom yang sama: `text`, `label` (0 = ham, 1 = spam), `source`.

- **Nazario** dan **SpamAssassin**: label diambil langsung dari kolom label/target yang sudah tersedia.
- **Enron**: label ditentukan via heuristik — mengandung substring `"spam"` pada path file → 1, selain itu → 0. Karena struktur folder Enron (mis. `allen-p/_sent_mail/1.`) tidak pernah mengandung kata "spam", secara praktis **seluruh email Enron berlabel ham**. Ini sejalan dengan tujuan penggunaannya sebagai sumber tambahan email sah, namun penting dicatat secara eksplisit karena bukan hal yang langsung terlihat dari kode.

### 3.3 Dua Skenario Dataset

**Case 1 — Skala besar, distribusi alami (imbalanced):**

| Tahap | Jumlah Baris | Catatan |
|---|---|---|
| Enron (sampel) | 7.500 | `enron_df.sample(n=7500, random_state=42)` |
| Gabungan mentah | 14.861 | label 0 = 11.400, label 1 = 3.461 |
| Setelah cleaning + dedup | 14.339 | 522 baris terhapus (duplikat/kosong) |
| Train / Test (80/20) | 11.471 / 2.868 | `random_state=42` |

**Case 2 — Skala kecil, kelas diseimbangkan (balanced):**

| Tahap | Jumlah Baris | Catatan |
|---|---|---|
| Enron (sampel) | 1.000 | `enron_df.sample(n=1000, random_state=42)` |
| Gabungan mentah | 8.361 | label 0 = 4.900, label 1 = 3.461 |
| Setelah undersampling | 6.922 | Kelas ham di-*undersample* agar 3.461 : 3.461 |
| Setelah cleaning + dedup | 6.531 | |
| Train / Test (80/20) | 5.224 / 1.307 | `random_state=42` |

Tujuan dua skenario ini adalah membandingkan performa model pada data yang **lebih besar tapi imbalanced** vs. data yang **lebih kecil tapi balanced** — sebuah trade-off klasik dalam klasifikasi teks.

---

## 4. Metodologi / Pipeline

### 4.1 Penggabungan & Standardisasi Kolom
Empat fungsi (`preprocess_enron1`, `preprocess_enron2`, `preprocess_nazario`, `preprocess_spamassassin`) menyeragamkan setiap sumber ke skema `(text, label, source)`, lalu digabung dengan `pd.concat`.

### 4.2 Pembersihan Teks (Text Cleaning)
Diterapkan terpisah untuk Case 1 dan Case 2 (`clean_text_c1`, `clean_text_c2`), dengan langkah identik:
- Lowercase
- Hapus URL (`http...`, `www...`)
- Hapus alamat email
- Hapus karakter non-huruf (angka, simbol, tanda baca)
- Normalisasi whitespace
- Hapus baris kosong (`dropna`) dan duplikat (`drop_duplicates`)

### 4.3 Exploratory Data Analysis (EDA)
- Distribusi panjang email (histogram) untuk Case 1 & 2.
- Analisis frekuensi kata — top 25 kata tersering (bar chart) untuk Case 1 & 2.
- Auto-EDA opsional via `ydata-profiling` (`ProfileReport`) — disediakan sebagai eksplorasi tambahan yang lebih mendalam (statistik per kolom, korelasi, missing value), namun cukup berat secara komputasi untuk dataset berukuran puluhan ribu baris.

### 4.4 Preprocessing Lanjutan (NLP)
- **Tokenization & Stopword Removal**: `nltk.word_tokenize` + daftar stopword Bahasa Inggris (`stopwords.words('english')`).
- **Lemmatization**: `WordNetLemmatizer`, diterapkan token demi token.

### 4.5 Split Data
`train_test_split` dengan `test_size=0.2`, `random_state=42`, dijalankan terpisah untuk Case 1 dan Case 2 (lihat tabel di Bagian 3.3 untuk ukurannya).

### 4.6 Vectorization (TF-IDF)
```python
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
```
- Case 1 → `X_train_vec_c1`: (11.471, 10.000)
- Case 2 → `X_train_vec_c2`: (5.224, 10.000)

> **Catatan teknis:** objek `tfidf` yang sama di-*fit* dua kali secara berurutan (sekali untuk Case 1, sekali untuk Case 2) di dalam sel yang sama. Vektor yang dihasilkan untuk masing-masing case tetap valid karena dihitung tepat setelah `fit_transform`-nya sendiri, **namun** vectorizer yang disimpan ke `tfidf_vectorizer.pkl` di akhir hanya merefleksikan vocabulary **Case 2** (karena fit terakhir menimpa fit sebelumnya). Lihat Bagian 8 untuk detail.

---

## 5. Pemodelan

### 5.1 Model 1: TF-IDF + SVM
```python
svm_model = SVC(kernel='linear', probability=True, random_state=42)
```
Dilatih langsung di atas vektor TF-IDF (sparse), dijalankan untuk Case 1 dan Case 2 secara independen.

### 5.2 Model 2: TF-IDF + Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
```
Sama seperti Model 1, dilatih langsung di atas vektor TF-IDF, untuk kedua case.

### 5.3 Model 3: TF-IDF + LSTM (Pseudo-Sequence Reshape)

Berbeda dari dua model di atas, Model 3 **tidak** memberi vektor TF-IDF apa adanya ke classifier. Karena layer `LSTM` di Keras mensyaratkan input 3D `(batch, timesteps, features)`, vektor TF-IDF 10.000-dimensi per dokumen di-*reshape* menjadi `(100, 100)` — 100 *timestep* dengan 100 fitur per *timestep* — agar memenuhi syarat bentuk tersebut.

```python
TIMESTEPS = 100
FEATURES_PER_STEP = 100   # 100 x 100 = 10.000 (max_features TF-IDF)

model = Sequential([
    LSTM(64, input_shape=(TIMESTEPS, FEATURES_PER_STEP)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
Dilatih 10 epoch, `batch_size=32`, `validation_split=0.2`, untuk Case 1 dan Case 2 secara independen.

**Catatan metodologis (penting):** Pendekatan ini sengaja dimasukkan sebagai **studi kritis/ablasi**, bukan sebagai rekomendasi arsitektur terbaik. Alasannya:
- TF-IDF mengubah setiap dokumen menjadi satu vektor jarang (*sparse*) dengan panjang tetap, di mana urutan kata sudah hilang sejak awal — setiap dimensi merepresentasikan bobot suatu istilah dalam kosakata, bukan posisi dalam kalimat.
- LSTM membutuhkan input berupa sekuens vektor berbasis langkah waktu yang punya hubungan berurutan secara semantik, agar mekanisme rekursifnya bermanfaat.
- Ketika vektor TF-IDF dipotong menjadi 100 segmen, "timestep ke-N" dan "timestep ke-(N+1)" tidak memiliki hubungan sekuensial apa pun — keduanya hanya kelompok indeks vocabulary yang berdekatan secara kebetulan.
- Akibatnya, jika model ini menghasilkan akurasi yang kompetitif, hal itu lebih mungkin disebabkan oleh kapasitas LSTM sebagai fungsi non-linear yang kuat (mirip MLP kecil dengan gating), bukan karena LSTM "memahami sekuens" dari data.

---

## 6. Hasil Evaluasi

### 6.1 Tabel Perbandingan Metrik

| Model | Case | Akurasi | Precision (spam) | Recall (spam) | F1 (spam) | FPR | ROC-AUC | Jumlah Salah Klasifikasi |
|---|---|---|---|---|---|---|---|---|
| **SVM** | Case 1 | **0,9969** | 1,00 | 0,99 | 0,99 | 0,0014 | 0,9998 | 9 / 2.868 |
| **SVM** | Case 2 | **0,9923** | 1,00 | 0,99 | 0,99 | 0,0029 | 0,9998 | 10 / 1.307 |
| **Random Forest** | Case 1 | 0,9895 | 0,99 | 0,96 | 0,98 | 0,0023 | 0,9993 | 30 / 2.868 |
| **Random Forest** | Case 2 | 0,9748 | 0,98 | 0,97 | 0,97 | 0,0188 | 0,9976 | 33 / 1.307 |
| **LSTM (TF-IDF reshape)** | Case 1 | 0,9700 | 0,95 | 0,92 | 0,93 | 0,0154 | 0,9913 | 86 / 2.868 |
| **LSTM (TF-IDF reshape)** | Case 2 | 0,9059 | 0,92 | 0,88 | 0,90 | 0,0694 | 0,9612 | 123 / 1.307 |

*(Precision/Recall/F1 dilaporkan untuk kelas spam = 1; angka untuk kelas ham = 0 dapat dilihat di classification report lengkap pada notebook.)*

### 6.2 Confusion Matrix & ROC Curve
Setiap model-kasus menghasilkan dua visualisasi di notebook: confusion matrix (heatmap, label Ham/Spam) dan ROC curve. Pola yang konsisten: SVM memiliki confusion matrix paling "bersih" (kesalahan minimal di kedua sel off-diagonal), sementara LSTM menunjukkan jumlah false negative (spam terlewat) yang lebih tinggi dibanding dua model lain, terutama di Case 2.

### 6.3 Error Analysis (Ringkasan)
Jumlah email yang salah diklasifikasikan meningkat tajam seiring kompleksitas/ketidaksesuaian arsitektur dengan representasi fiturnya:

| Model | Case 1 | Case 2 |
|---|---|---|
| SVM | 9 | 10 |
| Random Forest | 30 | 33 |
| LSTM (TF-IDF reshape) | 86 | 123 |

Notebook menyimpan contoh-contoh email yang salah klasifikasi per model (`misclassified_df_*`) untuk inspeksi manual lebih lanjut.

---

## 7. Diskusi

### 7.1 Mengapa SVM & Random Forest Mengungguli LSTM pada Data Ini
SVM dan Random Forest memproses vektor TF-IDF dalam bentuk aslinya — SVM mencari hyperplane pemisah linear di ruang fitur 10.000 dimensi tersebut, Random Forest melakukan split berbasis threshold pada masing-masing dimensi. Keduanya bekerja sesuai dengan struktur data TF-IDF yang sebenarnya (bag-of-words berbobot, tanpa asumsi urutan).

LSTM pada Model 3, sebaliknya, memproses versi vektor yang sudah "dipotong-potong" secara arbitrer, tanpa memperoleh manfaat dari mekanisme rekursifnya karena tidak ada hubungan urutan yang nyata antar potongan tersebut. Hasil 97% (Case 1) dan 90,6% (Case 2) tetap tinggi secara absolut, namun konsisten di bawah dua model lain — sejalan dengan hipotesis pada Bagian 5.3.

### 7.2 Dampak Balancing Kelas (Case 1 vs Case 2)
Menariknya, performa **seluruh model menurun** dari Case 1 ke Case 2, meskipun Case 2 dirancang dengan kelas yang seimbang (1:1). Penurunan paling besar terjadi pada LSTM (97,00% → 90,59%), sementara SVM relatif stabil (99,69% → 99,23%).

Kemungkinan penjelasannya: Case 2 memiliki **volume data latih yang jauh lebih kecil** (5.224 vs 11.471 baris) akibat downsampling Enron + undersampling kelas mayoritas. Pada kasus ini, volume data nampak lebih berpengaruh terhadap performa dibanding keseimbangan kelas semata — terutama untuk model deep learning yang umumnya lebih sensitif terhadap jumlah data latih. Ini merupakan hipotesis yang masuk akal berdasarkan pola hasil yang teramati, namun akan lebih kuat jika diuji lebih lanjut (misalnya dengan menjaga volume data tetap sambil memvariasikan rasio kelas secara terpisah).

### 7.3 Catatan tentang Skema Pelabelan Enron
Karena heuristik pelabelan Enron bergantung pada substring `"spam"` di path file — yang tidak pernah muncul di struktur folder Enron — seluruh kontribusi Enron terhadap dataset gabungan secara praktis berlabel ham. Ini konsisten dengan peran Enron sebagai korpus referensi email sah dalam riset spam-detection, namun perlu dicatat secara eksplisit di dokumentasi karena tidak segera tampak hanya dari membaca kode.

---

## 8. Keterbatasan & Catatan Teknis

1. **TF-IDF vectorizer reused & refit**: objek `tfidf` yang sama dipakai untuk fit Case 1 lalu Case 2 secara berurutan. Vektor yang dipakai untuk training/testing di dalam notebook tetap benar (dihitung tepat setelah fit masing-masing), tetapi file `tfidf_vectorizer.pkl` yang disimpan untuk re-use/inference hanya merefleksikan vocabulary Case 2.
2. **Bug pada penyimpanan split data**: `data_split_c1.pkl` disimpan menggunakan `y_train_c2` (bukan `y_train_c1`) — kemungkinan akibat copy-paste. Ini tidak memengaruhi hasil evaluasi pada notebook (karena variabel in-session tetap benar), tetapi membuat file pickle tersebut tidak konsisten jika dimuat ulang di sesi lain.
3. **Model 3 (LSTM + TF-IDF reshape) bukan model sekuensial yang valid secara teoretis** — sudah dibahas mendalam di Bagian 5.3 & 7.1; hasil performanya sebaiknya dipahami sebagai studi ablasi, bukan rekomendasi arsitektur produksi.
4. **Tidak ada cross-validation maupun hyperparameter tuning sistematis.** Seluruh model menggunakan parameter tetap (default atau ditentukan manual); angka yang dilaporkan berasal dari satu kali train/test split, bukan rata-rata beberapa fold.
5. **Tidak ada early stopping pada training LSTM** — jumlah epoch tetap (10), berisiko overfitting/underfitting tanpa pemantauan otomatis terhadap validation loss.
6. **Auto-EDA (`ydata-profiling`) cukup berat secara komputasi** untuk dataset berukuran puluhan ribu baris; cocok dipakai sesekali untuk eksplorasi manual, kurang ideal jika dijalankan otomatis tiap kali notebook dieksekusi ulang.

---

## 9. Kesimpulan

- **SVM linear** adalah model dengan performa terbaik secara konsisten pada kedua skenario data (akurasi 99,69% di Case 1, 99,23% di Case 2), dengan FPR terendah dan ROC-AUC tertinggi di antara ketiga model.
- **Random Forest** kompetitif namun konsisten sedikit di bawah SVM pada seluruh metrik.
- **LSTM dengan TF-IDF reshape** tetap mencapai performa tinggi secara absolut (97,00% / 90,59%) tetapi konsisten berada di posisi terbawah — mendukung argumen bahwa reshape vektor TF-IDF menjadi pseudo-sequence tidak benar-benar memberi LSTM keuntungan pemodelan sekuensial, karena ketiadaan struktur temporal yang valid pada data hasil reshape tersebut.
- Pada kasus klasifikasi spam-ham email ini, **pemilihan representasi fitur (TF-IDF) dan kesesuaiannya dengan asumsi dasar algoritma** terbukti lebih berpengaruh terhadap performa akhir dibanding sekadar memilih algoritma yang "lebih canggih" secara nominal (deep learning vs classical ML).

---

## 10. Rekomendasi Pengembangan Lanjutan

1. **Bandingkan dengan arsitektur LSTM yang lebih valid secara teoretis**, misalnya:
   - *TF-IDF-weighted embedding*: kalikan embedding tiap token dengan bobot TF-IDF-nya, sambil tetap menjaga urutan token asli.
   - *Sequence of chunk-level TF-IDF*: pecah dokumen menjadi kalimat/paragraf, hitung vektor TF-IDF per potongan, lalu jadikan potongan tersebut sebagai timestep yang benar-benar berurutan.
2. **Lakukan cross-validation (k-fold)** dan **hyperparameter tuning** (GridSearch/RandomizedSearch untuk SVM & Random Forest; tuning units/learning rate/dropout untuk LSTM) agar perbandingan antar model lebih robust secara statistik.
3. **Tambahkan early stopping & model checkpointing** pada training LSTM untuk mencegah overfitting dan menghemat waktu komputasi.
4. **Perbaiki bug teknis** pada Bagian 8 (refit vectorizer, kesalahan variabel pada `data_split_c1.pkl`) sebelum pipeline dipakai untuk inference produksi.
5. **Uji model berbasis Transformer** (mis. DistilBERT/BERT untuk teks Inggris) sebagai pembanding tambahan terhadap pendekatan TF-IDF klasik.
6. **Pisahkan eksperimen volume data vs rasio kelas** pada Case 2 agar dapat disimpulkan secara lebih pasti apakah penurunan performa disebabkan oleh volume data yang lebih kecil, rasio kelas yang berubah, atau kombinasi keduanya.

---

## 11. Lampiran

### 11.1 Lingkungan & Library Utama
`pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, `scikit-learn` (`TfidfVectorizer`, `SVC`, `RandomForestClassifier`, metrics), `nltk` (`stopwords`, `word_tokenize`, `WordNetLemmatizer`), `tensorflow`/`keras` (`Sequential`, `LSTM`, `Dense`, `Dropout`), `ydata-profiling`, `joblib`, `gdown`.

### 11.2 Struktur Notebook (Peta Navigasi)

| Bagian | Isi Utama |
|---|---|
| Setup | Instalasi dependency, import library |
| Load & Combine Datasets | Load 3 sumber data, sampling Enron, standardisasi kolom, penggabungan, undersampling Case 2 |
| EDA | Distribusi panjang email, frekuensi kata, auto-EDA opsional |
| Preprocessing | Text cleaning, tokenization, stopword removal, lemmatization, split data, TF-IDF |
| Training & Testing | Model 1 (SVM), Model 2 (Random Forest), Model 3 (LSTM) — masing-masing untuk Case 1 & 2 |
| Error Analysis | Daftar email yang salah klasifikasi per model |

### 11.3 Ringkasan Hyperparameter

| Model | Hyperparameter |
|---|---|
| SVM | `kernel='linear'`, `probability=True`, `random_state=42` |
| Random Forest | `n_estimators=100`, `max_depth=10`, `random_state=42` |
| LSTM (TF-IDF reshape) | `TIMESTEPS=100`, `FEATURES_PER_STEP=100`, `LSTM(64)`, `Dropout(0.5)`, optimizer=`adam`, loss=`binary_crossentropy`, `epochs=10`, `batch_size=32`, `validation_split=0.2` |
| TF-IDF Vectorizer | `max_features=10000`, `ngram_range=(1, 2)` |

---
