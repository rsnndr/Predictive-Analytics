# Laporan Proyek Machine Learning - Risna Dwi Indriani

## Domain Proyek

Kualitas buah merupakan salah satu faktor utama yang menentukan nilai jual dan kepuasan konsumen di industri pertanian. Dalam dunia agroindustri, proses penyortiran buah seringkali masih dilakukan secara manual, yang rentan terhadap kesalahan manusia, inkonsistensi, serta memakan waktu dan biaya yang cukup besar.

Dengan kemajuan teknologi, khususnya dalam bidang data science dan machine learning, kini memungkinkan untuk melakukan klasifikasi kualitas buah secara otomatis dan lebih akurat. Proyek ini berada dalam domain agroteknologi, yang memadukan ilmu pertanian dengan teknologi informasi untuk meningkatkan efisiensi dan akurasi proses produksi dan distribusi hasil pertanian.

Dalam proyek ini, dilakukan klasifikasi kualitas apel menjadi tiga kelas utama (baik, dan buruk) berdasarkan data numerik seperti berat, warna, dan ukuran. Sistem klasifikasi ini diharapkan dapat digunakan dalam proses grading apel secara otomatis untuk mendukung distribusi dan pengolahan hasil pertanian yang lebih baik.

## Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan kualitas apel secara otomatis dan konsisten tanpa harus dilakukan secara manual oleh manusia?
2. Bagaimana menghindari kesalahan klasifikasi apel yang dapat menyebabkan kerugian ekonomi maupun penurunan kualitas layanan?
3. Algoritma machine learning apa yang paling efektif dan akurat untuk melakukan klasifikasi kualitas apel berdasarkan data numerik seperti berat dan warna?

### Goals

1. Mengembangkan sistem klasifikasi otomatis menggunakan algoritma machine learning agar proses penilaian kualitas apel lebih efisien dan konsisten tanpa campur tangan manual.
2. Mengurangi tingkat kesalahan dalam pengkategorian apel melalui penerapan model klasifikasi yang akurat, sehingga bisa meningkatkan kualitas distribusi dan kepercayaan pelanggan.
3. Mengevaluasi beberapa algoritma machine learning seperti KNN, SVM, ExtraTreesClassifier, LGBMClassifier, dan LabelSpreading untuk menemukan model terbaik dalam mengklasifikasikan kualitas apel.

## Data Understanding

Dataset pada proyek ini berbentuk file CSV yang berisi 4001 sampel dengan 9 fitur, terdiri dari 7 fitur numerik dan 2 fitur kategorikal. Dataset ini memiliki 1 missing value yang perlu ditangani sebelum pemodelan. Data ini berisi karakteristik fisik apel untuk mengklasifikasikan kualitasnya ke dalam dua kelas: baik, dan buruk. Dataset ini berasal dari pengukuran manual atau sumber publik, dan dapat diakses di https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

- A_id : Pengidentifikasi unik untuk setiap buah
- Size : Ukuran buah
- Weight : Berat buah
- Sweetness : Tingkat kemanisan buah
- Crunchiness : Tekstur yang menunjukkan kerenyahan buah - Juiciness : Tingkat kesegaran buah
- Ripeness : Tahap kematangan buah
- Acidity : Tingkat keasaman buah
- Quality : Kualitas buah secara keseluruhan

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses persiapan data sebelum digunakan dalam pemodelan. Teknik-teknik data preparation yang dilakukan antara lain:

1. Menghapus Kolom yang Tidak Relevan
   
Kolom A_id dihapus karena hanya berfungsi sebagai identifikasi dan tidak memberikan informasi yang berguna untuk prediksi kualitas.

2. Menangani Missing Value
   
Pengecekan dilakukan untuk mengetahui apakah terdapat data yang hilang (missing values). Jika ditemukan, maka akan dilakukan penanganan seperti:
  - Mengisi nilai kosong dengan rata-rata/median jika merupakan fitur numerik.
  - Menghapus baris jika data terlalu banyak yang kosong dan tidak dapat diimputasi secara tepat.

3. Menangani Outlier

Outlier pada fitur numerik seperti Weight, Sweetness, atau Acidity dapat mempengaruhi hasil model. Oleh karena itu, dilakukan deteksi menggunakan metode statistik (seperti IQR).

4. Encoding Label Target

Variabel Quality merupakan label kategorikal yang terdiri dari kelas-kelas seperti baik dan buruk. Oleh karena itu, dilakukan proses label encoding untuk mengubah nilai-nilai tersebut menjadi format numerik agar dapat diproses oleh algoritma machine learning (contohnya: baik = 1, buruk = 0).

5. Normalisasi Data

Fitur-fitur numerik seperti Weight, Sweetness, dan Acidity memiliki skala yang berbeda-beda, maka dilakukan normalisasi agar semua fitur berada pada rentang nilai yang seragam. Normalisasi ini membantu meningkatkan performa model terutama yang berbasis jarak seperti KNN.

6. Train-Test Split

Dataset dibagi menjadi dua bagian, yaitu data latih (train) sebanyak 80% dan data uji (test) sebanyak 20%. Pembagian ini bertujuan agar model dapat belajar dari data latih dan kemudian diuji performanya pada data uji yang belum pernah dilihat sebelumnya. Dengan begitu, kita bisa mengevaluasi kemampuan model dalam memprediksi data baru secara objektif.

## Modeling

Pada tahap ini, dilakukan pemodelan dengan menggunakan beberapa algoritma machine learning untuk memprediksi kualitas buah berdasarkan fitur-fitur seperti ukuran, berat, tingkat kemanisan, kerenyahan, dan lainnya. Model-model yang digunakan merupakan campuran dari algoritma supervised dan semi-supervised learning. Berikut adalah 5 algoritma yang digunakan pada proyek, yaitu:

1. K-Nearest Neighbors (KNN)
merupakan algoritma machine learning yang tergolong sederhana dan intuitif, digunakan untuk tugas klasifikasi maupun regresi. Cara kerjanya adalah dengan mencari k data terdekat dari suatu data baru, lalu memprediksi kelas atau nilai data tersebut berdasarkan mayoritas label atau rata-rata dari tetangga-tetangganya. Parameter yang digunakan pada proyek ini adalah:
- n_neighbors= 5
- weights= distance

2. ExtraTreesClassifier
adalah algoritma ensemble berbasis pohon keputusan yang menggunakan banyak pohon untuk membuat prediksi yang lebih akurat dan stabil. Berbeda dengan Random Forest, ExtraTrees memperkenalkan lebih banyak randomisasi saat membangun setiap pohon, seperti pemilihan fitur dan nilai split secara acak, yang sering kali menghasilkan kecepatan pelatihan lebih tinggi dan mengurangi overfitting. Parameter yang digunakan pada proyek ini adalah:
- n_estimators= 100
- max_depth= 10
- n_jobs= 2
- random_state= 100

3. LGBMClassifier
adalah algoritma machine learning berbasis gradient boosting yang dikembangkan untuk efisiensi dan kecepatan tinggi. Algoritma ini membangun model secara bertahap dengan menambahkan pohon keputusan yang fokus pada kesalahan dari model sebelumnya. LGBM dirancang agar bisa menangani dataset besar dan kompleks dengan waktu pelatihan yang singkat. Parameter yang digunakan pada proyek ini adalah:
- n_estimators= 100
- max_depth= 10
- n_jobs= 2
- random_state= 100

4. SVC (Support Vector Classifier)
adalah salah satu jenis algoritma dari Support Vector Machine (SVM) yang digunakan untuk tugas klasifikasi. SVC bekerja dengan mencari hyperplane terbaik yang memisahkan data dari kelas yang berbeda dengan margin terbesar. Jika data tidak dapat dipisahkan secara linear, SVC dapat menggunakan kernel trick (seperti RBF, polynomial, sigmoid) untuk memetakan data ke dimensi yang lebih tinggi agar bisa dipisahkan. Parameter yang digunakan pada proyek ini adalah:

5. LabelSpreading
adalah algoritma semi-supervised learning yang digunakan untuk mengklasifikasikan data dengan cara menyebarkan label dari data yang sudah berlabel ke data yang belum berlabel berdasarkan kedekatan atau kemiripan antar data. Algoritma ini memanfaatkan struktur grafik data untuk memprediksi label data tanpa label dengan iterasi penyebaran informasi. Parameter yang digunakan pada proyek ini adalah:

## Evaluation
Dalam proyek ini, kami menggunakan beberapa metrik evaluasi untuk mengukur performa model klasifikasi, yaitu:

- Accuracy: Mengukur persentase prediksi yang benar dari keseluruhan data.
- Precision: Mengukur ketepatan model dalam memprediksi kelas positif, yaitu seberapa banyak prediksi positif yang benar-benar positif.
- Recall: Mengukur kemampuan model dalam menangkap seluruh data positif yang sebenarnya, yaitu seberapa banyak data positif yang berhasil terdeteksi.
- F1 Score: Merupakan rata-rata harmonis antara precision dan recall, yang memberikan keseimbangan antara kedua metrik tersebut.
  | Model                           | Accuracy | Precision | Recall | F1 Score |
| ------------------------------- | -------- | --------- | ------ | -------- |
| K-Nearest Neighbors             | 0.89     | 0.89      | 0.89   | 0.89     |
| ExtraTreesClassifier            | 0.85     | 0.85      | 0.85   | 0.85     |
| LGBMClassifier                  | 0.88     | 0.88      | 0.88   | 0.88     |
| Support Vector Classifier (SVC) | 0.88     | 0.88      | 0.88   | 0.88     |
| LabelSpreading                  | 0.88     | 0.88      | 0.88   | 0.88     |

Model K-Nearest Neighbors (KNN) memiliki performa terbaik dengan nilai akurasi, precision, recall, dan F1 score sebesar 0.89, menunjukkan prediksi yang akurat dan seimbang. Model ExtraTreesClassifier memiliki hasil sedikit lebih rendah, tapi semua model secara umum memberikan performa baik dengan akurasi di atas 85%. Oleh karena itu, KNN menjadi pilihan terbaik untuk klasifikasi pada dataset ini.
