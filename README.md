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

### Solution Statements
- Melakukan analisis data menggunakan metode univariat dan multivariat untuk memahami karakteristik data secara menyeluruh.
- Memanfaatkan visualisasi data guna memperkuat pemahaman, serta mengidentifikasi korelasi antar fitur dan mendeteksi outlier.
- Melaksanakan proses pembersihan data (data cleaning) untuk menghilangkan kesalahan atau data yang tidak valid.
- Melakukan normalisasi data agar skala fitur menjadi seragam, sehingga meningkatkan performa model prediksi.
- Membuat beberapa variasi model untuk memperoleh model terbaik dalam prediksi kualitas apel. Model-model yang digunakan antara lain:
  1.  K-Nearest Neighbors (KNN) merupakan algoritma sederhana yang mengklasifikasikan data baru berdasarkan kedekatan dengan data tetangga terdekat, dengan memberikan bobot lebih pada tetangga yang lebih dekat.
  2. ExtraTreesClassifier adalah algoritma ensemble yang menggunakan banyak pohon keputusan acak untuk meningkatkan akurasi prediksi melalui agregasi hasil dari tiap pohon.
  3. LGBMClassifier adalah model gradient boosting yang efisien dan cepat, menggabungkan beberapa pohon keputusan secara bertahap untuk meningkatkan performa klasifikasi.
  4. Support Vector Classifier (SVC) adalah algoritma yang mencari hyperplane optimal untuk memisahkan kelas dalam ruang fitur, efektif untuk klasifikasi dengan data berdimensi tinggi.
  5. LabelSpreading adalah metode semi-supervised learning yang menyebarkan label dari data berlabel ke data tak berlabel berdasarkan kemiripan antar data, sehingga dapat memanfaatkan data tak berlabel dalam proses pembelajaran.

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
   
Kolom A_id dihapus dari dataset karena hanya berfungsi sebagai penanda atau identifikasi data saja. Kolom ini tidak mengandung informasi yang relevan atau berkontribusi dalam proses prediksi kualitas, sehingga penghapusannya bertujuan untuk menyederhanakan data dan meningkatkan efisiensi pemodelan.

2. Menangani Missing Value
   
Pengecekan missing values dilakukan untuk mengetahui apakah terdapat data yang hilang dalam dataset. Apabila ditemukan nilai kosong, langkah penanganan yang dilakukan adalah mengisi nilai tersebut dengan rata-rata atau median pada fitur numerik agar data tetap lengkap dan dapat dianalisis. Namun, jika jumlah data kosong pada suatu baris terlalu banyak dan imputasi tidak memungkinkan secara akurat, maka baris tersebut akan dihapus untuk menjaga kualitas dan konsistensi data.

3. Mengubah Tipe Data
Pada tahap preprocessing, kolom Acidity diubah tipe datanya menjadi float64 menggunakan fungsi .astype(). Hal ini dilakukan agar data pada kolom tersebut dapat diproses secara numerik untuk analisis statistik dan pemodelan selanjutnya, karena sebelumnya data tersebut masih dalam format string atau tipe lain yang tidak mendukung operasi matematika.

4. Encoding Label Target

Variabel Quality merupakan label kategorikal yang terdiri dari kelas-kelas seperti baik dan buruk. Oleh karena itu, dilakukan proses label encoding untuk mengubah nilai-nilai tersebut menjadi format numerik agar dapat diproses oleh algoritma machine learning (contohnya: baik = 1, buruk = 0).

5. Menangani Outlier

Pada tahap ini, outlier atau data ekstrim pada variabel numerik diidentifikasi dan dihapus menggunakan metode Interquartile Range (IQR). Data yang berada di luar rentang Q1 - 1.5*IQR sampai Q3 + 1.5*IQR dianggap outlier dan dihilangkan. Proses ini bertujuan untuk meningkatkan kualitas data agar analisis dan pemodelan lebih akurat dan tidak terpengaruh oleh nilai-nilai ekstrem.

6. Memisahkan fitur dan target
Pada tahap ini, data dipisahkan menjadi dua bagian, yaitu fitur (X) dan target (y). Fitur adalah kumpulan variabel yang digunakan sebagai input untuk model, sedangkan target adalah variabel yang ingin diprediksi, dalam hal ini kolom Quality. Pemisahan ini diperlukan agar model dapat mempelajari pola dari fitur untuk memprediksi target dengan tepat.

7. Train-Test Split

Dataset dibagi menjadi dua bagian, yaitu data latih (train) sebanyak 80% dan data uji (test) sebanyak 20%. Pembagian ini bertujuan agar model dapat belajar dari data latih dan kemudian diuji performanya pada data uji yang belum pernah dilihat sebelumnya. Dengan begitu, kita bisa mengevaluasi kemampuan model dalam memprediksi data baru secara objektif.

8. Normalisasi Data

Fitur-fitur numerik seperti Weight, Sweetness, dan Acidity memiliki skala yang berbeda-beda, maka dilakukan normalisasi agar semua fitur berada pada rentang nilai yang seragam. Normalisasi ini membantu meningkatkan performa model terutama yang berbasis jarak seperti KNN.

## Modeling

Pada tahap ini, dilakukan pemodelan dengan menggunakan beberapa algoritma machine learning untuk memprediksi kualitas buah berdasarkan fitur-fitur seperti ukuran, berat, tingkat kemanisan, kerenyahan, dan lainnya. Model-model yang digunakan merupakan campuran dari algoritma supervised dan semi-supervised learning. Berikut adalah 5 algoritma yang digunakan pada proyek, yaitu:

1. K-Nearest Neighbors (KNN)
merupakan algoritma machine learning yang tergolong sederhana dan intuitif, digunakan untuk tugas klasifikasi maupun regresi. Cara kerjanya adalah dengan mencari k data terdekat dari suatu data baru, lalu memprediksi kelas atau nilai data tersebut berdasarkan mayoritas label atau rata-rata dari tetangga-tetangganya.

Parameter yang digunakan pada proyek ini adalah:
   - n_neighbors= 5
   - weights= distance

Kelebihan:
   - Sederhana dan mudah dipahami.
   - Tidak perlu pelatihan model (lazy learner).
   - Cocok untuk data yang tidak linear.

Kekurangan:
   - Lambat saat prediksi pada data besar (karena menghitung jarak ke semua data).
   - Sensitif terhadap fitur yang memiliki skala berbeda.
   - Rentan terhadap data outlier dan noise.

2. ExtraTreesClassifier
adalah algoritma ensemble berbasis pohon keputusan yang menggunakan banyak pohon untuk membuat prediksi yang lebih akurat dan stabil. Berbeda dengan Random Forest, ExtraTrees memperkenalkan lebih banyak randomisasi saat membangun setiap pohon, seperti pemilihan fitur dan nilai split secara acak, yang sering kali menghasilkan kecepatan pelatihan lebih tinggi dan mengurangi overfitting.

Parameter yang digunakan pada proyek ini adalah:
   - n_estimators= 100
   - max_depth= 10
   - n_jobs= 2
   - random_state= 100

Kelebihan:
   - Cepat karena memilih split secara acak.
   - Tidak mudah overfitting.
   - Dapat menangani data dengan banyak fitur.

Kekurangan:
   - Interpretasi hasil bisa sulit.
   - Butuh memori besar saat jumlah pohon banyak.

3. LGBMClassifier
adalah algoritma machine learning berbasis gradient boosting yang dikembangkan untuk efisiensi dan kecepatan tinggi. Algoritma ini membangun model secara bertahap dengan menambahkan pohon keputusan yang fokus pada kesalahan dari model sebelumnya. LGBM dirancang agar bisa menangani dataset besar dan kompleks dengan waktu pelatihan yang singkat.

Parameter yang digunakan pada proyek ini adalah:
   - n_estimators= 100
   - max_depth= 10
   - n_jobs= 2
   - random_state= 100

Kelebihan:
   - Sangat cepat dan efisien.
   - Bisa menangani dataset besar dan fitur tinggi.
   - Mendukung categorical features secara langsung.

Kekurangan:
   - Bisa overfitting jika tidak diatur dengan baik.
   - Kurang cocok untuk data yang kecil atau noisy.

4. SVC (Support Vector Classifier)
adalah salah satu jenis algoritma dari Support Vector Machine (SVM) yang digunakan untuk tugas klasifikasi. SVC bekerja dengan mencari hyperplane terbaik yang memisahkan data dari kelas yang berbeda dengan margin terbesar. Jika data tidak dapat dipisahkan secara linear, SVC dapat menggunakan kernel trick (seperti RBF, polynomial, sigmoid) untuk memetakan data ke dimensi yang lebih tinggi agar bisa dipisahkan.

Kelebihan:
   - Efektif pada data berdimensi tinggi.
   - Bisa digunakan untuk data non-linear dengan kernel.
   - Hasilnya biasanya sangat akurat.

Kekurangan:
   - Lambat saat data banyak.
   - Perlu tuning parameter yang tepat.
   - Tidak cocok untuk dataset besar.

5. LabelSpreading
adalah algoritma semi-supervised learning yang digunakan untuk mengklasifikasikan data dengan cara menyebarkan label dari data yang sudah berlabel ke data yang belum berlabel berdasarkan kedekatan atau kemiripan antar data. Algoritma ini memanfaatkan struktur grafik data untuk memprediksi label data tanpa label dengan iterasi penyebaran informasi.

Kelebihan:
   - Bisa digunakan saat label sangat sedikit.
   - Memanfaatkan informasi dari data tak berlabel.
   - Cocok untuk masalah di mana labeling mahal.

Kekurangan:
   - Butuh asumsi bahwa data yang serupa punya label yang sama.
   - Kurang efektif jika struktur data tidak jelas atau noisy.
   - Kurang cocok untuk data yang terlalu besar atau sparse.

Alasan KNN dipilih:

Model K-Nearest Neighbors (KNN) dipilih sebagai yang terbaik karena mencatat nilai akurasi, precision, recall, dan F1-score sebesar 0.89, menunjukkan performa prediksi yang akurat dan seimbang.
Meskipun ExtraTreesClassifier juga menunjukkan hasil baik, KNN unggul secara keseluruhan dan paling optimal untuk klasifikasi kualitas apel dalam dataset ini.

## Evaluation
Dalam proyek ini, kami menggunakan beberapa metrik evaluasi untuk mengukur performa model klasifikasi, yaitu:

- Accuracy: Mengukur persentase prediksi yang benar dari keseluruhan data.
Accuracy = (TP + TN) / (TP + TN + FP + FN)

- Precision: Mengukur ketepatan model dalam memprediksi kelas positif, yaitu seberapa banyak prediksi positif yang benar-benar positif.
Precision = TP / (TP + FP)

- Recall: Mengukur kemampuan model dalam menangkap seluruh data positif yang sebenarnya, yaitu seberapa banyak data positif yang berhasil terdeteksi.
Recall = TP / (TP + FN)

- F1 Score: Merupakan rata-rata harmonis antara precision dan recall, yang memberikan keseimbangan antara kedua metrik tersebut.
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

Keterangan:
TP = True Positive (jumlah data positif yang diprediksi benar oleh model)
TN = True Negative (jumlah data negatif yang diprediksi benar oleh model)
FP = False Positive (jumlah data negatif yang salah diprediksi sebagai positif)
FN = False Negative ( jumlah data positif yang salah diprediksi sebagai negatif)
  
| Model                           | Accuracy | Precision | Recall | F1 Score |
| ------------------------------- | -------- | --------- | ------ | -------- |
| K-Nearest Neighbors             | 0.89     | 0.89      | 0.89   | 0.89     |
| ExtraTreesClassifier            | 0.85     | 0.85      | 0.85   | 0.85     |
| LGBMClassifier                  | 0.88     | 0.88      | 0.88   | 0.88     |
| Support Vector Classifier (SVC) | 0.88     | 0.88      | 0.88   | 0.88     |
| LabelSpreading                  | 0.88     | 0.88      | 0.88   | 0.88     |

Penjelasan Hasil: 
- Model K-Nearest Neighbors (KNN) memiliki performa terbaik dengan nilai accuracy, precision, recall, dan f1 score tertinggi yaitu 0.89, sehingga sangat baik dalam mengklasifikasikan kualitas apel secara otomatis dan konsisten.

- Model lain seperti LGBMClassifier, SVC, dan LabelSpreading juga menunjukkan performa yang cukup baik dengan nilai metrik sekitar 0.88, sedangkan ExtraTreesClassifier memiliki performa sedikit lebih rendah yaitu 0.85.

- Penggunaan metrik precision dan recall penting agar model tidak hanya akurat secara keseluruhan, tetapi juga teliti dalam memprediksi kelas kualitas apel, sehingga mengurangi kesalahan klasifikasi yang dapat berdampak ekonomi.

- F1 Score sebagai keseimbangan antara precision dan recall memberikan gambaran performa model yang komprehensif, terutama dalam kasus data yang mungkin tidak seimbang.

Kesimpulan:
- Metrik evaluasi yang digunakan sudah sesuai dengan konteks klasifikasi kualitas apel, di mana ketepatan dan kemampuan deteksi kelas positif sangat penting untuk menghindari kesalahan klasifikasi yang dapat merugikan. Dari hasil evaluasi, KNN menjadi pilihan model paling efektif untuk masalah ini.


