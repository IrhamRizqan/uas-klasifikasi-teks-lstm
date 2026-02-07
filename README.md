# Dashboard Klasifikasi Teks Berita (LSTM)

Repository ini berisi implementasi dashboard interaktif untuk tugas besar mata kuliah
Pemrograman Dasar Sains Data. Model klasifikasi yang digunakan adalah Long Short-Term Memory (LSTM) yang dilatih
menggunakan dataset BBC News.

---

### Informasi Tugas
- Mata Kuliah : Pemrograman Dasar Sains Data
- Jenis Tugas : Tugas Akhir Semester (Kelompok)
- Program Studi : Teknik Informatika

### Anggota Kelompok
1. Rashad Shaquille Taofik 10124050
2. Hilyati Raihatul Jannah 10124053
3. Irham Rizqan Zakiy 10124055
4. Siti Atsiilah Maisaan Nurhaeni 10124057
5. Olivia Nathania Sipayung 10124087
6. Tabhita Gialaista 10124093

---

## Fitur Utama
- Eksplorasi dataset berita (jumlah data, distribusi kategori, contoh teks)
- Prediksi kategori teks berita menggunakan model LSTM
- Visualisasi probabilitas hasil prediksi
- Ringkasan performa model
- Antarmuka dashboard interaktif berbasis Streamlit

---

## Struktur Direktori
```
├── dashboard/
│ └── app.py
├── model/
│ ├── LSTM-Model_UAS.keras
│ ├── tokenizer.pkl
│ └── label_encoder.pkl
├── data/
│ └── BBC_News_processed.csv
├── requirements.txt
└── README.md
```

---

## Instalasi dan Menjalankan Aplikasi

### 1. Clone repository
```bash
git clone https://github.com/irhamrzq/Klasifikasi-Teks-Berita-LSTM.git
cd Klasifikasi-Teks-Berita-LSTM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run dashboard/app.py
```

---

### Catatan Model

Model dilatih menggunakan teks berita berbahasa Inggris.
Prediksi teks berbahasa Indonesia bersifat opsional dan digunakan untuk demonstrasi antarmuka.
Evaluasi performa model secara lengkap disajikan pada notebook analisis terpisah.

---

### Dataset

Dataset yang digunakan adalah BBC News Dataset yang terdiri dari lima kategori:

- Business
- Entertainment
- Politics
- Sport
- Tech

Dataset telah melalui tahap preprocessing sebelum digunakan untuk pelatihan model.

---