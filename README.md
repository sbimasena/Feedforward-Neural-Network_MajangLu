# Feedforward-Neural-Network_MajangLu

Tugas Besar 1 IF3270 Pembelajaran Mesin.

Proyek ini berisi implementasi Feedforward Neural Network (FFNN) dari scratch untuk tugas klasifikasi biner, disertai rangkaian eksperimen untuk menganalisis pengaruh:
- width (jumlah neuron per layer),
- depth (jumlah hidden layer),
- activation function,
- learning rate,
- regularisasi (none, L1, L2),
- serta perbandingan dengan `sklearn.neural_network.MLPClassifier`.

## Struktur Proyek

```text
Feedforward-Neural-Network_MajangLu/
|-- README.md
`-- src/
	|-- activation.py
	|-- layer.py
	|-- loss.py
	|-- model.py
	|-- datasetml_2026.csv
	`-- pipeline.ipynb
```

## Penjelasan Singkat Modul

- `activation.py`: fungsi aktivasi `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`.
- `layer.py`: kelas `DenseLayer` (inisialisasi bobot, forward, backward, update).
- `loss.py`: fungsi loss (`MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`).
- `model.py`: kelas `Neural_Network` (add layer, train, predict, visualisasi distribusi bobot/gradien).
- `pipeline.ipynb`: notebook utama untuk preprocessing data, eksperimen, visualisasi, dan ringkasan hasil.

## Kebutuhan Lingkungan

Disarankan Python 3.10+.

Install dependensi:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Cara Menjalankan

1. Masuk ke folder proyek.
2. Jalankan Jupyter Notebook:

```bash
jupyter notebook
```

3. Buka `src/pipeline.ipynb`.
4. Jalankan sel secara berurutan dari atas ke bawah.

Urutan eksekusi yang direkomendasikan di notebook:
- setup import dan seed,
- preprocessing,
- helper function,
- eksperimen width,
- eksperimen depth,
- eksperimen activation function,
- eksperimen learning rate + regularisasi,
- ringkasan hasil,
- baseline sklearn.

## Ringkasan Eksperimen

Notebook mencakup eksperimen berikut:

1. Variasi width: membandingkan beberapa jumlah neuron hidden layer.
2. Variasi depth: membandingkan beberapa jumlah hidden layer.
3. Variasi activation function: membandingkan aktivasi pada arsitektur dasar (minimal 3 layer).
4. Variasi learning rate dengan regularisasi:
   - tanpa regularisasi,
   - regularisasi L1,
   - regularisasi L2.

Setiap eksperimen menghasilkan:
- metrik akhir (accuracy, precision, recall, F1),
- nilai akhir training loss dan validation loss,
- grafik loss per epoch,
- visualisasi distribusi bobot dan gradien (sesuai blok eksperimen).

## Reproducibility

Proyek sudah diatur agar hasil eksperimen reproducible dengan pengaturan seed yang konsisten di notebook.

Agar hasil tetap konsisten saat rerun:
- jalankan sel secara berurutan,
- hindari menjalankan sel eksperimen secara acak tanpa menjalankan ulang sel setup/helper,
- gunakan kernel yang sama dan restart kernel bila ingin mengulang dari awal secara bersih.

## Pembuat

| Nama                             | NIM      | Tugas |
|----------------------------------|----------|----------------------|
| Wardatul Khoiroh                 | 13523001 | Mengimplementasikan fungsi Loss dan turunan pertamanya, inisialisasi bobot, mekanisme regulasi L1 dan L2, serta fungsi pelengkap dan bonus init bobot|
| Indah Novita Tangdililing        | 13523047 | Melakukan pemrosesan dataset di file Jupyter Notebook, Menjalankan eksperimen hyperparameter, regularisasi, dan perbandingan MLPClassifier|
| Sakti Bimasena                   | 13523053 | Membuat model utama FFNN dan kelas-kelasnya, Bonus Fungsi Aktivasi|
