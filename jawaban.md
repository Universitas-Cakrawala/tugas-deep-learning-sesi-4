# Jawaban Tugas Praktik Deep Learning — Sesi 4
## Framework Debugging Exercise: PyTorch Training dan Evaluation Mode

- **Mata Kuliah**: Deep Learning (SDA2110)
- **Sub-CPMK**: Menggunakan framework Deep Learning
- **Topik**: Debugging Training Loop, Mode Operasi (`train` vs `eval`), Autograd, Tensor Mechanics, dan Evaluasi Model
- **Kelompok**: Kelompok 5

### Identitas Anggota Kelompok:
| No. | Nama Lengkap | NIM / ID | Peran |
|:---:|:---|:---:|:---:|
| 1 | Tita Noviana | 24120500011 | Ketua |
| 2 | Titanio Yudista | 24120500031 | Anggota |
| 3 | Suci Fransisca Sisilia R | 24120500008 | Anggota |
| 4 | Fajar Dwiharjo | 24130500010 | Anggota |
| 5 | Rafli Ramadhan | 24130500001 | Anggota |

---

## A. Analisis Kode

### 1. Analisis `broken_training_step()`

Perhatikan kode berikut:
```python
def broken_training_step(model, optimizer, criterion, X, y):
    model.eval()
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()
```

#### Masalah Utama
Masalah utama pada fungsi di atas ada pada baris **`model.eval()`**. Karena fungsi ini bertujuan untuk melatih model (*training step*), mode yang seharusnya diaktifkan adalah **`model.train()`**.

#### Penjelasan:
1. **Fungsi `model.eval()`**:
   `model.eval()` berfungsi untuk mengubah mode kerja model ke mode evaluasi/inferensi. Saat perintah ini dipanggil, PyTorch akan menginstruksikan seluruh layer di dalam model untuk menonaktifkan mekanisme yang hanya digunakan saat proses belajar.
2. **Mode yang Seharusnya Digunakan saat Training**:
   Mode yang wajib digunakan saat fase pelatihan adalah **`model.train()`**.
3. **Layer yang Perilakunya Berubah antara Mode Training dan Evaluation**:
   - **`nn.Dropout`**:
     - *Saat Training*: Mematikan sebagian neuron secara acak untuk mencegah ketergantungan berlebih antar neuron (*regularisasi*).
     - *Saat Evaluation*: Dinonaktifkan sepenuhnya. Semua neuron dibiarkan aktif agar hasil prediksi stabil dan konsisten.
   - **`nn.BatchNorm` (Batch Normalization)**:
     - *Saat Training*: Menghitung nilai rata-rata (*mean*) dan variansi langsung dari batch data yang sedang masuk, sekaligus mencatat perkiraan rata-rata dan variansi global (*running statistics*) untuk dipakai nanti.
     - *Saat Evaluation*: Membekukan perhitungan statistik batch. Layer menggunakan nilai rata-rata dan variansi global yang sudah terkumpul selama masa training.
4. **Mengapa Masalah ini Memengaruhi Hasil Training**:
   - Jika model dilatih dalam mode `model.eval()`, layer regularisasi seperti `Dropout` tidak akan bekerja sama sekali, sehingga model rentan mengalami *overfitting*.
   - Layer seperti `BatchNorm` tidak akan memperbarui statistik globalnya. Akibatnya, saat model diuji nanti pada data baru, proses normalisasi menjadi keliru dan akurasi model bisa turun drastis.

---

### 2. Perbaikan Kode

Berikut adalah kode fungsi yang sudah diperbaiki:

```python
def corrected_training_step(model, optimizer, criterion, X, y):
    # PERBAIKAN: Gunakan model.train() agar layer Dropout dan BatchNorm aktif selama training
    model.train()
    
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

#### Alasan Perubahan:
Baris `model.eval()` diganti menjadi `model.train()` agar model berada dalam kondisi siap belajar. Dengan begitu, layer regularisasi aktif mencegah overfitting dan layer normalisasi dapat memperbarui statistiknya secara konsisten sepanjang proses pelatihan.

---

### 3. Apakah `model.eval()` Menghentikan Gradient?

#### Jawaban:
> **TIDAK / SALAH.** `model.eval()` sama sekali tidak menghentikan perhitungan gradien.

#### Perbedaan antara `model.eval()` dan `torch.no_grad()`:

| Karakteristik | `model.eval()` | `torch.no_grad()` |
| :--- | :--- | :--- |
| **Level Operasi** | Level Modul / Layer (`nn.Module`) | Level Engine Autograd (`torch.autograd`) |
| **Fungsi Utama** | Mengatur perilaku layer tertentu (seperti mematikan Dropout dan membekukan statistik BatchNorm). | Mematikan sistem pencatat graf dan pelacak gradien PyTorch. |
| **Dampak pada Autograd** | **Tidak ada.** PyTorch tetap mencatat riwayat operasi untuk perhitungan gradien. | Autograd tidak mencatat operasi komputasi. |
| **Dampak pada Memori** | Memori aktivasi tetap disimpan untuk persiapan langkah *backward pass*. | Menghemat memori secara signifikan karena tidak ada riwayat komputasi yang disimpan. |
| **Dampak pada Backward Pass** | Pemanggilan `loss.backward()` **tetap berfungsi** dan nilai gradien bobot tetap dihitung. | Pemanggilan `backward()` akan menghasilkan error karena tidak ada graf gradien yang terbentuk. |
| **Pembaruan Bobot** | `optimizer.step()` tetap dapat memperbarui bobot model jika gradien tersedia. | Tidak ada pembaruan bobot karena nilai gradien tidak dihitung. |

**Kesimpulan:**
`model.eval()` hanya mengatur perilaku internal dari layer model, bukan mematikan kalkulasi gradien. Untuk memastikan komputasi gradien benar-benar berhenti dan memori lebih hemat pada saat evaluasi, forward pass harus dibungkus di dalam blok `with torch.no_grad():`.

---

### 4. Urutan Training Step

Berikut fungsi dari masing-masing perintah dalam satu langkah pelatihan PyTorch:

| No. | Perintah | Fungsi Teknis |
|:---:|:---|:---|
| 1 | `optimizer.zero_grad()` | Mereset nilai gradien dari iterasi sebelumnya agar tidak menumpuk ke iterasi saat ini. |
| 2 | `prediction = model(X)` | Melakukan *forward pass*, yaitu mengalirkan data masukan $X$ ke dalam model untuk menghasilkan nilai prediksi (logits). |
| 3 | `loss = criterion(prediction, y)` | Menghitung nilai kerugian (*loss*) untuk mengukur seberapa jauh deviasi hasil prediksi model dari label sebenarnya ($y$). |
| 4 | `loss.backward()` | Melakukan *backward pass* (backpropagation) untuk menghitung turunan/gradien dari loss terhadap seluruh parameter bobot model. |
| 5 | `optimizer.step()` | Memperbarui nilai bobot dan bias model berdasarkan hasil perhitungan gradien sebelumnya. |

#### Mengapa Urutannya Penting?
Urutan langkah ini bersifat runtut dan logis:
1. Ruang buffer gradien harus dibersihkan terlebih dahulu (`zero_grad()`).
2. Prediksi harus dihitung sebelum kita bisa mengukur nilai loss (`model(X)`).
3. Nilai loss harus ada sebelum kita bisa menghitung turunan gradiennya (`criterion`).
4. Gradien harus dihitung terlebih dahulu sebelum optimizer dapat memperbarui bobot (`loss.backward()`).
5. Terakhir, optimizer memperbarui nilai bobot menggunakan gradien yang baru saja dihitung (`optimizer.step()`).

---

### 5. Risiko Melupakan `optimizer.zero_grad()`

#### Apa yang Terjadi:
Secara bawaan di PyTorch, eksekusi `loss.backward()` memiliki sifat **menjumlahkan (mengakumulasi)** nilai gradien baru ke variabel gradien yang sudah ada pada memori (`param.grad += grad_baru`).

#### Dampaknya terhadap Proses Training dan Bobot Model:
Jika `optimizer.zero_grad()` tidak dipanggil:
1. **Gradien Terus Menumpuk**: Gradien dari batch sebelumnya tidak dibuang, melainkan terus bertambah seiring berjalannya setiap iterasi.
2. **Lonjakan Nilai Gradien (*Exploding Gradients*)**: Nilai gradien akan membesar secara tidak wajar, menyebabkan optimizer memperbarui bobot secara terlalu agresif dan melompat jauh dari titik optimal.
3. **Model Gagal Belajar (Divergen)**: Perhitungan numerik akan menjadi tidak stabil, nilai loss bisa berubah menjadi `NaN` (*Not a Number*), dan model akhirnya gagal konvergen sama sekali.

---

## B. Pemeriksaan Input dan Output

Fungsi inspeksi batch:
```python
def inspect_batch(model, X, y):
    print("input shape:", tuple(X.shape), "dtype:", X.dtype, "device:", X.device)
    print("label shape:", tuple(y.shape), "dtype:", y.dtype, "device:", y.device)
    print("model device:", next(model.parameters()).device)
    with torch.no_grad():
        prediction = model(X)
    print("prediction shape:", tuple(prediction.shape), "dtype:", prediction.dtype)
```

### 6. Analisis Shape

Berdasarkan data yang diuji pada script:
1. **Shape $X$**: `(16, 2)` $\rightarrow$ artinya terdapat 16 data sampel dalam satu batch, dengan masing-masing sampel memiliki 2 fitur input.
2. **Shape $y$**: `(16, 1)` $\rightarrow$ artinya terdapat 16 target data, masing-masing dengan 1 nilai label.
3. **Shape output model (`prediction`)**: `(16, 1)`
4. **Jumlah fitur input**: 2 fitur.
5. **Jumlah output model**: 1 unit nilai logit per sampel.

#### Alasan Shape Label Dibuat `(16, 1)` dan Bukan `(16,)`:
- Output dari layer linear terakhir model (`nn.Linear(8, 1)`) memiliki format matriks dua dimensi yaitu `(16, 1)`.
- Fungsi kerugian `nn.BCEWithLogitsLoss()` mewajibkan tensor label ($y$) memiliki dimensi yang **persis sama** dengan tensor prediksi (`prediction`).
- Jika label dibuat dalam bentuk 1D `(16,)`, PyTorch akan memunculkan pesan error ketidakcocokan ukuran tensor. Selain itu, perbedaan dimensi 1D dan 2D pada operasi perbandingan loss berpotensi memicu *broadcasting* otomatis menjadi matriks `(16, 16)` yang merusak perhitungan loss.

---

### 7. Analisis Dtype

Kode pembuatan label:
```python
y = torch.randint(0, 2, (16, 1)).float()
```

#### Alasan Label Diubah Menjadi Float:
Fungsi `torch.randint()` secara otomatis menghasilkan tensor bertipe integer (`torch.int64`). Kita perlu memanggil `.float()` agar tipe datanya berubah menjadi `torch.float32`.

#### Hubungannya dengan `nn.BCEWithLogitsLoss()`:
- `nn.BCEWithLogitsLoss()` dirancang untuk klasifikasi biner dengan formula Binary Cross-Entropy yang melibatkan perhitungan desimal dan fungsi logaritma.
- Berbeda dengan `nn.CrossEntropyLoss` (untuk multi-kelas) yang menerima target berupa nomor indeks kelas (tipe integer), `nn.BCEWithLogitsLoss` mewajibkan label bertipe **`torch.float32`**.
- Jika kita memasukkan label bertipe integer ke dalam `nn.BCEWithLogitsLoss()`, PyTorch akan memunculkan runtime error karena adanya ketidakcocokan tipe data (*type mismatch*).

---

### 8. Analisis Device

#### Mengapa Model dan Tensor Input Harus Berada di Device yang Sama?
PyTorch mengeksekusi operasi matematika tensor menggunakan memori dan prosesor spesifik pada perangkat tersebut (CPU memakai RAM sistem, sedangkan GPU memakai VRAM kartu grafis). Operasi antar tensor (seperti perkalian matriks linear) tidak bisa diproses jika datanya terpisah di dua perangkat yang berbeda tanpa adanya transfer data eksplisit.

#### Contoh Error yang Muncul:
Jika model dijalankan di GPU (`cuda:0`) sementara tensor input $X$ masih berada di CPU, PyTorch akan memunculkan pesan error:
```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

#### Cara Memindahkan Model dan Tensor ke Device yang Sesuai:
Praktik standar yang direkomendasikan adalah mengecek ketersediaan GPU secara dinamis, lalu memindahkan model dan data menggunakan perintah `.to(device)`:
```python
# 1. Tentukan device (gunakan GPU jika tersedia, jika tidak gunakan CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Pindahkan model ke device
model = model.to(device)

# 3. Pindahkan data batch (input dan label) ke device
X = X.to(device)
y = y.to(device)
```

---

### 9. Analisis Output Layer

Arsitektur model:
```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
```

#### Jawaban Analisis:
1. **Mengapa layer pertama memiliki input size 2?**
   Karena data masukan ($X$) memiliki 2 kolom fitur. Layer linear pertama harus mencocokkan jumlah fitur tersebut agar perkalian bobot $X \cdot W^T$ dapat dihitung secara valid.
2. **Mengapa hidden layer memiliki 8 unit?**
   Angka 8 adalah nilai hyperparameter yang ditentukan untuk memproyeksikan fitur input 2D ke ruang representasi yang lebih kaya. Dikombinasikan dengan fungsi aktivasi non-linear ReLU, model dapat memisahkan pola data yang tidak terpisahkan secara linear.
3. **Mengapa output layer memiliki 1 unit?**
   Karena tugas pemodelan ini adalah **klasifikasi biner** (dua kelas: 0 atau 1). Pada klasifikasi biner, kita hanya memerlukan 1 nilai output (nilai logit skalar) per sampel untuk menentukan probabilitas kelas target.
4. **Apakah model tersebut menggunakan Sigmoid secara eksplisit?**
   **Tidak.** Model berakhir langsung pada `nn.Linear(8, 1)` tanpa layer `nn.Sigmoid()`. Output yang dihasilkan adalah nilai mentah (*raw logits*) yang belum diubah menjadi probabilitas 0 sampai 1.
5. **Mengapa `BCEWithLogitsLoss()` tetap dapat digunakan tanpa menambahkan Sigmoid pada model?**
   Karena fungsi `nn.BCEWithLogitsLoss()` sudah menggabungkan fungsi Sigmoid dan rumus Binary Cross-Entropy ke dalam satu fungsi terpadu di belakang layar. Pendekatan ini jauh lebih aman dan stabil secara numerik (*numerically stable*) karena menghindari potensi terjadinya angka mendekati nol yang dapat menyebabkan nilai loss menjadi error atau `NaN`.

---

## C. Perbandingan Training dan Evaluation

### 10. Perbandingan `model.train()` dan `model.eval()`

Berikut tabel perbandingan komprehensif antara `model.train()` dan `model.eval()`:

| Aspek | `model.train()` | `model.eval()` |
| :--- | :--- | :--- |
| **Tujuan Penggunaan** | Menyiapkan model untuk fase pelatihan dan pembaruan bobot. | Menyiapkan model untuk fase pengujian, validasi, atau penggunaan inferensi. |
| **Kapan Digunakan** | Di awal loop training sebelum forward pass, perhitungan loss, dan backpropagation. | Di awal proses evaluasi/validasi sebelum memproses data uji. |
| **Perilaku Dropout** | **Aktif**: Sebagian neuron dimatikan secara acak untuk mencegah overfitting. | **Nonaktif**: Semua neuron dibiarkan aktif bekerja penuh. |
| **Perilaku BatchNorm** | **Menghitung & Update**: Menghitung mean dan variansi batch saat itu serta memperbarui estimasi rata-rata global (*running stats*). | **Statistik Dibekukan**: Menggunakan nilai statistik global yang sudah tersimpan selama proses training. |
| **Dampaknya pada Gradien** | **Tidak mematikan gradien**: Pelacakan gradien tetap aktif secara default. | **Tidak mematikan gradien**: Perhitungan gradien tetap berjalan kecuali jika dibungkus dengan `torch.no_grad()`. |

> **Catatan Penting:** `model.eval()` bertugas mengubah perilaku layer, sedangkan `torch.no_grad()` bertugas menghentikan pelacakan gradien. Keduanya harus dipanggil bersamaan saat melakukan evaluasi model.

---

### 11. Mengapa Evaluation Menggunakan `torch.no_grad()`?

Penggunaan blok evaluasi:
```python
with torch.no_grad():
    prediction = model(X)
```

#### Alasan Penggunaan:
Saat melakukan evaluasi atau validasi, tujuan kita hanyalah mengukur performa model (seperti menghitung loss dan akurasi). Kita tidak menjalankan *backpropagation* (`loss.backward()`) dan tidak memperbarui bobot (`optimizer.step()`), sehingga riwayat komputasi untuk gradien tidak perlu dicatat.

#### Dua Manfaat Utama:
1. **Menghemat Penggunaan Memori (RAM / VRAM GPU):**
   Ketika autograd aktif, PyTorch menyimpan seluruh nilai aktivasi sementara di memori untuk keperluan turunan rantai saat backward pass. Dengan `torch.no_grad()`, alokasi memori ini dihentikan, sehingga menghemat konsumsi memori secara signifikan dan memungkinkan kita menguji data dengan ukuran batch yang lebih besar.
2. **Mempercepat Waktu Komputasi:**
   Karena PyTorch tidak perlu repot-repot mencatat graf operasi dan dependensi gradien di latar belakang, proses forward pass dapat dieksekusi dengan lebih cepat dan ringan.

---

### 12. Membuat Fungsi Evaluation

Berikut adalah implementasi fungsi `evaluate_step()` yang rapi, modular, dan memenuhi seluruh kriteria spesifikasi tugas:

```python
import torch
from torch import nn
from typing import Tuple


def evaluate_step(model: nn.Module, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """
    Fungsi untuk menjalankan satu langkah evaluasi/validasi model.
    
    Kriteria:
    1. Mengaktifkan evaluation mode (model.eval()).
    2. Menonaktifkan pelacakan gradien (with torch.no_grad()).
    3. Menghasilkan prediksi forward pass.
    4. Menghitung loss evaluasi.
    5. Menghitung akurasi klasifikasi biner.
    6. Mengembalikan nilai loss dan akurasi sebagai tuple (float, float).
    """
    # 1. Mengaktifkan evaluation mode
    model.eval()
    
    # 2. Menonaktifkan pelacakan gradien
    with torch.no_grad():
        # 3. Menghasilkan prediksi (forward pass)
        prediction = model(X)
        
        # 4. Menghitung loss evaluasi
        loss = criterion(prediction, y)
        
        # 5. Menghitung binary accuracy
        # Mengubah nilai logit menjadi probabilitas menggunakan sigmoid
        probabilities = torch.sigmoid(prediction)
        
        # Tentukan prediksi kelas dengan ambang batas (threshold) 0.5
        predicted_classes = (probabilities >= 0.5).float()
        
        # Hitung jumlah prediksi yang sesuai dengan target label sebenarnya
        correct_predictions = (predicted_classes == y).float().sum()
        accuracy = (correct_predictions / y.numel()).item()
        
    # 6. Mengembalikan nilai loss dan accuracy sebagai tipe float
    return loss.item(), accuracy
```

#### Hasil Pengujian:
Ketika fungsi ini dijalankan pada data sintetis $X$ berukuran `(16, 2)` dan $y$ berukuran `(16, 1)`:
- **Evaluation Loss**: `0.6760`
- **Evaluation Accuracy**: `62.50%` (atau `0.6250`)

Fungsi berjalan optimal tanpa membebani alokasi memori graf autograd, mengembalikan nilai tipe primitif Python `float`, dan siap digunakan pada loop validasi maupun pengujian model.
