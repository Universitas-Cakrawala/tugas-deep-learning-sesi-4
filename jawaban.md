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
Masalah utama pada fungsi di atas ada pada baris **`model.eval()`**. Karena fungsi ini bertujuan untuk melatih model (*training step*), sehingga mode yang seharusnya diaktifkan adalah **`model.train()`**.

Dalam PyTorch, `model.eval()` digunakan
untuk proses evaluasi atau pengujian, bukan untuk melatih model. `model.eval()` tidak menghentikan proses perhitungan gradient, tapi hanya mengubah perilaku beberapa layer tertentu.


#### Penjelasan :
1. **Fungsi `model.eval()`**:
   `model.eval()` berfungsi untuk mengubah mode kerja model ke mode evaluasi/inferensi. Saat perintah ini dipanggil, PyTorch akan menginstruksikan seluruh layer di dalam model untuk menonaktifkan mekanisme yang hanya digunakan saat proses belajar.
   Perubahan ini tidak menghentikan proses perhitungan gradien, tetapi hanya mengubah cara beberapa layer memproses data.
2. **Mode yang Seharusnya Digunakan saat Training**:
   Mode yang wajib digunakan saat fase pelatihan adalah **`model.train()`**.
   Mode ini memberi informasi kepada PyTorch bahwa model sedang berada dalam tahap training sehingga layer tertentu dapat menjalankan mekanisme pembelajaran seperti biasanya.
3. **Layer yang Perilakunya Berubah antara Mode Training dan Evaluation**:
Beberapa layer memiliki respons berbeda tergantung mode yang digunakan, di antaranya:
   - **`nn.Dropout`**:
     - *Saat Training*: Mematikan sebagian neuron secara acak untuk mencegah ketergantungan berlebih antar neuron (*regularisasi*) dan mengurangir resiko overfitting.
     - *Saat Evaluation*: Dinonaktifkan sepenuhnya. Semua neuron dibiarkan aktif agar hasil prediksi stabil dan konsisten.
   - **`nn.BatchNorm` (Batch Normalization)**:
     - *Saat Training*: Menghitung nilai rata-rata (*mean*) dan variansi langsung dari batch data yang sedang masuk, sekaligus mencatat perkiraan rata-rata dan variansi global (*running statistics*) untuk dipakai nanti.
     - *Saat Evaluation*: Membekukan perhitungan statistik batch. Layer menggunakan nilai rata-rata dan variansi global yang sudah terkumpul selama masa training.
4. **Mengapa Masalah ini Memengaruhi Hasil Training**:
Jika model dilatih menggunakan `model.eval()`, proses training masih dapat berjalan karena `backward()` dan `optimizer.step()` tetap bekerja. Namun, hasil pembelajaran dapat menjadi tidak optimal.

Beberapa dampak yang mungkin terjadi:

-   Layer Dropout tidak aktif sehingga fungsi regularisasi hilang dan
    risiko *overfitting* meningkat.
-   BatchNorm tidak memperbarui nilai statistiknya sehingga performa
    model saat digunakan pada data baru dapat menurun.

Dengan kata lain, kesalahan penggunaan mode ini termasuk *silent bug*, yaitu kesalahan yang tidak selalu menghasilkan error tetapi dapat memengaruhi kualitas model.


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
Perubahan dilakukan dengan mengganti `model.eval()` menjadi
`model.train()`.

Dengan menggunakan `model.train()`, model kembali berada pada kondisi yang sesuai untuk proses pembelajaran. Layer seperti Dropout dapat melakukan regularisasi dan BatchNorm dapat memperbarui statistik berdasarkan data training yang masuk.

Walaupun perubahan kode hanya satu baris, hal tersebut berpengaruh terhadap bagaimana model belajar, terutama pada arsitektur yang menggunakan layer seperti Dropout dan BatchNorm.

---

### 3. Apakah `model.eval()` Menghentikan Gradient?

#### Jawaban:
> **TIDAK.** `model.eval()` sama sekali tidak menghentikan perhitungan gradien.

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
`model.eval()` hanya mengubah perilaku beberapa layer dalam model. Fungsi ini tidak berhubungan langsung dengan penghentian gradient.

Jika tujuan utama adalah menghemat memori dan menghentikan pencatatan gradien ketika evaluasi, maka proses forward harus dilakukan menggunakan:

``` python
with torch.no_grad(): 
    prediction = model(X) 
```

Biasanya, saat evaluasi model, `model.eval()` dan `torch.no_grad()` digunakan secara bersamaan karena keduanya memiliki fungsi yang berbeda.

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
Urutan tersebut tidak dapat dilakukan secara sembarangan karena setiap
tahap bergantung pada tahap sebelumnya.

1.  Pertama, gradien lama harus dihapus menggunakan `zero_grad()` agar tidak terjadi penumpukan nilai gradien dari iterasi sebelumnya.
2.  Setelah itu, model menerima input dan menghasilkan prediksi melalui proses *forward pass*.
3.  Nilai prediksi kemudian dibandingkan dengan label asli untuk menghitung nilai loss.
4.  Setelah nilai loss diperoleh, PyTorch dapat menghitung gradien melalui `loss.backward()`.
5.  Terakhir, optimizer menggunakan gradien tersebut untuk memperbarui parameter model melalui `optimizer.step()`.

Jika salah satu tahap dilewati atau urutannya tidak tepat, proses pembelajaran model dapat terganggu.

---

### 5. Risiko Melupakan `optimizer.zero_grad()`

#### Apa yang Terjadi:
Secara bawaan di PyTorch, eksekusi `loss.backward()` memiliki sifat **menjumlahkan (mengakumulasi)** nilai gradien baru ke variabel gradien yang sudah ada pada memori (`param.grad += grad_baru`).

Artinya, jika `optimizer.zero_grad()` tidak dijalankan, nilai gradien dari batch sebelumnya akan tetap tersimpan dan terus terakumulasi pada iterasi berikutnya.

#### Dampaknya terhadap Proses Training dan Bobot Model:
Jika `optimizer.zero_grad()` tidak dipanggil:
1. **Gradien Terus Menumpuk**: Gradien dari batch sebelumnya tidak dibuang, melainkan terus bertambah seiring berjalannya setiap iterasi, sehingga nilainya
semakin besar dan tidak lagi merepresentasikan kondisi batch saat ini.
2. **Lonjakan Nilai Gradien Menjadi Tidak Stabil (*Exploding Gradients*)**: Nilai gradien akan membesar secara tidak wajar, menyebabkan optimizer memperbarui bobot secara terlalu agresif dan melompat jauh dari titik optimal.
3. **Model Gagal Belajar (Divergen)**: Akumulasi gradien dapat membuat nilai loss menjadi tidak stabil. Dalam
kondisi tertentu, loss dapat meningkat atau menghasilkan nilai `NaN`, sehingga model gagal mencapai konvergensi.

Oleh karena itu, pemanggilan `optimizer.zero_grad()` sebelum setiap proses *backward pass* merupakan langkah penting dalam training PyTorch

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
Output dari layer terakhir model menghasilkan tensor dengan ukuran `(16,1)`. Oleh karena itu, label harus memiliki bentuk yang sama agar dapat dibandingkan secara langsung saat menghitung loss.

Selain itu, fungsi:

``` python
nn.BCEWithLogitsLoss() 
```

mengharuskan ukuran prediksi dan target memiliki dimensi yang sama.

Jika label dibuat dalam bentuk `(16,)`, PyTorch dapat mengalami masalah saat melakukan perhitungan karena dimensi tensor tidak sesuai.

---

### 7. Analisis Dtype

Kode pembuatan label:
```python
y = torch.randint(0, 2, (16, 1)).float()
```

#### Alasan Label Diubah Menjadi Float:
Fungsi `torch.randint()` menghasilkan data bertipe integer (`torch.int64`). Namun, pada kasus klasifikasi biner menggunakan `BCEWithLogitsLoss()`, label harus menggunakan tipe data `float32`.

Perintah:

``` python
.float() 
```

digunakan untuk mengubah tipe data tersebut menjadi `float32`.

### Hubungannya dengan `BCEWithLogitsLoss()`

`BCEWithLogitsLoss()` menggunakan perhitungan Binary Cross Entropy yang melibatkan operasi desimal dan logaritma.

Berbeda dengan `CrossEntropyLoss()` yang menggunakan label berupa indeks kelas integer, `BCEWithLogitsLoss()` membutuhkan nilai target berupa angka float, biasanya:

-   kelas 0 → `0.0`
-   kelas 1 → `1.0`

Jika label masih berupa integer, PyTorch dapat menghasilkan error karena tipe data tidak sesuai.

---

### 8. Analisis Device

#### Mengapa Model dan Tensor Input Harus Berada di Device yang Sama?
PyTorch menjalankan operasi tensor menggunakan perangkat tertentu, seperti CPU atau GPU.

CPU menggunakan memori RAM, sedangkan GPU menggunakan VRAM. Karena keduanya memiliki lokasi penyimpanan berbeda, operasi matematika antar tensor tidak dapat dilakukan jika berada pada device yang berbeda.

Sebagai contoh:

-   Model berada di GPU (`cuda:0`)
-   Input `X` masih berada di CPU

Maka PyTorch akan menghasilkan error:

``` text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! 
```

---

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
   Jumlah 8 merupakan nilai *hyperparameter* yang ditentukan saat membuat model.
   Delapan neuron memberikan ruang bagi model untuk mempelajari pola yang lebih kompleks dari data input.
   Dengan tambahan fungsi aktivasi ReLU, model dapat mempelajari hubungan yang tidak hanya bersifat linear.
3. **Mengapa output layer memiliki 1 unit?**
   Karena model digunakan untuk  **klasifikasi biner** yaitu untuk membedakan dua kelas (dua kelas: 0 atau 1). Karena hanya ada dua kemungkinan kelas (0 dan 1), maka satu nilai output (*logit*) sudah cukup untuk menentukan prediksi.
4. **Apakah model tersebut menggunakan Sigmoid secara eksplisit?**
   **Tidak.** 
   Model berhenti pada:
   ``` python
   nn.Linear(8,1) 
   ```
   
   tanpa layer `nn.Sigmoid()`. 
   
   
   Sehingga. output yang dihasilkan adalah nilai mentah (*raw logits*) yang belum diubah menjadi probabilitas 0 sampai 1 atau raw probabilitas

5. **Mengapa `BCEWithLogitsLoss()` tetap dapat digunakan tanpa menambahkan Sigmoid pada model?**
   Karena fungsi `nn.BCEWithLogitsLoss()` sudah menggabungkan fungsi Sigmoid dan rumus Binary Cross-Entropy ke dalam satu fungsi terpadu di belakang layar.
   Pendekatan ini jauh lebih aman dan stabil secara numerik (*numerically stable*) karena menghindari potensi terjadinya angka mendekati nol yang dapat menyebabkan nilai loss menjadi error atau `NaN`.

---

## C. Perbandingan Training dan Evaluation

### 10. Perbandingan `model.train()` dan `model.eval()`

Dalam PyTorch, `model.train()` dan `model.eval()` digunakan untuk menentukan mode kerja model. Keduanya tidak mengubah struktur model, tetapi mengatur perilaku beberapa layer tertentu selama proses training maupun evaluasi.

Berikut tabel perbandingan komprehensif antara `model.train()` dan `model.eval()`:

| Aspek | `model.train()` | `model.eval()` |
| :--- | :--- | :--- |
| **Tujuan Penggunaan** | Menyiapkan model untuk fase pelatihan dan pembaruan bobot. | Menyiapkan model untuk fase pengujian, validasi, atau penggunaan inferensi. |
| **Kapan Digunakan** | Di awal loop training sebelum forward pass, perhitungan loss, dan backpropagation. | Di awal proses evaluasi/validasi sebelum memproses data uji. |
| **Perilaku Dropout** | **Aktif**: Sebagian neuron dimatikan secara acak untuk mencegah overfitting. | **Nonaktif**: Semua neuron dibiarkan aktif bekerja penuh. |
| **Perilaku BatchNorm** | **Menghitung & Update**: Menghitung mean dan variansi batch saat itu serta memperbarui estimasi rata-rata global (*running stats*). | **Statistik Dibekukan**: Menggunakan nilai statistik global yang sudah tersimpan selama proses training. |
| **Dampaknya pada Gradien** | **Tidak mematikan gradien**: Pelacakan gradien tetap aktif secara default. | **Tidak mematikan gradien**: Perhitungan gradien tetap berjalan kecuali jika dibungkus dengan `torch.no_grad()`. |

> **Catatan Penting:** 

`model.train()` dan `model.eval()` memiliki fungsi yang berbeda dengan
`torch.no_grad()`.

-   `model.train()` dan `model.eval()` mengatur perilaku layer dalam model.
-   `torch.no_grad()` mengatur apakah PyTorch perlu mencatat operasi untuk perhitungan gradient atau tidak.

Oleh karena itu, ketika melakukan evaluasi model, biasanya kedua perintah berikut digunakan secara bersamaan:

``` python
model.eval() 
 
with torch.no_grad(): 
    prediction = model(X) 
```

Tujuannya agar model berada dalam mode evaluasi sekaligus menghemat penggunaan memori.

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
   Karena PyTorch tidak perlu membuat dan menyimpan computation graph, proses forward pass dapat berjalan lebih ringan.
   
   Akibatnya, evaluasi model dapat dilakukan dengan waktu komputasi yang lebih cepat.

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

Fungsi berjalan optimal karena  tidak membuat graph gradient baru, tidak melakukan pembaruan bobot, dan menghasilkan nilai loss dan accuracy dalam format float.


---

## D. Eksperimen dan Interpretasi

### 13. Membandingkan Fungsi Broken dan Corrected

Karena `broken_training_step()` pada script sudah diperbaiki (isinya `model.train()`) dan bug aslinya dipindahkan ke `original_broken_training_step()`, perbandingan berikut menjalankan **kedua versi secara terpisah** — model dan optimizer baru untuk masing-masing, dengan seed dan inisialisasi bobot yang identik — supaya perbandingannya adil.

**Hasil (fair comparison — model & optimizer terpisah, seed sama):**

| | `original_broken_training_step` (pakai `model.eval()`) | `corrected_training_step` (pakai `model.train()`) |
|---|---|---|
| Mode sebelum dipanggil | `training=True` | `training=True` |
| Mode di akhir fungsi | `training=False` (macet di eval, tidak pernah dikembalikan ke train) | `training=True` |
| Loss | `0.733285` | `0.733285` |
| Parameter berubah? | **Ya** | **Ya** |

**Hasil (perbandingan sekuensial pada model & optimizer yang SAMA — cara yang tidak fair):**

| Panggilan | Loss |
|---|---|
| 1 — broken | `0.733285` |
| 2 — corrected | `0.731702` |

Selisih pada perbandingan sekuensial ini (`-0.001584`) **bukan efek dari bug eval/train**, melainkan murni karena bobot model sudah berubah oleh `optimizer.step()` pada panggilan pertama. Ini persis jebakan yang diingatkan di soal: membandingkan loss dua fungsi yang dijalankan berurutan pada model yang sama itu tidak fair, karena kondisi awalnya sudah berbeda.

**Catatan:**

- **Apakah parameter model berubah?** Ya, di **kedua** versi — baik broken maupun corrected. Ini konsisten dengan jawaban soal 3: `model.eval()` tidak menghentikan gradient tracking, jadi `loss.backward()` dan `optimizer.step()` tetap jalan dan tetap mengubah bobot, walau modelnya "salah mode".
- **Apakah hasil ini cukup untuk menyimpulkan bahwa fungsi corrected selalu lebih baik?** Tidak. Pada perbandingan yang fair, loss kedua fungsi ini **identik** (`0.733285` vs `0.733285`). Alasannya: arsitektur model di latihan ini (`nn.Linear` → `ReLU` → `nn.Linear`) tidak memiliki layer `Dropout` atau `BatchNorm`, yaitu satu-satunya jenis layer yang perilakunya benar-benar berubah antara mode train dan eval. Karena itu, untuk model spesifik ini, bug `model.eval()` saat training **tidak menghasilkan perbedaan angka yang terlihat** sama sekali.
- **Keterbatasan perbandingan ini:**
  1. Satu kali forward-backward pass pada satu batch data acak bukan sampel yang representatif, hasilnya bisa kebetulan sama karena inisialisasi dan data yang identik, bukan karena bug-nya tidak berdampak secara umum.
  2. Kesimpulan "tidak ada bedanya" ini **hanya berlaku untuk arsitektur model ini**. Begitu model punya `Dropout` atau `BatchNorm` (yang sangat umum di model Deep Learning nyata), efek bug `model.eval()` saat training akan terlihat jelas — dropout tidak aktif (hilang efek regularisasi) dan running statistics BatchNorm tidak pernah ter-update (rusak saat dipakai inferensi nanti).
  3. Membandingkan loss dari satu langkah training saja tidak bisa menyimpulkan performa jangka panjang (banyak epoch, konvergensi, generalisasi ke data baru) — itu perlu eksperimen training penuh dengan data validasi terpisah.

### 14. Mengapa Model Kecil Dapat Tetap Memiliki Masalah?

Ukuran model yang kecil tidak menjamin bahwa model tersebut bebas dari bug.
Bahkan, model kecil terkadang lebih sulit ditemukan kesalahannya karena program tetap berjalan tanpa menghasilkan error. Beberapa masalah yang tetap bisa terjadi meski model sangat kecil:

1. **Mode training/evaluation tertukar (silent bug)** — seperti dibuktikan di soal 13: kalau model kebetulan tidak punya Dropout/BatchNorm, bug ini tidak akan terlihat dari angka loss sama sekali. Begitu model diperbesar dan ditambah Dropout/BatchNorm (hal yang sangat umum), bug yang sama bisa merusak hasil training secara signifikan.
2. **Shape input tidak sesuai** — kalau `X` punya 3 fitur tapi `nn.Linear(2, 8)` mengharapkan 2, PyTorch akan melempar `RuntimeError` soal ukuran matriks — ini gampang ketahuan karena error langsung muncul, tapi tetap sering terjadi kalau lupa cek shape setelah preprocessing data berubah.
3. **Dtype label salah** — kalau `y` dibiarkan `int64` (bukan `.float()`), `BCEWithLogitsLoss()` akan menolak dengan error tipe data. Sekali lagi ini "untung" ketahuan lewat error, bukan lewat logika.
4. **Gradient tidak dihapus (`zero_grad()` lupa dipanggil)** — ini juga silent bug: kode tetap jalan tanpa error, tapi gradient terakumulasi dari batch-batch sebelumnya dan training jadi tidak stabil pelan-pelan, bukan langsung crash.
5. **Learning rate tidak sesuai** — model sekecil apapun bisa gagal belajar (loss stuck atau meledak) kalau learning rate terlalu besar atau kecil; ini murni masalah hyperparameter, tidak ada hubungannya dengan ukuran model.

Kesimpulannya: ukuran model menentukan seberapa *mahal* bug-nya secara komputasi, bukan seberapa *mungkin* bug itu terjadi. Bug logika (silent bug) sama berbahayanya di model kecil maupun besar — bahkan bisa dibilang lebih berbahaya di model kecil karena orang cenderung lebih santai memeriksanya.

### 15. Membuat Checklist Debugging

Checklist pemeriksaan sebelum menjalankan training panjang (di luar 4 contoh yang sudah diberikan di soal):

```
[ ] Shape input sudah diperiksa (X.shape cocok dengan in_features layer pertama)
[ ] Shape label sudah diperiksa (sama persis dengan shape output model, bukan cuma "mirip")
[ ] Dtype input dan label sudah sesuai (float32 untuk BCEWithLogitsLoss, long untuk CrossEntropyLoss)
[ ] Model dan data berada pada device yang sama (cek lewat inspect_batch sebelum training)
[ ] model.train() dipanggil di awal setiap training step, model.eval() di awal setiap evaluation step
[ ] optimizer.zero_grad() dipanggil sebelum setiap loss.backward() pada tiap iterasi
[ ] Forward pass saat evaluasi/validasi dibungkus with torch.no_grad()
[ ] Loss function cocok dengan bentuk output model (mis. tidak menambahkan Sigmoid manual kalau sudah pakai BCEWithLogitsLoss)
[ ] Loss dicatat/dilog pakai .item() (bukan tensor mentah) supaya tidak menahan computation graph di memori
[ ] Learning rate dan jumlah epoch masuk akal (loss dicek beberapa iterasi awal, tidak langsung NaN atau diam saja)
[ ] Random seed di-set (torch.manual_seed) supaya eksperimen bisa direproduksi dan dibandingkan secara adil
[ ] Parameter model benar-benar berubah setelah training step (cek dengan membandingkan state_dict sebelum/sesudah)
```

---

## E. Pertanyaan Reflektif

### 16. Penjelasan Teknis

Kode yang tidak menghasilkan error belum tentu kode yang benar, karena Python/PyTorch hanya memeriksa apakah operasi-operasinya **valid secara sintaks dan bentuk data** (tensor bisa dikalikan, dimensi cocok, tipe data sesuai) — bukan apakah **logikanya sesuai maksud programmer**. Selama operasi matematikanya sah dijalankan, PyTorch akan tetap mengeksekusinya walau hasilnya secara konsep salah.

Contoh nyata dari latihan ini: `broken_training_step()` memanggil `model.eval()` padahal sedang melakukan training. Kode ini berjalan sempurna tanpa satupun error — bahkan `loss.backward()` dan `optimizer.step()` tetap bekerja dan bobot model tetap ter-update. Lebih jauh lagi, dari eksperimen soal 13, angka loss yang dihasilkan bahkan **identik** dengan versi yang sudah benar (karena model ini tidak punya Dropout/BatchNorm). Jadi kalau hanya mengandalkan "kode jalan tanpa error" atau "angkanya kelihatan wajar" sebagai tolok ukur kebenaran, bug ini akan lolos sepenuhnya — padahal secara desain, kode tersebut salah dan berpotensi merusak model lain yang strukturnya sedikit berbeda (yang punya Dropout/BatchNorm).

### 17. Peran Framework

PyTorch memang mengotomatiskan bagian yang sifatnya mekanis: menghitung turunan (autograd), mengalokasikan/membebaskan memori komputasi, dan menyediakan optimizer siap pakai. Tapi framework tidak tahu apa yang *seharusnya* terjadi secara konseptual — itu tetap tanggung jawab programmer/mahasiswa. Beberapa alasan konkretnya:

- **Forward pass & loss function** — framework hanya menjalankan operasi yang diberikan; kalau arsitektur atau loss function-nya tidak cocok dengan masalahnya (misal loss klasifikasi dipakai untuk regresi), tidak ada peringatan otomatis.
- **Backward pass** — autograd menghitung gradient dengan benar secara matematis, tapi kalau urutan operasinya salah (misal lupa `zero_grad()`), autograd tetap "patuh" menghitung gradient yang salah secara konsep (terakumulasi).
- **Optimizer update** — optimizer akan selalu mengupdate parameter yang punya gradient, tanpa peduli apakah update itu masuk akal (contoh: kalau `model.eval()` lupa dikembalikan ke `model.train()`, optimizer tetap jalan seperti biasa).
- **Training vs evaluation mode** — ini murni instruksi manual dari programmer (`model.train()` / `model.eval()`); PyTorch tidak bisa menebak kapan seharusnya dipanggil.
- **Shape & dtype** — PyTorch hanya memvalidasi kompatibilitas teknis, bukan apakah datanya secara semantik benar (misal label 0/1 yang salah urutan kelasnya tetap dianggap valid selama shape & dtype cocok).

Jadi automatic differentiation membebaskan mahasiswa dari menurunkan gradient secara manual, tapi tidak membebaskan dari memahami *apa* yang sedang dihitung dan *kenapa* — karena di situlah letak bug yang paling berbahaya (silent bug), seperti yang dibuktikan langsung di latihan ini.

---

## 18. Kesimpulan

Berdasarkan latihan yang telah dilakukan, dapat disimpulkan bahwa kesalahan pada fungsi `broken_training_step()` bukan berasal dari error program, melainkan dari penggunaan mode model yang tidak sesuai.

Penggunaan:

``` python
model.eval() 
```

pada proses training merupakan kesalahan karena mode tersebut dirancang untuk evaluasi, bukan pembelajaran.

Perbaikannya cukup sederhana, yaitu mengganti menjadi:

``` python
model.train() 
```

agar model kembali bekerja dalam kondisi training.

Selain itu, latihan ini juga menunjukkan bahwa `model.eval()` tidak sama dengan menghentikan gradient. Fungsi tersebut hanya mengubah beberapa layer tertentu seperti Dropout dan BatchNorm yang mengontrol level berbeda (autograd engine, bukan perilaku layer). Karena itu keduanya sering dipakai bersamaan saat evaluasi, tapi untuk tujuan yang berbeda.

Untuk menghentikan pencatatan gradient dan menghemat memori saat evaluasi, PyTorch menyediakan:

``` python
torch.no_grad() 
```

yang bekerja pada sistem autograd.

Latihan ini juga memperlihatkan pentingnya melakukan pengecekan terhadap beberapa komponen sebelum training, seperti:

-   bentuk tensor (*shape*);
-   tipe data (*dtype*);
-   device model dan data;
-   kesesuaian loss function;
-   mode training dan evaluasi.

berbeda dengan bug mode training/evaluation yang bisa lolos tanpa error maupun perubahan angka yang mencolok, terutama pada model kecil tanpa Dropout/BatchNorm seperti pada latihan ini. 

Pelajaran terpenting yang kami dapat: kode yang berjalan lancar dan menghasilkan angka yang masuk akal belum tentu benar secara konsep dan verifikasi harus dilakukan secara sadar terhadap logikanya, bukan hanya terhadap hasil eksekusinya.
