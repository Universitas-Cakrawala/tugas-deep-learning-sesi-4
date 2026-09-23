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
Masalah utamanya ada pada baris **`model.eval()`**. Fungsi ini dipakai untuk *training step*, jadi mode yang seharusnya digunakan adalah **`model.train()`**.

Dalam PyTorch, `model.eval()` dipakai saat evaluasi atau pengujian, bukan saat model sedang dilatih. Yang perlu diperhatikan, `model.eval()` juga tidak menghentikan perhitungan gradient. Fungsi ini hanya mengubah perilaku beberapa layer tertentu.


#### Penjelasan :
1. **Fungsi `model.eval()`**:
   `model.eval()` mengubah mode kerja model menjadi mode evaluasi atau inferensi. Saat perintah ini dipanggil, beberapa layer akan mengubah perilakunya dari mode training ke mode evaluasi.
   Perubahan ini tidak menghentikan perhitungan gradien. Yang berubah adalah cara beberapa layer memproses data.
2. **Mode yang Seharusnya Digunakan saat Training**:
   Mode yang wajib digunakan saat fase pelatihan adalah **`model.train()`**.
   Mode ini memberi tahu PyTorch bahwa model sedang berada pada tahap training. Dengan begitu, layer tertentu dapat menjalankan perilakunya seperti saat proses pembelajaran.
3. **Layer yang Perilakunya Berubah antara Mode Training dan Evaluation**:
Beberapa layer memiliki respons berbeda tergantung mode yang digunakan, di antaranya:
   - **`nn.Dropout`**:
     - *Saat Training*: Mematikan sebagian neuron secara acak untuk mencegah ketergantungan berlebih antar neuron (*regularisasi*) dan mengurangir resiko overfitting.
     - *Saat Evaluation*: Dinonaktifkan sepenuhnya. Semua neuron dibiarkan aktif agar hasil prediksi stabil dan konsisten.
   - **`nn.BatchNorm` (Batch Normalization)**:
     - *Saat Training*: Menghitung nilai rata-rata (*mean*) dan variansi langsung dari batch data yang sedang masuk, sekaligus mencatat perkiraan rata-rata dan variansi global (*running statistics*) untuk dipakai nanti.
     - *Saat Evaluation*: Membekukan perhitungan statistik batch. Layer menggunakan nilai rata-rata dan variansi global yang sudah terkumpul selama masa training.
4. **Mengapa Masalah ini Memengaruhi Hasil Training**:
Jika model dilatih menggunakan `model.eval()`, proses training sebenarnya masih bisa berjalan. `backward()` dan `optimizer.step()` tetap bekerja. Masalahnya, perilaku beberapa layer sudah berubah ke mode evaluasi, sehingga hasil pembelajarannya bisa tidak sesuai yang diharapkan.

Beberapa dampak yang mungkin terjadi:

-   Layer Dropout tidak aktif sehingga fungsi regularisasi hilang dan
    risiko *overfitting* meningkat.
-   BatchNorm tidak memperbarui nilai statistiknya sehingga performa
    model saat digunakan pada data baru dapat menurun.

Jadi, kesalahan seperti ini termasuk *silent bug*. Program tetap bisa berjalan tanpa error, tetapi perilaku dan hasil model dapat ikut terpengaruh.


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
Perubahannya sebenarnya sederhana: `model.eval()` diganti menjadi `model.train()`.

Dengan `model.train()`, model kembali berada pada mode yang sesuai untuk pembelajaran. Dropout dapat menjalankan regularisasi, sedangkan BatchNorm dapat memperbarui statistik berdasarkan data training yang masuk.

Walaupun hanya satu baris yang berubah, efeknya bisa cukup penting, terutama jika arsitektur model menggunakan Dropout atau BatchNorm.

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
`model.eval()` hanya mengubah perilaku beberapa layer. Fungsi ini tidak digunakan untuk menghentikan perhitungan gradient.

Kalau tujuannya adalah menghemat memori sekaligus menghentikan pencatatan gradien saat evaluasi, proses forward dapat dilakukan menggunakan:

``` python
with torch.no_grad(): 
    prediction = model(X) 
```

Biasanya kedua perintah ini digunakan bersama saat evaluasi karena fungsinya memang berbeda.

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
Urutan tersebut tidak bisa dibalik begitu saja karena setiap tahap bergantung pada hasil dari tahap sebelumnya.

1.  Pertama, gradien lama harus dihapus menggunakan `zero_grad()` agar tidak terjadi penumpukan nilai gradien dari iterasi sebelumnya.
2.  Setelah itu, model menerima input dan menghasilkan prediksi melalui proses *forward pass*.
3.  Nilai prediksi kemudian dibandingkan dengan label asli untuk menghitung nilai loss.
4.  Setelah nilai loss diperoleh, PyTorch dapat menghitung gradien melalui `loss.backward()`.
5.  Terakhir, optimizer menggunakan gradien tersebut untuk memperbarui parameter model melalui `optimizer.step()`.

Kalau salah satu tahap dilewati atau urutannya tidak tepat, proses pembelajaran model bisa terganggu.

---

### 5. Risiko Melupakan `optimizer.zero_grad()`

#### Apa yang Terjadi:
Secara bawaan di PyTorch, eksekusi `loss.backward()` memiliki sifat **menjumlahkan (mengakumulasi)** nilai gradien baru ke variabel gradien yang sudah ada pada memori (`param.grad += grad_baru`).

Artinya, kalau `optimizer.zero_grad()` tidak dijalankan, gradien dari batch sebelumnya masih tersimpan dan akan terus ditambahkan pada iterasi berikutnya.

#### Dampaknya terhadap Proses Training dan Bobot Model:
Jika `optimizer.zero_grad()` tidak dipanggil:
1. **Gradien Terus Menumpuk**: Gradien dari batch sebelumnya tidak dibuang, melainkan terus bertambah seiring berjalannya setiap iterasi, sehingga nilainya
semakin besar dan tidak lagi merepresentasikan kondisi batch saat ini.
2. **Lonjakan Nilai Gradien Menjadi Tidak Stabil (*Exploding Gradients*)**: Nilai gradien akan membesar secara tidak wajar, menyebabkan optimizer memperbarui bobot secara terlalu agresif dan melompat jauh dari titik optimal.
3. **Model Gagal Belajar (Divergen)**: Akumulasi gradien dapat membuat nilai loss menjadi tidak stabil. Dalam
kondisi tertentu, loss dapat meningkat atau menghasilkan nilai `NaN`, sehingga model gagal mencapai konvergensi.

Karena itu, `optimizer.zero_grad()` perlu dipanggil sebelum proses *backward pass* pada setiap iterasi training PyTorch.

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
Layer terakhir model menghasilkan tensor berukuran `(16,1)`. Karena itu, label dibuat dengan bentuk yang sama agar keduanya bisa langsung dibandingkan saat menghitung loss.

Selain itu, fungsi berikut:

``` python
nn.BCEWithLogitsLoss() 
```

mengharuskan ukuran prediksi dan target memiliki dimensi yang sama.

Kalau label dibuat dalam bentuk `(16,)`, bentuk tensor menjadi berbeda dari output model dan perhitungan loss dapat bermasalah.

---

### 7. Analisis Dtype

Kode pembuatan label:
```python
y = torch.randint(0, 2, (16, 1)).float()
```

#### Alasan Label Diubah Menjadi Float:
`torch.randint()` menghasilkan data bertipe integer (`torch.int64`). Sementara itu, pada klasifikasi biner dengan `BCEWithLogitsLoss()`, target yang digunakan berupa `float32`.

Perintah ini:

``` python
.float() 
```

mengubah tipe data tersebut menjadi `float32`.

### Hubungannya dengan `BCEWithLogitsLoss()`

`BCEWithLogitsLoss()` menggunakan perhitungan Binary Cross Entropy yang melibatkan operasi desimal dan logaritma.

Berbeda dengan `CrossEntropyLoss()` yang menggunakan label berupa indeks kelas integer, `BCEWithLogitsLoss()` membutuhkan nilai target berupa angka float, biasanya:

-   kelas 0 → `0.0`
-   kelas 1 → `1.0`

Kalau label masih berupa integer, tipe datanya tidak sesuai dengan yang dibutuhkan oleh fungsi tersebut dan PyTorch dapat menghasilkan error.

---

### 8. Analisis Device

#### Mengapa Model dan Tensor Input Harus Berada di Device yang Sama?
PyTorch menjalankan operasi tensor pada device tertentu, misalnya CPU atau GPU.

CPU menggunakan RAM, sedangkan GPU menggunakan VRAM. Karena berada pada device yang berbeda, tensor yang digunakan dalam satu operasi perlu ditempatkan pada device yang sama.

Contohnya:

-   Model berada di GPU (`cuda:0`)
-   Input `X` masih berada di CPU

Maka PyTorch akan menghasilkan error:

``` text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! 
```

---

#### Cara Memindahkan Model dan Tensor ke Device yang Sesuai:
Cara yang umum digunakan adalah mengecek apakah GPU tersedia, lalu memindahkan model dan data ke device yang dipilih dengan `.to(device)`:
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
   Data masukan ($X$) memiliki 2 kolom fitur. Karena itu, layer linear pertama juga menerima 2 fitur agar perkalian bobot $X \cdot W^T$ dapat dilakukan dengan benar.
2. **Mengapa hidden layer memiliki 8 unit?**
   Jumlah 8 merupakan nilai *hyperparameter* yang ditentukan saat membuat model.
   Delapan neuron memberikan ruang bagi model untuk mempelajari pola yang lebih kompleks dari data input.
   Dengan tambahan fungsi aktivasi ReLU, model dapat mempelajari hubungan yang tidak hanya bersifat linear.
3. **Mengapa output layer memiliki 1 unit?**
   Model ini digunakan untuk **klasifikasi biner**, yaitu membedakan kelas 0 dan 1. Jadi, satu nilai output berupa *logit* sudah cukup untuk menentukan kelas prediksi.
4. **Apakah model tersebut menggunakan Sigmoid secara eksplisit?**
   **Tidak.** 
   Model berhenti pada:
   ``` python
   nn.Linear(8,1) 
   ```
   
   tanpa layer `nn.Sigmoid()`. 
   
   
   Jadi, output yang dihasilkan masih berupa nilai mentah (*raw logits*), belum berupa probabilitas 0 sampai 1.

5. **Mengapa `BCEWithLogitsLoss()` tetap dapat digunakan tanpa menambahkan Sigmoid pada model?**
   Karena fungsi `nn.BCEWithLogitsLoss()` sudah menggabungkan fungsi Sigmoid dan rumus Binary Cross-Entropy ke dalam satu fungsi terpadu di belakang layar.
   Pendekatan ini jauh lebih aman dan stabil secara numerik (*numerically stable*) karena menghindari potensi terjadinya angka mendekati nol yang dapat menyebabkan nilai loss menjadi error atau `NaN`.

---

## C. Perbandingan Training dan Evaluation

### 10. Perbandingan `model.train()` dan `model.eval()`

Dalam PyTorch, `model.train()` dan `model.eval()` menentukan mode kerja model. Keduanya tidak mengubah struktur model, tetapi memengaruhi perilaku beberapa layer saat training dan evaluasi.

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

Pada evaluasi, kita dapat menggunakan blok berikut:
```python
with torch.no_grad():
    prediction = model(X)
```

#### Alasan Penggunaan:
Saat evaluasi atau validasi, kita hanya perlu mengukur performa model, misalnya melalui loss dan akurasi. Kita tidak melakukan *backpropagation* (`loss.backward()`) atau memperbarui bobot dengan `optimizer.step()`. Karena itu, riwayat komputasi untuk gradient tidak perlu disimpan.

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
- **Evaluation Loss**: `0.7301`
- **Evaluation Accuracy**: `56.25%` (atau `0.5625`)

Fungsi berjalan optimal karena tidak membuat graph gradient baru, tidak melakukan pembaruan bobot, dan menghasilkan nilai loss dan accuracy dalam format float.


---

## D. Eksperimen dan Interpretasi

### 13. Membandingkan Fungsi Broken dan Corrected

Karena `broken_training_step()` pada script sudah diperbaiki dan bug aslinya dipindahkan ke `original_broken_training_step()`, kedua versi dibandingkan **secara terpisah**. Masing-masing menggunakan model dan optimizer baru, dengan seed serta inisialisasi bobot yang sama supaya perbandingannya tetap adil.

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

Selisih pada perbandingan sekuensial ini (`-0.001584`) **bukan disebabkan oleh bug eval/train**. Bobot model sudah berubah setelah `optimizer.step()` pada panggilan pertama. Jadi, ketika fungsi kedua dijalankan, kondisi awal modelnya sudah berbeda. Itulah sebabnya membandingkan dua fungsi secara berurutan pada model yang sama tidak fair.

**Catatan:**

- **Apakah parameter model berubah?** Ya, di **kedua** versi — baik broken maupun corrected. Ini konsisten dengan jawaban soal 3: `model.eval()` tidak menghentikan gradient tracking, jadi `loss.backward()` dan `optimizer.step()` tetap jalan dan tetap mengubah bobot, walau modelnya "salah mode".
- **Apakah hasil ini cukup untuk menyimpulkan bahwa fungsi corrected selalu lebih baik?** Tidak. Pada perbandingan yang fair, loss kedua fungsi ini **identik** (`0.733285` vs `0.733285`). Alasannya: arsitektur model di latihan ini (`nn.Linear` → `ReLU` → `nn.Linear`) tidak memiliki layer `Dropout` atau `BatchNorm`, yaitu satu-satunya jenis layer yang perilakunya benar-benar berubah antara mode train dan eval. Karena itu, untuk model spesifik ini, bug `model.eval()` saat training **tidak menghasilkan perbedaan angka yang terlihat** sama sekali.
- **Keterbatasan perbandingan ini:**
  1. Satu kali forward-backward pass pada satu batch data acak bukan sampel yang representatif, hasilnya bisa kebetulan sama karena inisialisasi dan data yang identik, bukan karena bug-nya tidak berdampak secara umum.
  2. Kesimpulan "tidak ada bedanya" ini **hanya berlaku untuk arsitektur model ini**. Begitu model punya `Dropout` atau `BatchNorm` (yang sangat umum di model Deep Learning nyata), efek bug `model.eval()` saat training akan terlihat jelas — dropout tidak aktif (hilang efek regularisasi) dan running statistics BatchNorm tidak pernah ter-update (rusak saat dipakai inferensi nanti).
  3. Membandingkan loss dari satu langkah training saja tidak bisa menyimpulkan performa jangka panjang (banyak epoch, konvergensi, generalisasi ke data baru) — itu perlu eksperimen training penuh dengan data validasi terpisah.

### 14. Mengapa Model Kecil Dapat Tetap Memiliki Masalah?

Model yang kecil bukan berarti bebas dari bug. Bahkan, beberapa bug justru sulit terlihat karena program tetap berjalan tanpa menghasilkan error. Beberapa masalah masih bisa muncul meskipun modelnya sederhana:

1. **Mode training/evaluation tertukar (silent bug)** — seperti dibuktikan di soal 13: kalau model kebetulan tidak punya Dropout/BatchNorm, bug ini tidak akan terlihat dari angka loss sama sekali. Begitu model diperbesar dan ditambah Dropout/BatchNorm (hal yang sangat umum), bug yang sama bisa merusak hasil training secara signifikan.
2. **Shape input tidak sesuai** — kalau `X` punya 3 fitur tapi `nn.Linear(2, 8)` mengharapkan 2, PyTorch akan melempar `RuntimeError` soal ukuran matriks — ini gampang ketahuan karena error langsung muncul, tapi tetap sering terjadi kalau lupa cek shape setelah preprocessing data berubah.
3. **Dtype label salah** — kalau `y` dibiarkan `int64` (bukan `.float()`), `BCEWithLogitsLoss()` akan menolak dengan error tipe data. Sekali lagi ini "untung" ketahuan lewat error, bukan lewat logika.
4. **Gradient tidak dihapus (`zero_grad()` lupa dipanggil)** — ini juga silent bug: kode tetap jalan tanpa error, tapi gradient terakumulasi dari batch-batch sebelumnya dan training jadi tidak stabil pelan-pelan, bukan langsung crash.
5. **Learning rate tidak sesuai** — model sekecil apapun bisa gagal belajar (loss stuck atau meledak) kalau learning rate terlalu besar atau kecil; ini murni masalah hyperparameter, tidak ada hubungannya dengan ukuran model.

Kesimpulannya, ukuran model lebih menentukan seberapa besar dampak bug dari sisi komputasi, bukan apakah bug tersebut bisa terjadi atau tidak. *Silent bug* tetap bisa muncul pada model kecil maupun besar. Pada model kecil, bug seperti ini malah bisa lebih mudah terlewat karena model terlihat sederhana.

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

Kode yang tidak menghasilkan error belum tentu berarti kodenya benar. Python/PyTorch dapat memeriksa apakah operasi yang diberikan **valid secara sintaks dan bentuk data**, tetapi tidak selalu bisa mengetahui apakah **logikanya sesuai dengan maksud programmer**. Selama operasinya valid secara matematis, PyTorch tetap akan menjalankannya meskipun konsep yang digunakan ternyata salah.

Contoh nyata dari latihan ini: `broken_training_step()` memanggil `model.eval()` padahal sedang melakukan training. Kode ini berjalan sempurna tanpa satupun error — bahkan `loss.backward()` dan `optimizer.step()` tetap bekerja dan bobot model tetap ter-update. Lebih jauh lagi, dari eksperimen soal 13, angka loss yang dihasilkan bahkan **identik** dengan versi yang sudah benar (karena model ini tidak punya Dropout/BatchNorm). Jadi kalau hanya mengandalkan "kode jalan tanpa error" atau "angkanya kelihatan wajar" sebagai tolok ukur kebenaran, bug ini akan lolos sepenuhnya — padahal secara desain, kode tersebut salah dan berpotensi merusak model lain yang strukturnya sedikit berbeda (yang punya Dropout/BatchNorm).

### 17. Peran Framework

PyTorch memang mengotomatiskan banyak bagian yang sifatnya mekanis, seperti menghitung turunan melalui autograd, mengelola memori komputasi, dan menyediakan optimizer. Tetapi framework tidak mengetahui apa yang **seharusnya** terjadi secara konseptual. Bagian itu tetap menjadi tanggung jawab programmer atau mahasiswa. Contohnya:

- **Forward pass & loss function** — framework hanya menjalankan operasi yang diberikan; kalau arsitektur atau loss function-nya tidak cocok dengan masalahnya (misal loss klasifikasi dipakai untuk regresi), tidak ada peringatan otomatis.
- **Backward pass** — autograd menghitung gradient dengan benar secara matematis, tapi kalau urutan operasinya salah (misal lupa `zero_grad()`), autograd tetap "patuh" menghitung gradient yang salah secara konsep (terakumulasi).
- **Optimizer update** — optimizer akan selalu mengupdate parameter yang punya gradient, tanpa peduli apakah update itu masuk akal (contoh: kalau `model.eval()` lupa dikembalikan ke `model.train()`, optimizer tetap jalan seperti biasa).
- **Training vs evaluation mode** — ini murni instruksi manual dari programmer (`model.train()` / `model.eval()`); PyTorch tidak bisa menebak kapan seharusnya dipanggil.
- **Shape & dtype** — PyTorch hanya memvalidasi kompatibilitas teknis, bukan apakah datanya secara semantik benar (misal label 0/1 yang salah urutan kelasnya tetap dianggap valid selama shape & dtype cocok).

Jadi, automatic differentiation memang membebaskan mahasiswa dari menghitung gradient secara manual. Tetapi kita tetap perlu memahami *apa* yang sedang dihitung dan *kenapa*. Di situlah *silent bug* seperti pada latihan ini bisa terlewat.

---

## 18. Kesimpulan

Dari latihan ini terlihat bahwa masalah pada `broken_training_step()` bukan berupa error program, tetapi penggunaan mode model yang tidak sesuai.

Penggunaan:

``` python
model.eval() 
```

saat proses training merupakan kesalahan karena mode tersebut digunakan untuk evaluasi, bukan untuk pembelajaran.

Perbaikannya cukup sederhana, yaitu mengganti menjadi:

``` python
model.train() 
```

agar model kembali bekerja dalam mode training.

Latihan ini juga menunjukkan bahwa `model.eval()` bukan berarti menghentikan gradient. Fungsi tersebut mengubah perilaku beberapa layer, seperti Dropout dan BatchNorm, sedangkan `torch.no_grad()` bekerja pada sistem autograd. Karena fungsinya berbeda, keduanya sering digunakan bersamaan saat evaluasi.

Untuk menghentikan pencatatan gradient dan menghemat memori saat evaluasi, PyTorch menyediakan:

``` python
torch.no_grad() 
```

yang bekerja pada sistem autograd dan tidak mengubah mode layer model.

Latihan ini juga memperlihatkan bahwa beberapa komponen perlu diperiksa sebelum training, seperti:

-   bentuk tensor (*shape*);
-   tipe data (*dtype*);
-   device model dan data;
-   kesesuaian loss function;
-   mode training dan evaluasi.

berbeda dengan bug mode training/evaluation yang bisa lolos tanpa error maupun perubahan angka yang mencolok, terutama pada model kecil tanpa Dropout/BatchNorm seperti pada latihan ini. 

Pelajaran terpenting yang kami dapat: kode yang berjalan lancar dan menghasilkan angka yang masuk akal belum tentu benar secara konsep dan verifikasi harus dilakukan secara sadar terhadap logikanya, bukan hanya terhadap hasil eksekusinya.
