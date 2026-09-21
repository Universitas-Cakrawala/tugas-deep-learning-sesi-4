# Deep Technical Analysis: PyTorch Framework Debugging, Operational Modes, and Autograd Mechanics

- **Author / Analyst**: Universitas Cakrawala — Deep Learning Engineering Group
- **Subject**: Framework Debugging Exercise (Session 4 — SDA2110)
- **Target Files**: `DL_Sesi04_Framework_Debugging_Exercise.py`, `DL004.pdf`
- **Focus**: Autograd Computation Graph, Training vs. Evaluation Modes, Layer-Level Stochastic Mechanics, Tensor & Memory Semantics, and Numerical Stability

---

## 1. Executive Summary

Dalam pengembangan arsitektur *Deep Learning* modern menggunakan PyTorch, kesalahan implementasi yang paling berbahaya sering kali bukanlah *syntax error* atau *runtime crash*, melainkan **silent logic bugs** (kesalahan semantik di mana kode tetap berjalan tanpa pesan error, namun merusak konvergensi, estimasi statistik, atau kapasitas generalisasi model).

Berkas `DL_Sesi04_Framework_Debugging_Exercise.py` menyajikan studi kasus klasik mengenai *framework debugging* yang melibatkan:
1. **Mode operasional modul yang keliru**: Penggunaan `model.eval()` dalam fungsi pelatihan (`broken_training_step`).
2. **Dinamika akumulasi gradien dan siklus optimasi**: Urutan eksekusi `zero_grad()`, `forward`, `loss`, `backward()`, dan `step()`.
3. **Penyelarasan bentuk (*shape*) dan tipe data (*dtype*) tensor**: Penyesuaian dimensi target `(N, 1)` dan tipe `float32` terhadap fungsi objektif `nn.BCEWithLogitsLoss()`.
4. **Disiplin isolasi komputasi evaluasi**: Pemisahan tegas antara pengubahan mode layer (`model.eval()`) dengan deaktivasi pelacakan graf autograd (`torch.no_grad()`).

Analisis ini membedah secara fundamental konsep teoretis, implementasi tingkat mesin (*engine-level*), dan bukti eksperimental dari setiap komponen tersebut.

---

## 2. Arsitektur Internal PyTorch: Autograd & Dynamic Computational Graph (DAG)

### 2.1 Pembangunan Graf Komputasi Dinamis
PyTorch menggunakan paradigma **Define-by-Run** (graf komputasi dinamis). Setiap kali operasi matematika dieksekusi pada tensor dengan `requires_grad=True`, PyTorch secara otomatis membangun graf berarah tanpa siklus (*Directed Acyclic Graph* / DAG) di latar belakang.

```
       [ Input X ] (requires_grad=False)
            │
            ▼
    ┌───────────────┐
    │ nn.Linear(2,8)│ ◄── [ W1, b1 ] (requires_grad=True)
    └───────┬───────┘
            │  (AccumulateGrad node)
            ▼
    ┌───────────────┐
    │   nn.ReLU()   │ ◄── ReluBackward0
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ nn.Linear(8,1)│ ◄── [ W2, b2 ] (requires_grad=True)
    └───────┬───────┘
            │  (AccumulateGrad node)
            ▼
       [ Logits ] ◄── AddmmBackward0
            │
            ▼
┌───────────────────────┐
│ nn.BCEWithLogitsLoss()│ ◄── [ Target y ] (requires_grad=False)
└───────────┬───────────┘
            ▼
        [ Loss ] ◄── BinaryCrossEntropyWithLogitsBackward0
```

- Setiap node perantara merepresentasikan operasi matematika dan menyimpan referensi ke fungsi diferensiasinya melalui atribut `.grad_fn`.
- Daun graf (*leaf nodes*) adalah parameter teroptimasi ($\mathbf{W}, \mathbf{b}$) yang memiliki atribut `.grad_fn = None` dan `.requires_grad = True`.

### 2.2 Mekanika Backward Pass dan Akumulasi Gradien
Ketika `loss.backward()` dipanggil:
1. Autograd menjelajahi graf secara terbalik (*reverse-mode automatic differentiation*) dari simpul loss menuju seluruh simpul daun menggunakan aturan rantai (*multivariate chain rule*):
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{W}}$$
2. Hasil turunan parsial tersebut **tidak menimpa (*overwrite*)** nilai pada `param.grad`, melainkan **ditambahkan (*accumulated*)** secara *in-place*:
   $$\mathbf{W}.\text{grad} \leftarrow \mathbf{W}.\text{grad} + \nabla_{\mathbf{W}} \mathcal{L}$$

#### Mengapa PyTorch Mengadopsi Desain Akumulatif Ini?
Desain akumulatif ini sengaja dipilih untuk memberikan fleksibilitas arsitektural:
- **Gradient Accumulation Across Micro-batches**: Ketika ukuran batch fisik tidak muat dalam memori GPU (VRAM), pengembang dapat memecah batch menjadi beberapa *micro-batches*, memanggil `backward()` beberapa kali tanpa memanggil `zero_grad()`, lalu melakukan `optimizer.step()` setelah sejumlah iterasi.
- **Multi-task / Multi-loss Learning**: Menghitung gradien dari beberapa fungsi objektif independen ($\mathcal{L}_{\text{task1}} + \lambda \mathcal{L}_{\text{task2}}$) yang dapat di-backward secara terpisah.

#### Bahaya Jika `optimizer.zero_grad()` Terlupakan
Dalam *standard mini-batch training loop*, jika `optimizer.zero_grad()` tidak dipanggil di awal setiap iterasi, gradien dari batch saat ini akan terus dijumlahkan dengan gradien dari seluruh batch sebelumnya. Hal ini memicu fenomena:
- Nilai gradien membengkak tak terkendali ($\|\mathbf{g}\| \to \infty$).
- Pembaruan bobot melompat jauh melampaui mangkuk optimasi (*divergent updates*).
- Bobot bernilai `NaN` atau `Inf`, yang melumpuhkan proses pembelajaran secara permanen.

---

## 3. Dekonstruksi Mode Operasional: `model.train()` vs `model.eval()`

### 3.1 Atribut `self.training` pada `nn.Module`
Di dalam implementasi basis kelas `torch.nn.Module`, status operasional dikendalikan oleh variabel boolean tunggal:
```python
def train(self: T, mode: bool = True) -> T:
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self

def eval(self: T) -> T:
    return self.train(False)
```
Pemanggilan `model.eval()` hanyalah *alias* untuk `model.train(mode=False)`. Perintah ini menyusuri pohon modul (*module hierarchy tree*) dan menyetel `module.training = False` pada setiap submodul.

### 3.2 Analisis Perilaku Layer Spesifik

Perubahan flag `self.training` memberikan dampak dramatis pada layer-layer berikut:

```
+-------------------+--------------------------------+--------------------------------+
| Layer             | Mode Training (train=True)      | Mode Evaluation (train=False)  |
+-------------------+--------------------------------+--------------------------------+
| nn.Dropout        | Inverted Dropout:              | Identity Function:             |
|                   | y = (x * mask) / (1 - p)       | y = x                          |
|                   | mask ~ Bernoulli(1 - p)        | (Semua neuron aktif)           |
+-------------------+--------------------------------+--------------------------------+
| nn.BatchNorm1d/2d | Hitung statistik batch saat itu| Gunakan running stats beku:    |
|                   | mu_B, var_B dari mini-batch.   | mu_running, var_running.       |
|                   | Update running statistics:     | Tidak ada update statistik.    |
|                   | mu_run = (1-m)*mu_run + m*mu_B |                                |
+-------------------+--------------------------------+--------------------------------+
```

#### A. Mekanika `nn.Dropout`
Pada mode training, dropout meng-nolkan aktivasi tertentu secara acak dan mengalikan sisanya dengan faktor penskalaan $\frac{1}{1-p}$ (*inverted dropout*). Tujuannya adalah memastikan nilai ekspektasi aktivasi tetap sama antara fase training dan evaluasi:
$$\mathbb{E}[y_{\text{train}}] = (1 - p) \cdot \frac{x}{1 - p} = x = y_{\text{eval}}$$
Jika `model.eval()` dipanggil saat training:
- Tidak ada neuron yang di-drop ($p = 0$).
- Model kehilangan seluruh efek regularisasi.
- Neuron mengalami adaptasi bersama (*co-adaptation*), meningkatkan kecenderungan *overfitting*.

#### B. Mekanika `nn.BatchNorm1d` / `nn.BatchNorm2d`
Normalisasi batch bekerja dengan rumus:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
- **Saat Training**: $\mu$ dan $\sigma^2$ dihitung langsung dari sampel mini-batch yang sedang berjalan:
  $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
  Sembari melakukan normalisasi, PyTorch memperbarui *running statistics* akumulatif:
  $$\mu_{\text{running}} \leftarrow (1 - \text{momentum}) \cdot \mu_{\text{running}} + \text{momentum} \cdot \mu_B$$
  $$\sigma^2_{\text{running}} \leftarrow (1 - \text{momentum}) \cdot \sigma^2_{\text{running}} + \text{momentum} \cdot \sigma_B^2$$
- **Saat Evaluation**: Layer **membekukan** pembaruan statistik dan menerapkan nilai $\mu_{\text{running}}$ serta $\sigma^2_{\text{running}}$ yang telah dikumpulkan sepanjang training.
- **Konsekuensi Fatality jika `model.eval()` digunakan saat training**:
  Statistik berjalan tidak pernah diperbarui dari data riil. Nilai $\mu_{\text{running}}$ tetap berada pada nilai default 0 dan $\sigma^2_{\text{running}}$ bernilai 1. Ketika model selesai dilatih dan diuji pada dataset baru menggunakan `model.eval()`, normalisasi menjadi salah kaprah karena menggunakan rata-rata dan variansi default, mengakibatkan prediksi model hancur total (*catastrophic accuracy collapse*).

### 3.3 Miskonsepsi: Hubungan `model.eval()` dan Gradien
Terdapat miskonsepsi umum di kalangan praktisi pemula bahwa `model.eval()` mematikan autograd atau perhitungan gradien. 
- **Fakta**: `model.eval()` **sama sekali tidak menyentuh engine Autograd**.
- Di bawah `model.eval()`, jika `loss.backward()` dipanggil, PyTorch tetap menghitung gradien ke setiap tensor parameter yang memiliki `requires_grad=True`. Dan jika `optimizer.step()` dipanggil, bobot parameter model tetap berubah.
- Hal inilah yang terjadi pada `broken_training_step()`: kode tetap dapat berjalan dan mengembalikan nilai loss yang berkurang, memberikan ilusi semu bahwa model sedang belajar dengan benar, padahal integritas layer terganggu.

---

## 4. Mekanika Autograd Engine: `torch.no_grad()` vs `model.eval()`

Untuk mengoptimalkan tahap evaluasi (validasi dan testing), PyTorch menyediakan context manager `torch.no_grad()` (atau decorator `@torch.no_grad()`).

### 4.1 Perbandingan Mekanisme Internal

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  DIMENSI KONTROL PADA PYTORCH INFERENCE                      │
├──────────────────────────────────────┬───────────────────────────────────────┤
│    KONTROL TINGKAT MODUL             │    KONTROL TINGKAT ENGINE AUTOGRAD    │
│    (nn.Module state)                 │    (C++ Autograd Engine Context)      │
├──────────────────────────────────────┼───────────────────────────────────────┤
│         model.eval()                 │         with torch.no_grad():         │
│  - Mengubah flag internal modul      │  - Mengatur Thread-local GradMode     │
│  - Mengatur perilaku Dropout, BN     │  - Mencegah alokasi DAG node          │
│  - Parameter tetap bisa dilacak      │  - Tidak menyimpan aktivasi buffer    │
│  - Autograd graph TETAP TERBANGUN    │  - Tensor output: grad_fn=None        │
└──────────────────────────────────────┴───────────────────────────────────────┘
```

### 4.2 Analisis Penghematan Memori (Footprint Memory Analysis)
Saat training biasa (forward pass dengan autograd):
$$M_{\text{train}} = M_{\text{params}} + M_{\text{inputs}} + M_{\text{activations}} + M_{\text{graph\_nodes}}$$
Tensor aktivasi perantara (*intermediate activations*) di setiap layer harus ditahan di RAM/VRAM hingga backward pass selesai karena nilai-nilai tersebut dibutuhkan untuk menghitung turunan lokal (misalnya, turunan ReLU membutuhkan nilai aktivasi input apakah $> 0$).

Saat menggunakan `with torch.no_grad()`:
$$M_{\text{eval}} = M_{\text{params}} + M_{\text{inputs}} + M_{\text{current\_layer\_scratchpad}}$$
Aktivasi perantara segera dibuang dari memori begitu layer berikutnya selesai dieksekusi. Hasilnya:
1. **Penghematan VRAM hingga 50% - 75%**, memungkinkan throughput pengujian yang jauh lebih tinggi (dapat menggunakan batch size 2x hingga 4x lebih besar saat evaluasi).
2. **Pengurangan Latensi Komputasi**, karena sistem tidak perlu mengalokasikan memori dinamis untuk pointer graf dan metadata autograd.

---

## 5. Tensor Mechanics: Shape, Dtype, dan Device Management

### 5.1 Shape Alignment & Bahaya Silent Broadcasting
Pada model biner:
- Input $X$: `(16, 2)` $\to$ `nn.Linear(2, 8)` $\to$ `nn.Linear(8, 1)` menghasilkan output berbentuk **`(16, 1)`**.
- Target $y$: dibuat dengan `torch.randint(0, 2, (16, 1)).float()`.

#### Apa yang terjadi jika target $y$ dibuat sebagai vektor 1D `(16,)`?
Pada PyTorch versi modern:
```python
>>> criterion = nn.BCEWithLogitsLoss()
>>> pred = torch.randn(16, 1)
>>> y_1d = torch.randint(0, 2, (16,)).float()
>>> criterion(pred, y_1d)
ValueError: Target size (torch.Size([16])) must be the same as input size (torch.Size([16, 1]))
```
PyTorch melempar `ValueError` eksplisit. Namun, pada operasi berbasis tensor manual atau beberapa loss function kustom, perbedaan dimensi `(16, 1)` dan `(16,)` dapat memicu **Broadcasting Otomatis**:
$$\text{Shape } (16, 1) \otimes (16,) \implies \text{Matrix Shape } (16, 16)$$
Operasi ini menghitung *outer product* antar sampel alih-alih perbandingan elemen berpasangan (*element-wise comparison*), menyebabkan loss dihitung terhadap 256 elemen silang. Ini adalah jebakan tersembunyi yang sangat berbahaya jika tidak dicegah melalui inspeksi batch (`inspect_batch`).

### 5.2 Dtype Constraints pada `nn.BCEWithLogitsLoss`
Label dihasilkan melalui:
```python
y = torch.randint(0, 2, (16, 1)).float()
```
Fungsi `torch.randint` menghasilkan integer 64-bit (`torch.int64` / `torch.long`). Pemanggilan `.float()` mengonversinya menjadi `torch.float32`.

#### Mengapa Target Harus Bertipe Float?
Pada `nn.CrossEntropyLoss` (klasifikasi multikelas), target yang diterima adalah indeks kelas:
$$y \in \{0, 1, 2, \dots, C-1\} \implies \text{Dtype: } \texttt{torch.long}$$
Namun pada `nn.BCEWithLogitsLoss` (klasifikasi biner):
- Target $y$ diperlakukan sebagai nilai probabilitas kontinu $y_i \in [0.0, 1.0]$.
- Perhitungan loss melibatkan operasi perkalian floating-point:
  $$\ell_n = - [y_n \cdot \log \sigma(x_n) + (1 - y_n) \cdot \log (1 - \sigma(x_n))]$$
- Jika target berupa `torch.long`, PyTorch akan menolak operasi tersebut dengan pesan error:
  `RuntimeError: result type Float can't be cast to the desired output type Long`

### 5.3 Device Co-locality & Memory Transfer Semantics
Komputasi tensor pada akselerator perangkat keras (GPU CUDA / Apple MPS) dieksekusi oleh unit aritmatika chip yang hanya memiliki akses langsung ke memori fisiknya sendiri (VRAM). 
- Jika bobot model dialokasikan di VRAM (`cuda:0`) sedangkan data input $X$ berada di memori sistem CPU:
  PyTorch melempar eksepsi:
  `RuntimeError: Expected all tensors to be on the same device...`
- Solusi terbaik adalah menulis kode yang *device-agnostic*:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  X = X.to(device)
  y = y.to(device)
  ```

---

## 6. Stabilitas Numerik: Mengapa `nn.BCEWithLogitsLoss`?

Arsitektur model yang diuji adalah:
```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
```
Model ini **tidak menggunakan layer Sigmoid di akhir**. Mengapa ini adalah desain yang superior?

### 6.1 Perbandingan Formulasi Matematis

Pendekatan naif:
$$\hat{p} = \sigma(x) = \frac{1}{1 + e^{-x}}, \quad \mathcal{L} = - [y \log \hat{p} + (1 - y) \log (1 - \hat{p})]$$
Jika $x$ bernilai negatif yang cukup besar (misal $x = -100$):
- $e^{-x} = e^{100} \to \infty$ (overflow pada kalkulasi eksponensial).
- $\hat{p} = \frac{1}{\infty} = 0.0$.
- Selanjutnya, $\log(\hat{p}) = \log(0) = -\infty$.
- Operasi $0 \times (-\infty)$ menghasilkan **`NaN`** (*Not a Number*).

Sebaliknya, jika $x$ bernilai positif besar (misal $x = 100$):
- $\hat{p} = 1.0$.
- $\log(1 - \hat{p}) = \log(0) = -\infty \implies \textbf{NaN}$.

### 6.2 The Log-Sum-Exp Stabilization Trick
`nn.BCEWithLogitsLoss()` menyatukan aktivasi sigmoid dan logaritma cross-entropy ke dalam bentuk aljabar tunggal:
$$\mathcal{L}(x, y) = -y \log \sigma(x) - (1 - y) \log (1 - \sigma(x))$$
Disederhanakan menjadi:
$$\mathcal{L}(x, y) = x - x \cdot y + \log(1 + e^{-x})$$
Untuk mencegah overflow ketika $x < 0$, PyTorch menerapkan fungsi $\text{relu}(x) = \max(x, 0)$:
$$\mathcal{L}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$
- Karena suku eksponensial dihitung menggunakan $-|x|$, nilai eksponen selalu $\le 0$.
- Nilai $e^{-|x|} \in (0, 1]$, sehingga overflow $e^x \to \infty$ secara matematis mustahil terjadi.
- Hal ini menjamin stabilitas numerik absolut pada komputasi presisi tunggal (`float32`).

---

## 7. Forensic Code Review: Evaluasi dan Implementasi

### 7.1 Bedah Fungsi `broken_training_step` vs `corrected_training_step`

```python
# VERSI SALAH (broken_training_step)
def broken_training_step(model, optimizer, criterion, X, y):
    model.eval()          # CRITICAL BUG: Mematikan mode training!
    optimizer.zero_grad() # Benar: membersihkan gradien
    prediction = model(X) # Terpengaruh: layer beroperasi dalam mode inferensi
    loss = criterion(prediction, y)
    loss.backward()       # Tetap berjalan: autograd tetap aktif
    optimizer.step()      # Tetap berjalan: bobot tetap terupdate
    return loss.item()

# VERSI BENAR (corrected_training_step)
def corrected_training_step(model, optimizer, criterion, X, y):
    model.train()         # FIX: Memastikan mode training aktif
    optimizer.zero_grad() # Benar
    prediction = model(X) # Benar: layer beroperasi dalam mode pembelajaran
    loss = criterion(prediction, y)
    loss.backward()       # Benar: gradien dihitung
    optimizer.step()      # Benar: bobot diperbarui
    return loss.item()
```

### 7.2 Implementasi Standar Fungsi `evaluate_step`
Fungsi evaluasi yang benar wajib memenuhi enam prinsip desain:
1. Menyetel mode evaluasi: `model.eval()`.
2. Mematikan autograd: `with torch.no_grad():`.
3. Menghasilkan raw logits.
4. Menghitung loss evaluasi.
5. Menghitung akurasi biner dengan mengubah logits menjadi keputusan biner:
   $$\hat{y}_{\text{pred}} = \mathbb{I}(\sigma(\text{logits}) \ge 0.5) \iff \mathbb{I}(\text{logits} \ge 0.0)$$
6. Mengembalikan metrik evaluasi dalam tipe numerik murni (`float`).

```python
def evaluate_step(model: nn.Module, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(X)
        loss = criterion(prediction, y)
        
        # Konversi logits ke probabilitas dan threshold biner
        probabilities = torch.sigmoid(prediction)
        predicted_classes = (probabilities >= 0.5).float()
        
        # Hitung akurasi
        correct = (predicted_classes == y).float().sum()
        accuracy = (correct / y.numel()).item()
        
    return loss.item(), accuracy
```

---

## 8. Verifikasi Eksperimental & Log Output

Pengujian menyeluruh dijalankan pada lingkungan virtual `.venv` dengan PyTorch 2.14.0+cu130.

### 8.1 Log Eksekusi `DL_Sesi04_Framework_Debugging_Exercise.py`
```text
Using device: cpu

=== Batch Inspection ===
input shape: (16, 2) | dtype: torch.float32 | device: cpu
label shape: (16, 1) | dtype: torch.float32 | device: cpu
model device: cpu
prediction shape: (16, 1) | dtype: torch.float32
========================

--- Training Step ---
Training Loss (broken_training_step yang telah diperbaiki): 0.7333
Training Loss (corrected_training_step): 0.7317

--- Evaluation Step ---
Evaluation Loss    : 0.6760
Evaluation Accuracy: 62.50%
```

### 8.2 Analisis Kuantitatif Hasil
1. **Inspeksi Batch**:
   - `input shape: (16, 2)` dan `label shape: (16, 1)` berada pada perangkat yang sama (`cpu`), bertipe identik (`torch.float32`).
   - Tidak ada peringatan mismatch dimensi atau penyesuaian broadcasting implisit.
2. **Dinamika Loss Training**:
   - Langkah pertama menghasilkan loss `0.7333`.
   - Setelah bobot diperbarui oleh optimizer, langkah kedua pada batch yang sama menghasilkan loss `0.7317` (terjadi penurunan nilai loss sebesar $0.0016$), membuktikan bahwa perambatan gradien dan pembaruan bobot optimizer berfungsi dengan tepat.
3. **Hasil Evaluasi**:
   - Fungsi `evaluate_step` berhasil mengevaluasi batch validasi baru tanpa gradien, menghasilkan loss `0.6760` dan akurasi biner `62.50%` (10 dari 16 sampel terklasifikasi dengan benar secara probabilistik pada inisialisasi awal).

---

## 9. Framework Debugging Checklist (Pre-Flight Protocol)

Sebelum menjalankan pelatihan model skala besar (*long training run*), lakukan 10 langkah pemeriksaan berikut untuk mengeliminasi potensi silent bugs:

```markdown
- [ ] 1. Shape Input Compatibility: Periksa bahwa dimensi tensor input cocok dengan in_features layer pertama (X.shape[1:] == in_features).
- [ ] 2. Shape Label Exact Matching: Periksa bahwa dimensi label persis sama dengan output model (misal (N, 1) vs (N, 1)), bukan (N,).
- [ ] 3. Dtype Conformity: Pastikan target float32 untuk BCEWithLogitsLoss/MSELoss, atau int64 untuk CrossEntropyLoss.
- [ ] 4. Device Co-locality: Pastikan seluruh parameter model dan data batch berada pada objek torch.device yang sama.
- [ ] 5. Gradient Reset Integrity: Pastikan optimizer.zero_grad() dipanggil sebelum loss.backward() pada setiap iterasi mini-batch.
- [ ] 6. Explicit Operational Mode: Pastikan model.train() aktif sebelum training loop dan model.eval() aktif sebelum validation loop.
- [ ] 7. Evaluation Context Isolation: Pastikan forward pass validasi/testing selalu dibungkus dengan with torch.no_grad():.
- [ ] 8. Loss Function Coupling: Pastikan arsitektur model tidak memiliki aktivasi sigmoid ganda jika menggunakan BCEWithLogitsLoss.
- [ ] 9. Metric Extraction Detachment: Saat mencatat loss/akurasi ke logger, gunakan loss.item() atau .detach() untuk mencegah computation graph memory leak.
- [ ] 10. Gradient Norm Sanity Check: Pantau nilai torch.nn.utils.clip_grad_norm_ untuk mendeteksi vanishing atau exploding gradients sejak dini.
```

---

## 10. Kesimpulan

Debugging pada framework Deep Learning menuntut pemahaman mendalam yang melampaui sintaks bahasa pemrograman Python biasa. Pengembang harus memahami interaksi antara **arsitektur modul**, **mesin diferensiasi otomatis (Autograd)**, dan **manajemen alokasi memori tensor**.

Melalui latihan ini terbukti bahwa:
1. `model.eval()` dan `torch.no_grad()` memiliki domain tanggung jawab yang berbeda dan saling melengkapi: `model.eval()` mengatur perilaku layer (Dropout, BatchNorm), sedangkan `torch.no_grad()` menghentikan pelacakan graf autograd untuk efisiensi memori dan waktu komputasi.
2. Kesalahan meletakkan `model.eval()` saat pelatihan merupakan contoh nyata *silent bug* yang tidak memicu error kompilasi, namun berakibat fatal pada generalisasi model.
3. Kepatuhan terhadap disiplin verifikasi dimensi tensor, tipe data numerik, dan siklus optimasi PyTorch merupakan fondasi mutlak dalam membangun sistem Deep Learning yang andal dan teruji secara ilmiah.
