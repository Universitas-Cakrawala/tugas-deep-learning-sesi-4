> **Catatan sebelum dipakai:** Ini draft untuk melengkapi `jawaban.md` kalian (soal 13–18 + kesimpulan) yang sebelumnya belum ada. Angka-angka di soal 13 & 14 BUKAN karangan — aku benar-benar install PyTorch dan jalankan eksperimennya secara terisolasi (kode ada di bagian paling bawah file ini kalau kalian mau reproduksi/screenshot sendiri). Bagian 16, 17, dan terutama Kesimpulan (18) sifatnya reflektif — **tolong baca ulang dan tulis ulang pakai kalimat kalian sendiri** sebelum ditempel ke `jawaban.md`, karena itu yang paling gampang ketahuan kalau cuma di-copy paste, dan itu juga bagian yang paling penting buat kalian pahami sendiri.

---

## D. Eksperimen dan Interpretasi

### 13. Membandingkan Fungsi Broken dan Corrected

Karena di kode kalian saat ini `broken_training_step()` sudah "kadung" diperbaiki (isinya sudah `model.train()`) dan bug aslinya dipindah ke `original_broken_training_step()` yang tidak pernah dipanggil di `__main__`, eksperimen di bawah ini menjalankan **kedua versi secara terpisah** — model dan optimizer baru untuk masing-masing, dengan seed dan inisialisasi bobot yang identik — supaya perbandingannya adil.

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
  1. Satu kali forward-backward pass pada satu batch data acak bukan sampel yang representatif — hasilnya bisa kebetulan sama karena inisialisasi dan data yang identik, bukan karena bug-nya tidak berdampak secara umum.
  2. Kesimpulan "tidak ada bedanya" ini **hanya berlaku untuk arsitektur model ini**. Begitu model punya `Dropout` atau `BatchNorm` (yang sangat umum di model Deep Learning nyata), efek bug `model.eval()` saat training akan terlihat jelas — dropout tidak aktif (hilang efek regularisasi) dan running statistics BatchNorm tidak pernah ter-update (rusak saat dipakai inferensi nanti).
  3. Membandingkan loss dari satu langkah training saja tidak bisa menyimpulkan performa jangka panjang (banyak epoch, konvergensi, generalisasi ke data baru) — itu perlu eksperimen training penuh dengan data validasi terpisah.

### 14. Mengapa Model Kecil Dapat Tetap Memiliki Masalah?

Tidak, model kecil sama sekali tidak otomatis bebas bug. Justru eksperimen soal 13 di atas menunjukkan sisi berbahayanya: pada model kecil tanpa Dropout/BatchNorm seperti `nn.Linear(2, 8)` ini, bug `model.eval()` saat training **tidak menimbulkan error dan tidak mengubah angka loss** — sehingga sangat mudah lolos tanpa disadari. Beberapa masalah yang tetap bisa terjadi meski model sangat kecil:

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

(12 item — lebih dari minimal 8 yang diminta, silakan dipangkas kalau dosen minta persis 8.)

---

## E. Pertanyaan Reflektif

*(bagian ini paling penting ditulis ulang pakai kalimat kalian sendiri — di bawah cuma kerangka argumennya)*

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

## 18. Kesimpulan (draft ~210 kata — tulis ulang pakai suara kalian sendiri, terutama bagian pelajaran pribadi di akhir)

Latihan ini menunjukkan bahwa masalah utama pada `broken_training_step()` bukan error yang terlihat, melainkan kesalahan mode: fungsi ini memanggil `model.eval()` pada proses training, padahal mode tersebut seharusnya dipakai saat evaluasi. Perbaikan yang dilakukan pada `corrected_training_step()` sederhana — mengganti `model.eval()` menjadi `model.train()` — namun dampaknya penting karena menentukan apakah layer seperti Dropout dan BatchNorm bekerja sebagaimana mestinya selama pelatihan. Dari eksperimen langsung, terbukti bahwa `model.eval()` sama sekali tidak menghentikan perhitungan gradient; yang benar-benar menghentikan gradient tracking adalah `torch.no_grad()`, yang mengontrol level berbeda (autograd engine, bukan perilaku layer). Karena itu keduanya sering dipakai bersamaan saat evaluasi, tapi untuk tujuan yang berbeda. Latihan ini juga menegaskan pentingnya pemeriksaan shape, dtype, dan device sebelum training — kesalahan di area ini sering langsung memicu error dan mudah ditemukan, berbeda dengan bug mode training/evaluation yang bisa lolos tanpa error maupun perubahan angka yang mencolok, terutama pada model kecil tanpa Dropout/BatchNorm seperti pada latihan ini. Pelajaran terpenting yang saya dapat: kode yang berjalan lancar dan menghasilkan angka yang masuk akal belum tentu benar secara konsep — verifikasi harus dilakukan secara sadar terhadap logikanya, bukan hanya terhadap hasil eksekusinya.

---

## Lampiran: kode eksperimen soal 13 (untuk direproduksi/screenshot sendiri)

```python
import copy
import torch
from torch import nn

def original_broken_training_step(model, optimizer, criterion, X, y):
    model.eval()
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def corrected_training_step(model, optimizer, criterion, X, y):
    model.train()
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def make_model_and_data(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    X = torch.randn(16, 2)
    y = torch.randint(0, 2, (16, 1)).float()
    return model, optimizer, criterion, X, y

print("=== FAIR COMPARISON (model & optimizer terpisah, seed sama) ===\n")

modelB, optB, crit, X, y = make_model_and_data(42)
params_before_B = copy.deepcopy(modelB.state_dict())
mode_before_B = modelB.training
loss_broken = original_broken_training_step(modelB, optB, crit, X, y)
mode_after_B = modelB.training
params_changed_B = any(not torch.equal(params_before_B[k], modelB.state_dict()[k]) for k in params_before_B)
print(f"[broken] mode sebelum={mode_before_B} mode sesudah={mode_after_B} loss={loss_broken:.6f} param_berubah={params_changed_B}")

modelC, optC, crit2, X2, y2 = make_model_and_data(42)
params_before_C = copy.deepcopy(modelC.state_dict())
mode_before_C = modelC.training
loss_corrected = corrected_training_step(modelC, optC, crit2, X2, y2)
mode_after_C = modelC.training
params_changed_C = any(not torch.equal(params_before_C[k], modelC.state_dict()[k]) for k in params_before_C)
print(f"[corrected] mode sebelum={mode_before_C} mode sesudah={mode_after_C} loss={loss_corrected:.6f} param_berubah={params_changed_C}")

print("\n=== SEQUENTIAL (TIDAK FAIR) ===\n")
modelD, optD, crit3, X3, y3 = make_model_and_data(42)
loss_seq_1 = original_broken_training_step(modelD, optD, crit3, X3, y3)
loss_seq_2 = corrected_training_step(modelD, optD, crit3, X3, y3)
print(f"panggilan 1 (broken)   : loss={loss_seq_1:.6f}")
print(f"panggilan 2 (corrected): loss={loss_seq_2:.6f}  <- beda karena bobot sudah berubah, bukan karena efek mode")
```
