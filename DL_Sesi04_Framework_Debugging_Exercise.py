"""Deep Learning Session 4 — Framework Debugging Exercise.

Kode ini telah diperbaiki dan dilengkapi dengan implementasi evaluate_step()
serta perbaikan pada fungsi broken_training_step().
"""

import torch
from torch import nn
from typing import Tuple


def broken_training_step(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Versi yang telah diperbaiki dari broken_training_step.
    
    Bug sebelumnya:
        Menggunakan model.eval() saat proses training, yang menonaktifkan perilaku
        khusus layer tertentu (seperti Dropout dan BatchNorm).
    
    Perbaikan:
        Mengubah model.eval() menjadi model.train() agar model berada pada mode pelatihan.
    """
    model.train()  # PERBAIKAN: Mengaktifkan mode training (sebelumnya model.eval())
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def original_broken_training_step(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Intentionally broken example for classroom code review and comparison."""
    model.eval()  # BUG: training should normally use model.train().
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def corrected_training_step(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Corrected version with the training mode explicitly enabled."""
    model.train()
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_step(model: nn.Module, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """
    Langkah evaluasi model pada dataset validasi/pengujian.
    
    Fungsi ini melakukan:
    1. Mengaktifkan evaluation mode (model.eval()).
    2. Menonaktifkan gradient tracking (with torch.no_grad()).
    3. Menghasilkan prediksi (forward pass).
    4. Menghitung loss evaluasi.
    5. Menghitung binary classification accuracy.
    6. Mengembalikan nilai loss dan accuracy sebagai tuple (loss, accuracy).
    """
    # 1. Mengaktifkan evaluation mode
    model.eval()
    
    # 2. Menonaktifkan gradient tracking untuk efisiensi memori dan komputasi
    with torch.no_grad():
        # 3. Menghasilkan prediksi
        prediction = model(X)
        
        # 4. Menghitung loss
        loss = criterion(prediction, y)
        
        # 5. Menghitung binary accuracy
        probabilities = torch.sigmoid(prediction)
        predicted_labels = (probabilities >= 0.5).float()
        correct_predictions = (predicted_labels == y).float().sum()
        accuracy = (correct_predictions / y.numel()).item()
        
    # 6. Mengembalikan nilai loss dan accuracy
    return loss.item(), accuracy


def inspect_batch(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    """Print the checks students should perform before a long training run."""
    print("=== Batch Inspection ===")
    print("input shape:", tuple(X.shape), "| dtype:", X.dtype, "| device:", X.device)
    print("label shape:", tuple(y.shape), "| dtype:", y.dtype, "| device:", y.device)
    print("model device:", next(model.parameters()).device)
    with torch.no_grad():
        prediction = model(X)
    print("prediction shape:", tuple(prediction.shape), "| dtype:", prediction.dtype)
    print("=" * 24)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Inisialisasi arsitektur model biner
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Data batch sintetis (batch_size=16, input_features=2)
    X = torch.randn(16, 2, device=device)
    y = torch.randint(0, 2, (16, 1), device=device).float()

    # 1. Pemeriksaan dimensi, tipe data, dan perangkat
    inspect_batch(model, X, y)

    # 2. Demonstrasi langkah pelatihan
    print("\n--- Training Step ---")
    train_loss = broken_training_step(model, optimizer, criterion, X, y)
    print(f"Training Loss (broken_training_step yang telah diperbaiki): {train_loss:.4f}")

    train_loss_corrected = corrected_training_step(model, optimizer, criterion, X, y)
    print(f"Training Loss (corrected_training_step): {train_loss_corrected:.4f}")

    # 3. Perbandingan adil antara original_broken_training_step dan corrected_training_step
    #    Model & optimizer dibuat TERPISAH untuk masing-masing fungsi (tidak dipanggil
    #    berurutan pada model yang sama), dengan seed yang sama, agar keduanya berangkat
    #    dari inisialisasi bobot yang identik dan hasilnya bisa dibandingkan secara adil.
    print("\n--- Fair Comparison: original_broken_training_step vs corrected_training_step ---")

    torch.manual_seed(42)
    model_broken = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device)
    optimizer_broken = torch.optim.Adam(model_broken.parameters(), lr=1e-3)

    print(f"[original_broken_training_step] mode sebelum: training={model_broken.training}")
    loss_broken = original_broken_training_step(model_broken, optimizer_broken, criterion, X, y)
    print(f"[original_broken_training_step] mode sesudah : training={model_broken.training}")
    print(f"[original_broken_training_step] loss={loss_broken:.6f}")

    torch.manual_seed(42)
    model_corrected = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device)
    optimizer_corrected = torch.optim.Adam(model_corrected.parameters(), lr=1e-3)

    print(f"[corrected_training_step] mode sebelum: training={model_corrected.training}")
    loss_corrected = corrected_training_step(model_corrected, optimizer_corrected, criterion, X, y)
    print(f"[corrected_training_step] mode sesudah : training={model_corrected.training}")
    print(f"[corrected_training_step] loss={loss_corrected:.6f}")

    # 4. Demonstrasi langkah evaluasi
    print("\n--- Evaluation Step ---")
    X_val = torch.randn(16, 2, device=device)
    y_val = torch.randint(0, 2, (16, 1), device=device).float()
    eval_loss, eval_acc = evaluate_step(model, criterion, X_val, y_val)
    print(f"Evaluation Loss    : {eval_loss:.4f}")
    print(f"Evaluation Accuracy: {eval_acc * 100:.2f}%")
