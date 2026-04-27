
from rfdetr import RFDETRBase
import torch

def main():
    # # Device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inisialisasi model
    model = RFDETRBase(
        num_classes=4,   # ball, goalkeeper, player, referee
        pretrain_weights="rf-detr-baseh"
        freeze_backbone=True  # 🔹 stabilkan training awal
    )

    # model.to(device)

    # Training
    model.train(
        # dataset_dir="Football Dataset with Augmentation Version 1 (RF-DETR)",
        dataset_dir = "E:\FOLDER TUGAS AKHIR 2\Football2\Dataset\Football Dataset with Augmentation Version 1 (RF-DETR)",
        # dataset_dir = "E:\FOLDER TUGAS AKHIR 2\Football2\Dataset (Roboflow)\Football Dataset Version 3 after Train (RF-DETR)",
        epochs=120,                    # 🔹 tambah epoch
        batch_size=2,
        grad_accum_steps=4,           # 🔹 batch efektif = 8
        lr=5e-5,                      # 🔹 lebih stabil
        lr_scheduler="cosine",        # 🔹 penting!
        warmup_epochs=5,              # 🔹 cegah early instability
        weight_decay=1e-4,
        use_ema=True,                 # 🔹 EMA terbukti lebih baik
        ema_decay=0.9998,             
        output_dir="models/RFDETR Result Dataset With Augmentation Version 1",
        num_workers=2,
        save_best=True                # 🔹 simpan model terbaik
    )

if __name__ == "__main__":
    main()