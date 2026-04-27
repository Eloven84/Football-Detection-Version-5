from rfdetr import RFDETRBase
import torch

def main():
    model = RFDETRBase(
        num_classes=4,
        pretrain_weights=r"C:\AppEliza\File Tugas Akhir\Football2\models\RFDETR Result Dataset With Augmentation Version 1\checkpoint0119.pth",
        # 🔹 Langsung pakai checkpoint0119.pth, versi 1.3 bisa handle!
        freeze_backbone=True
    )

    model.train(
        dataset_dir=r"C:\AppEliza\File Tugas Akhir\Football2\Dataset (Roboflow)\Football Dataset Version 3 (RF-DETR)",
        epochs=20,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-5,
        lr_drop=20,
        warmup_epochs=2,
        weight_decay=1e-4,
        use_ema=True,
        ema_decay=0.9998,
        output_dir=r"C:\AppEliza\File Tugas Akhir\Football2\models\RFDETR Result Dataset With Augmentation Version 2",
        num_workers=2,
    )

if __name__ == "__main__":
    main()

# from rfdetr import RFDETRBase
# import torch
# import argparse

# torch.serialization.add_safe_globals([argparse.Namespace])

# def main():
#     model = RFDETRBase(
#         num_classes=4,
#         pretrain_weights=r"C:\AppEliza\File Tugas Akhir\Football2\models\RFDETR Result Dataset With Augmentation Version 1\checkpoint_best_ema.pth"
#     )

#     for name, param in model.model.model.named_parameters():
#         if "backbone" in name:
#             param.requires_grad = False

#     model.train(
#         dataset_dir=r"C:\AppEliza\File Tugas Akhir\Football2\Dataset (Roboflow)\Football Dataset Version 3 (RF-DETR)",
#         epochs=20,
#         batch_size=2,
#         grad_accum_steps=4,
#         lr=1e-5,
#         lr_scheduler="cosine",
#         warmup_epochs=2,
#         weight_decay=1e-4,
#         use_ema=True,
#         ema_decay=0.9998,
#         output_dir=r"C:\AppEliza\File Tugas Akhir\Football2\models\RFDETR Result Dataset With Augmentation Version 2",
#         num_workers=2,
#         save_best=True,
#         resume=r"C:\AppEliza\File Tugas Akhir\Football2\models\RFDETR Result Dataset With Augmentation Version 1\checkpoint0119.pth"
#         # 🔹 Ganti ke checkpoint0119.pth bukan checkpoint_best_ema.pth
#     )

# if __name__ == "__main__":
#     main()