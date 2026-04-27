import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import numpy as np
import pandas as pd
from IPython.display import Image
import cv2
from glob import glob
import re
import yaml

# ==============================
# SET PATH (WINDOWS VERSION)
# ==============================

sn_track_base = 'C:/AppEliza/File Tugas Akhir/Football2/SoccerNet_dataset'
yolo_base = 'C:/AppEliza/File Tugas Akhir/Football2/yolo8'

os.makedirs(sn_track_base, exist_ok=True)

# bikin dummy file (sesuai kode asli)
open(os.path.join(sn_track_base, "test.zip"), 'a').close()
open(os.path.join(sn_track_base, "challenge.zip"), 'a').close()

# ==============================
# DOWNLOAD DATASET
# ==============================

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=sn_track_base)

mySoccerNetDownloader.downloadDataTask(
    task="tracking",
    split=["train", "test", "challenge"]  # download train dulu
)

# ==============================
# EXTRACT ZIP
# ==============================
train_zip_path = os.path.join(sn_track_base, "tracking", "train.zip")

print("Path yang dipakai:", train_zip_path)

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(sn_track_base, "tracking"))

print("Download dan extract selesai.")

# ==============================
# TAMPILKAN SAMPLE IMAGE
# ==============================

base_path = os.path.join(sn_track_base, "tracking", "train")

match_folder = os.listdir(base_path)[0]
img_folder = os.path.join(base_path, match_folder, "img1")

first_image = os.listdir(img_folder)[10]
image_path = os.path.join(img_folder, first_image)

img = Image.open(image_path)

plt.imshow(img)
plt.axis("off")
plt.title(first_image)
plt.show()