"""
Konfigurasi global — sesuaikan bagian ini dengan setup lokal kamu.
"""
import os
 
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MySQL / phpMyAdmin ────────────────────────────────────────
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_DATABASE = 'football_detection'
# SOCKET = '/Applications/XAMPP/xamppfiles/var/mysql/mysql.sock'
# CONNECTION_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}?unix_socket={SOCKET}"
CONNECTION_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}"

# ── Model RF-DETR ─────────────────────────────────────────────
MODEL_WEIGHTS  = "E:\FOLDER TUGAS AKHIR 2\Football2\models\RFDETR Result Dataset With Augmentation Version 1\checkpoint_best_total.pth"

# ── Flask ─────────────────────────────────────────────────────
SECRET_KEY     = 'football_detection'
MAX_CONTENT_MB = 2048
ALLOWED_EXTS   = {'mp4', 'avi', 'mov', 'mkv'}

# ── Celery + Redis ────────────────────────────────────────────
# Jika tidak punya Redis, ganti ke SQLite broker:
#   CELERY_BROKER_URL = 'sqla+sqlite:///celery_broker.db'
#   CELERY_RESULT_BACKEND = 'db+sqlite:///celery_results.db'
CELERY_BROKER_URL     = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

"""
Solusi dengan Celery
Celery memisahkan "terima request" dan "kerjakan tugas" menjadi dua proses berbeda:
User klik Apply Model
        ↓
Flask: "Oke, sudah diterima! Video masuk antrian."  ← langsung balas (< 1 detik)
        ↓
Celery Worker: mengerjakan analisis RF-DETR di background
        ↓
Frontend polling /api/status setiap 3 detik
        ↓
Setelah selesai → tampilkan hasil


Bayangkan kamu memesan makanan di restoran:
- Tanpa Celery = kamu berdiri di depan kasir, menunggu sampai makanannya jadi (5–30 menit), baru bisa pergi
- Dengan Celery = kamu pesan, dapat nomor antrian, duduk bebas, nanti dipanggil kalau sudah jadi
"""