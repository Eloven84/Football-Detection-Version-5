"""
Flask Web App — Football Analysis System
Calvin Institute of Technology
 
Routes HTML:
  GET  /                        → halaman utama (SPA single-page)
 
Routes API:
  POST   /api/upload            → upload video, masukkan antrian
  GET    /api/videos            → daftar semua video (+ pagination, search)
  GET    /api/video/<id>        → detail + hasil analisis satu video
  PUT    /api/video/<id>        → update nama / tim video
  DELETE /api/video/<id>        → hapus video + semua hasil
  GET    /outputs/<id>/<file>   → serve file output (video, gambar)
"""

import os

