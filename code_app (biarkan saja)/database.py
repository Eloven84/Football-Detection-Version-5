"""
Database Schema untuk Aplikasi Analisis Video Sepak Bola
Tabel:
  Videos            → video_id (PK), video_name, team1_name, team2_name,
                       file_path_raw, duration_sec, fps, resolution, uploaded_at
  AnalysisResults   → analysis_id (PK), video_id (FK), model_name, tracking_method,
                       processing_time_sec, anotated_video_path, analysis_date,
                       ball_posession_home, ball_posession_away, total_passes
  AnalysisMetadata  → metadata_id (PK), analysis_id (FK), analysis_data (JSON)
"""
from flask_sqlalchemy import SQLAlchemy
# from flask import Flask, request, jsonify
from datetime import datetime, date
import json

db = SQLAlchemy()

