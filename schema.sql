-- ═══════════════════════════════════════════════════════════
--  Football Detection — Database Schema (Updated)
--  Calvin Institute of Technology — Tugas Akhir
--
--  Perubahan dari versi sebelumnya:
--    + team1_goalkeeper_color, team2_goalkeeper_color di Videos
--    + zone_grid_path, zone_voronoi_path di AnalysisResults
--    + Fix syntax: hilangkan koma yang hilang setelah team1_color_secondary
--    + Tambah index untuk performa query
--  Jalankan di phpMyAdmin > SQL tab
-- ═══════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS football_detection
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE football_detection;

-- ─────────────────────────────────────────────────────────
--  Table: Videos
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Videos (
    video_id                  INT AUTO_INCREMENT PRIMARY KEY,

    -- Informasi video
    video_name                VARCHAR(255)  NOT NULL,
    file_path_raw             TEXT,
    file_size_mb              FLOAT,

    -- Informasi tim
    team1_name                VARCHAR(100)  DEFAULT 'Tim 1',
    team1_color_primary       VARCHAR(10)   DEFAULT '#FFFFFF',
    team1_color_secondary     VARCHAR(10)   DEFAULT '#E53E3E',
    team1_goalkeeper_color    VARCHAR(10)   DEFAULT '#F6AD55',

    team2_name                VARCHAR(100)  DEFAULT 'Tim 2',
    team2_color_primary       VARCHAR(10)   DEFAULT '#FFFFFF',
    team2_color_secondary     VARCHAR(10)   DEFAULT '#3182CE',
    team2_goalkeeper_color    VARCHAR(10)   DEFAULT '#48BB78',

    referee_color             VARCHAR(10)   DEFAULT '#ECC94B',

    -- Status pemrosesan
    status                    VARCHAR(50)   DEFAULT 'queued',
    progress_pct              INT           DEFAULT 0,
    error_msg                 TEXT,

    -- Metadata video
    duration_sec              INT,
    fps                       FLOAT,
    resolution                VARCHAR(50),

    created_at                TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    updated_at                TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status   (status),
    INDEX idx_created  (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────
--  Table: AnalysisResults
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisResults (
    analysis_id              INT AUTO_INCREMENT PRIMARY KEY,
    video_id                 INT           NOT NULL,

    -- Model & tracking info
    model_name               VARCHAR(100)  DEFAULT 'RF-DETR Base',
    tracking_method          VARCHAR(100)  DEFAULT 'ByteTrack',
    processing_time_sec      FLOAT,
    analysis_date            DATETIME,

    -- Path file output
    anotated_video_path      TEXT,
    heatmap_home_path        TEXT,
    heatmap_away_path        TEXT,
    zone_grid_path           TEXT,          -- ← BARU: path zone grid PNG
    zone_voronoi_path        TEXT,          -- ← BARU: path zone voronoi PNG

    -- Statistik utama
    ball_posession_home      FLOAT         DEFAULT 50.0,
    ball_posession_away      FLOAT         DEFAULT 50.0,
    total_passes             INT           DEFAULT 0,
    ball_speed_max_kmh       FLOAT,
    ball_speed_avg_kmh       FLOAT,
    player_speed_max_kmh     FLOAT,
    total_frames             INT,

    created_at               TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES Videos(video_id) ON DELETE CASCADE,
    INDEX idx_video_id (video_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────
--  Table: AnalysisMetadata
--  Menyimpan data JSON mentah (kecepatan per tim, stats
--  heatmap, prediksi, dll) agar fleksibel tanpa ALTER TABLE.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisMetadata (
    metadata_id     INT  AUTO_INCREMENT PRIMARY KEY,
    analysis_id     INT  NOT NULL,
    analysis_data   JSON,                  -- dict Python di-serialize ke JSON
    FOREIGN KEY (analysis_id) REFERENCES AnalysisResults(analysis_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────
--  Contoh isi analysis_data (JSON):
--  {
--    "avg_speed_team1": 12.3,
--    "avg_speed_team2": 11.8,
--    "heatmap_home_path": "outputs/1/heatmap_home.jpg",
--    "heatmap_away_path": "outputs/1/heatmap_away.jpg",
--    "zone_grid_path":    "outputs/1/1_zone_grid.png",
--    "zone_voronoi_path": "outputs/1/1_zone_voronoi.png",
--    "heatmap_stats": {
--        "total_frames_analyzed": 750,
--        "ball_possession": { "team1_pct": 54.2, "team2_pct": 40.1, "no_possession_pct": 5.7 },
--        "zone_coverage":   { "team1_pct": 38.4, "team2_pct": 29.1 },
--        "position_datapoints": { "team1": 8200, "team2": 7600 }
--    },
--    "ball_speed": { "max_kmh": 0, "avg_kmh": 0 },
--    "prediction": { "predicted_winner": 1, "confidence_pct": 67, "score_team1": "...", "score_team2": "..." }
--  }
-- ─────────────────────────────────────────────────────────


-- ═══════════════════════════════════════════════════════════
--  MIGRATION: Jalankan bagian ini HANYA jika sudah ada
--  database sebelumnya dan ingin menambah kolom baru.
--  Jika fresh install, blok di bawah bisa di-skip / comment.
-- ═══════════════════════════════════════════════════════════

-- ALTER TABLE Videos
--     ADD COLUMN team1_goalkeeper_color VARCHAR(10) DEFAULT '#F6AD55' AFTER team1_color_secondary,
--     ADD COLUMN team2_goalkeeper_color VARCHAR(10) DEFAULT '#48BB78' AFTER team2_color_secondary;

-- ALTER TABLE AnalysisResults
--     ADD COLUMN zone_grid_path    TEXT AFTER heatmap_away_path,
--     ADD COLUMN zone_voronoi_path TEXT AFTER zone_grid_path;