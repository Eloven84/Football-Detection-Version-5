-- ═══════════════════════════════════════════════════════════
--  Football Detection — Database Schema
--  Jalankan di phpMyAdmin > SQL tab
-- ═══════════════════════════════════════════════════════════
CREATE DATABASE IF NOT EXISTS soccana_football_detection
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE soccana_football_detection;

-- ─────────────────────────────────────────────────────────
--  Table: Videos
--  Menyimpan metadata video input + warna tim yang dipilih user
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Videos (
    video_id                INT AUTO_INCREMENT PRIMARY KEY,
    video_name              VARCHAR(255)    NOT NULL,

    -- Tim 1
    team1_name              VARCHAR(100)    DEFAULT 'Tim 1',
    team1_color_primary     VARCHAR(10)     DEFAULT '#FFFFFF',   -- auto-detected dari jersey (hex)
    team1_color_secondary   VARCHAR(10)     DEFAULT '#E53E3E',   -- dipilih user (hex)
    team1_goalkeeper_color  VARCHAR(10)     DEFAULT '#F6AD55',   -- GK tim 1 (hex)

    -- Tim 2
    team2_name              VARCHAR(100)    DEFAULT 'Tim 2',
    team2_color_primary     VARCHAR(10)     DEFAULT '#FFFFFF',   -- auto-detected dari jersey (hex)
    team2_color_secondary   VARCHAR(10)     DEFAULT '#3182CE',   -- dipilih user (hex)
    team2_goalkeeper_color  VARCHAR(10)     DEFAULT '#48BB78',   -- GK tim 2 (hex)

    -- Wasit
    referee_color           VARCHAR(10)     DEFAULT '#ECC94B',   -- dipilih user (hex)

    -- File info
    file_path_raw           TEXT,           -- path video input di lokal
    file_path_output        TEXT,           -- path video output (annotated) di lokal
    file_size_mb            FLOAT,

    -- Status pipeline
    status                  VARCHAR(50)     DEFAULT 'queued',    -- queued | processing | done | error
    progress_pct            INT             DEFAULT 0,
    error_msg               TEXT,

    -- Metadata video
    duration_sec            INT,
    fps                     FLOAT,
    resolution              VARCHAR(50),    -- contoh: "1920x1080"

    created_at              TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ─────────────────────────────────────────────────────────
--  Table: AnalysisResults
--  Menyimpan hasil numerik analisis + path video output
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisResults (
    analysis_id             INT AUTO_INCREMENT PRIMARY KEY,
    video_id                INT             NOT NULL,

    -- Model info
    model_name              VARCHAR(100)    DEFAULT 'RF-DETR Base',
    tracking_method         VARCHAR(100)    DEFAULT 'ByteTrack',
    processing_time_sec     FLOAT,

    -- Output video (disimpan di lokal, path-nya di sini)
    anotated_video_path     TEXT,           -- path file .avi/.mp4 output

    -- Statistik utama
    ball_posession_home     FLOAT           DEFAULT 50.0,
    ball_posession_away     FLOAT           DEFAULT 50.0,
    total_passes            INT             DEFAULT 0,
    ball_speed_max_kmh      FLOAT,
    ball_speed_avg_kmh      FLOAT,
    player_speed_max_kmh    FLOAT,
    avg_speed_team1_kmh     FLOAT,
    avg_speed_team2_kmh     FLOAT,
    total_frames            INT,

    analysis_date           DATETIME,
    created_at              TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES Videos(video_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ─────────────────────────────────────────────────────────
--  Table: AnalysisImages
--  Menyimpan gambar output sebagai BLOB di database
--  Kenapa di database?  → mudah di-query, tidak perlu urus file system,
--  bisa ditampilkan langsung via Flask endpoint /api/image/<id>
--  Ukuran: PNG heatmap ~300–800 KB, zone ~200–500 KB → aman di LONGBLOB (max 4 GB)
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisImages (
    image_id        INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id     INT             NOT NULL,
    image_type      VARCHAR(50)     NOT NULL,   -- heatmap_team1 | heatmap_team2 | heatmap_combined
                                                -- zone_control | zone_grid | zone_voronoi
    image_data      LONGBLOB        NOT NULL,   -- raw bytes PNG/JPG
    mime_type       VARCHAR(30)     DEFAULT 'image/png',
    file_size_kb    INT,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (analysis_id) REFERENCES AnalysisResults(analysis_id) ON DELETE CASCADE,
    INDEX idx_analysis_type (analysis_id, image_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ─────────────────────────────────────────────────────────
--  Table: AnalysisMetadata
--  Menyimpan data JSON tambahan (zone stats, heatmap stats, dll.)
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisMetadata (
    metadata_id     INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id     INT     NOT NULL,

    -- Ringkasan heatmap (dari HeatmapAnalyzer.get_summary_stats())
    heatmap_stats   JSON,

    -- Ringkasan zone (dari ZoneAnalyzer._compute_stats())
    zone_stats      JSON,

    -- Data extra / extensible
    extra_data      JSON,

    FOREIGN KEY (analysis_id) REFERENCES AnalysisResults(analysis_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;