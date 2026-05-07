-- ================================================================
-- Football Analysis System — Database Schema
-- Calvin Institute of Technology
-- Import file ini ke phpMyAdmin
-- ================================================================

CREATE DATABASE IF NOT EXISTS football_detection
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE football_detection;

-- ── Tabel 1: Videos ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Videos (
  video_id      INT           NOT NULL AUTO_INCREMENT,
  video_name    VARCHAR(255)  NOT NULL,
  team1_name    VARCHAR(255)  DEFAULT NULL,
  team2_name    VARCHAR(255)  DEFAULT NULL,
  file_path_raw VARCHAR(512)  NOT NULL,
  duration_sec  INTEGER       DEFAULT 0,
  fps           FLOAT         DEFAULT 0,
  resolution    VARCHAR(20)   DEFAULT NULL,
  uploaded_at   DATE          DEFAULT (CURDATE()),
  -- Kolom operasional (tidak ada di desain, tapi dibutuhkan sistem)
  status        VARCHAR(30)   DEFAULT 'uploaded'
                              COMMENT 'uploaded | queued | processing | done | error',
  progress_pct  TINYINT       DEFAULT 0,
  task_id       VARCHAR(100)  DEFAULT NULL,
  error_msg     TEXT          DEFAULT NULL,
  file_size_mb  FLOAT         DEFAULT 0,
  created_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME      DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (video_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Tabel 2: AnalysisResults ──────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisResults (
  analysis_id          INT           NOT NULL AUTO_INCREMENT,
  video_id             INTEGER       NOT NULL,
  model_name           VARCHAR(50)   DEFAULT 'RF-DETR Base',
  tracking_method      VARCHAR(50)   DEFAULT 'ByteTrack',
  processing_time_sec  FLOAT         DEFAULT 0,
  anotated_video_path  VARCHAR(512)  DEFAULT NULL,
  analysis_date        DATETIME      DEFAULT CURRENT_TIMESTAMP,
  ball_posession_home  FLOAT         DEFAULT 0
                                     COMMENT 'Persentase penguasaan bola Tim 1 (0–100)',
  ball_posession_away  FLOAT         DEFAULT 0
                                     COMMENT 'Persentase penguasaan bola Tim 2 (0–100)',
  total_passes         INTEGER       DEFAULT 0,
  -- Kolom tambahan hasil analisis RF-DETR
  ball_speed_max_kmh   FLOAT         DEFAULT 0,
  ball_speed_avg_kmh   FLOAT         DEFAULT 0,
  player_speed_max_kmh FLOAT         DEFAULT 0,
  total_frames         INTEGER       DEFAULT 0,
  dashboard_path       VARCHAR(512)  DEFAULT NULL,
  heatmap_home_path    VARCHAR(512)  DEFAULT NULL,
  heatmap_away_path    VARCHAR(512)  DEFAULT NULL,
  PRIMARY KEY (analysis_id),
  CONSTRAINT fk_ar_video
    FOREIGN KEY (video_id)
    REFERENCES Videos(video_id)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Tabel 3: AnalysisMetadata ─────────────────────────────────
CREATE TABLE IF NOT EXISTS AnalysisMetadata (
  metadata_id   INT      NOT NULL AUTO_INCREMENT,
  analysis_id   INTEGER  NOT NULL,
  analysis_data JSON     NOT NULL
                         COMMENT 'speed_timeline, top5_players, pass_network, dll.',
  PRIMARY KEY (metadata_id),
  CONSTRAINT fk_am_analysis
    FOREIGN KEY (analysis_id)
    REFERENCES AnalysisResults(analysis_id)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Index ─────────────────────────────────────────────────────
CREATE INDEX idx_videos_status      ON Videos(status);
CREATE INDEX idx_videos_uploaded    ON Videos(uploaded_at);
CREATE INDEX idx_results_video_id   ON AnalysisResults(video_id);
CREATE INDEX idx_metadata_analysis  ON AnalysisMetadata(analysis_id);