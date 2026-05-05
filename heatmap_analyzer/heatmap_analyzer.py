"""
heatmap_analyzer.py
Menghasilkan heatmap posisi pemain per tim dan zone control
sebagai file PNG/JPG terpisah (bukan overlay video).

Output:
  - heatmap_team1.png       — heatmap posisi tim 1
  - heatmap_team2.png       — heatmap posisi tim 2
  - heatmap_combined.png    — kedua tim dalam satu canvas
  - zone_control.png        — peta zona dominasi per tim
  - summary_stats.json      — data mentah untuk ditampilkan di UI

Cara pakai di main.py:
    from heatmap_analyzer import HeatmapAnalyzer

    analyzer = HeatmapAnalyzer(canvas_w=960, canvas_h=560)

    # Di dalam loop render tactical map:
    analyzer.collect(frame_num, tracks, homography_calc, all_kp[frame_num], team_ball_control)

    # Setelah semua frame selesai:
    analyzer.save_all(output_dir="output_analysis", video_name="08fd33_4")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta warna tim (BGR untuk OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

TEAM_COLORS = {
    1: (220, 80,  30),   # Tim 1 — biru
    2: (30,  80, 220),   # Tim 2 — merah
}

TEAM_LABELS = {
    1: "Tim 1",
    2: "Tim 2",
}

# Warna untuk colormap heatmap per tim
# Tim 1: putih → biru, Tim 2: putih → merah
CMAP_TEAM = {
    1: cv2.COLORMAP_COOL,
    2: cv2.COLORMAP_HOT,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pitch background (gambar lapangan sederhana)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pitch_background(canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    Gambar lapangan sepak bola sederhana sebagai background.
    Warna hijau gelap, garis putih.
    """
    img = np.full((canvas_h, canvas_w, 3), (34, 85, 34), dtype=np.uint8)

    lw = max(1, canvas_w // 200)   # line width proporsional
    c  = (200, 200, 200)           # warna garis

    # Border lapangan
    cv2.rectangle(img, (2, 2), (canvas_w - 3, canvas_h - 3), c, lw)

    # Garis tengah
    mid_x = canvas_w // 2
    cv2.line(img, (mid_x, 2), (mid_x, canvas_h - 3), c, lw)

    # Lingkaran tengah
    r_center = int(canvas_h * 0.15)
    cv2.circle(img, (mid_x, canvas_h // 2), r_center, c, lw)
    cv2.circle(img, (mid_x, canvas_h // 2), 3, c, -1)

    # Kotak penalti kiri & kanan (proporsi FIFA)
    pen_w = int(canvas_w * 0.13)
    pen_h = int(canvas_h * 0.40)
    py    = (canvas_h - pen_h) // 2

    # Kiri
    cv2.rectangle(img, (2, py), (2 + pen_w, py + pen_h), c, lw)
    # Kanan
    cv2.rectangle(img, (canvas_w - 3 - pen_w, py),
                  (canvas_w - 3, py + pen_h), c, lw)

    # Kotak gawang kiri & kanan
    goal_w = int(canvas_w * 0.04)
    goal_h = int(canvas_h * 0.18)
    gy     = (canvas_h - goal_h) // 2
    cv2.rectangle(img, (2, gy), (2 + goal_w, gy + goal_h), c, lw)
    cv2.rectangle(img, (canvas_w - 3 - goal_w, gy),
                  (canvas_w - 3, gy + goal_h), c, lw)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# HeatmapAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapAnalyzer:
    """
    Kumpulkan posisi pemain per frame, lalu generate berbagai output analisis.

    Parameters
    ----------
    canvas_w, canvas_h : int
        Ukuran canvas tactical map (harus sama dengan TacticalMapRenderer).
    gaussian_radius : int
        Radius blur untuk heatmap. Makin besar → heatmap makin "menyebar".
    output_scale : float
        Scale faktor output PNG (1.0 = canvas size, 2.0 = 2× lebih besar).
    """

    def __init__(
        self,
        canvas_w: int        = 960,
        canvas_h: int        = 560,
        gaussian_radius: int = 25,
        output_scale: float  = 2.0,
    ):
        self.canvas_w        = canvas_w
        self.canvas_h        = canvas_h
        self.gaussian_radius = gaussian_radius
        self.output_scale    = output_scale

        # Akumulator posisi: team_id → list of (x, y)
        self._positions: dict[int, list[tuple[int, int]]] = {1: [], 2: []}

        # Untuk zone control: posisi per team per frame (dipakai voting per pixel)
        self._frame_positions: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}

        # Statistik penguasaan bola
        self._possession_frames = {1: 0, 2: 0, -1: 0}

        self._total_frames = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Collect — dipanggil tiap frame dari main.py
    # ─────────────────────────────────────────────────────────────────────────

    def collect(
        self,
        frame_num:       int,
        tracks:          dict,
        homography_calc,               # HomographyCalculator instance
        keypoints:       dict,         # all_kp[frame_num]
        team_ball_control: np.ndarray, # array hasil TeamBallControl
    ) -> None:
        """
        Kumpulkan posisi semua pemain (sudah di-transform ke canvas) untuk frame ini.
        Panggil ini di dalam loop utama setelah transform_point tersedia.
        """
        self._total_frames += 1

        # Penguasaan bola frame ini
        if frame_num < len(team_ball_control):
            controlling_team = int(team_ball_control[frame_num])
            self._possession_frames[controlling_team] = \
                self._possession_frames.get(controlling_team, 0) + 1

        # Dapatkan H
        H = homography_calc.get_homography(keypoints)

        frame_pos: dict[int, list[tuple[int, int]]] = {1: [], 2: []}

        for player_id, track in tracks["player"][frame_num].items():
            team = track.get("team", -1)
            if team not in (1, 2):
                continue

            pos = track.get("position_adjusted") or track.get("position")
            if pos is None:
                continue

            map_pos = homography_calc.transform_point(
                pos, H,
                canvas_w=self.canvas_w,
                canvas_h=self.canvas_h,
                margin=5,
            )
            if map_pos is None:
                continue

            self._positions[team].append(map_pos)
            frame_pos[team].append(map_pos)

        # Simpan posisi per frame untuk zone control
        self._frame_positions[1].append(frame_pos[1])
        self._frame_positions[2].append(frame_pos[2])

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_density_map(self, positions: list[tuple[int, int]]) -> np.ndarray:
        """Buat density map (float32) dari list posisi."""
        density = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        for x, y in positions:
            if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                density[y, x] += 1.0

        # Gaussian blur untuk smooth
        k = self.gaussian_radius * 2 + 1
        density = cv2.GaussianBlur(density, (k, k), 0)
        return density

    def _normalize(self, density: np.ndarray) -> np.ndarray:
        """Normalize ke 0–255."""
        if density.max() == 0:
            return np.zeros_like(density, dtype=np.uint8)
        return ((density / density.max()) * 255).astype(np.uint8)

    def _overlay_heatmap(
        self,
        base: np.ndarray,
        density: np.ndarray,
        colormap: int,
        alpha: float = 0.65,
    ) -> np.ndarray:
        """Overlay heatmap (density) di atas base image dengan alpha blending."""
        norm    = self._normalize(density)
        colored = cv2.applyColorMap(norm, colormap)

        # Mask: hanya blend di area yang ada data (density > threshold)
        mask = (norm > 10).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)[:, :, np.newaxis]

        result = base.astype(np.float32)
        result = result * (1 - alpha * mask) + colored.astype(np.float32) * alpha * mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_title_bar(
        self,
        img: np.ndarray,
        title: str,
        subtitle: str = "",
        bg_color: tuple = (20, 20, 20),
    ) -> np.ndarray:
        """Tambahkan title bar di atas gambar."""
        bar_h   = 60
        bar     = np.full((bar_h, img.shape[1], 3), bg_color, dtype=np.uint8)

        cv2.putText(bar, title, (20, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (240, 240, 240), 2)

        if subtitle:
            tw = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(bar, subtitle, (img.shape[1] - tw - 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        return np.vstack([bar, img])

    def _add_legend(
        self,
        img: np.ndarray,
        items: list[tuple[tuple, str]],   # [(bgr_color, label), ...]
    ) -> np.ndarray:
        """Tambahkan legend bar di bawah gambar."""
        bar_h = 50
        bar   = np.full((bar_h, img.shape[1], 3), (20, 20, 20), dtype=np.uint8)

        x = 20
        for color, label in items:
            cv2.rectangle(bar, (x, 15), (x + 28, 35), color, -1)
            cv2.putText(bar, label, (x + 36, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 70

        return np.vstack([img, bar])

    def _scale_output(self, img: np.ndarray) -> np.ndarray:
        if self.output_scale == 1.0:
            return img
        w = int(img.shape[1] * self.output_scale)
        h = int(img.shape[0] * self.output_scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # ─────────────────────────────────────────────────────────────────────────
    # Generate individual heatmaps
    # ─────────────────────────────────────────────────────────────────────────

    def generate_team_heatmap(self, team_id: int) -> np.ndarray:
        """Generate heatmap untuk satu tim. Return BGR image."""
        pitch  = draw_pitch_background(self.canvas_w, self.canvas_h)
        density = self._build_density_map(self._positions[team_id])
        result  = self._overlay_heatmap(pitch, density, CMAP_TEAM[team_id])

        label    = TEAM_LABELS[team_id]
        n_frames = sum(1 for p in self._positions[team_id] if p)
        subtitle = f"{len(self._positions[team_id])} data points"

        result = self._add_title_bar(result, f"Heatmap Posisi — {label}", subtitle)
        result = self._add_legend(result, [(TEAM_COLORS[team_id], label)])
        return self._scale_output(result)

    def generate_combined_heatmap(self) -> np.ndarray:
        """Kedua tim dalam satu canvas, alpha blending."""
        pitch    = draw_pitch_background(self.canvas_w, self.canvas_h)
        density1 = self._build_density_map(self._positions[1])
        density2 = self._build_density_map(self._positions[2])

        result = self._overlay_heatmap(pitch,   density1, CMAP_TEAM[1], alpha=0.55)
        result = self._overlay_heatmap(result,  density2, CMAP_TEAM[2], alpha=0.55)

        result = self._add_title_bar(result, "Heatmap Posisi — Kedua Tim",
                                     f"{self._total_frames} frames dianalisis")
        result = self._add_legend(result, [
            (TEAM_COLORS[1], TEAM_LABELS[1]),
            (TEAM_COLORS[2], TEAM_LABELS[2]),
        ])
        return self._scale_output(result)

    def generate_zone_control(self) -> np.ndarray:
        """
        Peta zona dominasi per tim.
        Tiap piksel canvas → tim mana yang rata-rata lebih dekat ke sana.
        Menggunakan voting berbasis density map.
        """
        density1 = self._build_density_map(self._positions[1])
        density2 = self._build_density_map(self._positions[2])

        pitch  = draw_pitch_background(self.canvas_w, self.canvas_h)
        result = pitch.copy().astype(np.float32)

        total  = density1 + density2 + 1e-6
        ratio1 = density1 / total   # 0..1, makin tinggi makin dominan tim 1
        ratio2 = density2 / total

        # Warna zona: biru untuk tim 1, merah untuk tim 2
        # Alpha proporsional dengan selisih dominasi
        dominance = np.abs(ratio1 - ratio2)   # 0 = seimbang, 1 = total dominasi

        mask1 = (ratio1 > ratio2).astype(np.float32) * dominance * 0.7
        mask2 = (ratio2 > ratio1).astype(np.float32) * dominance * 0.7

        c1 = np.array(TEAM_COLORS[1], dtype=np.float32)
        c2 = np.array(TEAM_COLORS[2], dtype=np.float32)

        for ch in range(3):
            result[:, :, ch] += mask1 * c1[ch]
            result[:, :, ch] += mask2 * c2[ch]

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Overlay garis lapangan lagi agar tetap terlihat
        lines = draw_pitch_background(self.canvas_w, self.canvas_h)
        line_mask = (lines > 60).any(axis=2)   # area garis
        result[line_mask] = lines[line_mask]

        # Tambah colorbar legend
        result = self._add_title_bar(result, "Zone Control per Tim",
                                     "Intensitas warna = tingkat dominasi")
        result = self._add_legend(result, [
            (TEAM_COLORS[1], f"{TEAM_LABELS[1]} dominan"),
            (TEAM_COLORS[2], f"{TEAM_LABELS[2]} dominan"),
            ((100, 100, 100), "Seimbang"),
        ])
        return self._scale_output(result)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary stats
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary_stats(self) -> dict:
        """
        Hitung ringkasan statistik untuk ditampilkan di UI atau disimpan ke DB.
        """
        total_poss = sum(self._possession_frames.values())
        if total_poss == 0:
            total_poss = 1

        def pct(v): return round(v / total_poss * 100, 1)

        # Zone coverage: berapa % area canvas yang dikuasai tiap tim
        d1 = self._build_density_map(self._positions[1])
        d2 = self._build_density_map(self._positions[2])
        covered1 = int((d1 > 0.5).sum())
        covered2 = int((d2 > 0.5).sum())
        total_px  = self.canvas_w * self.canvas_h

        return {
            "total_frames_analyzed": self._total_frames,
            "ball_possession": {
                "team1_pct": pct(self._possession_frames.get(1, 0)),
                "team2_pct": pct(self._possession_frames.get(2, 0)),
                "no_possession_pct": pct(self._possession_frames.get(-1, 0)),
            },
            "zone_coverage": {
                "team1_pct": round(covered1 / total_px * 100, 1),
                "team2_pct": round(covered2 / total_px * 100, 1),
            },
            "position_datapoints": {
                "team1": len(self._positions[1]),
                "team2": len(self._positions[2]),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Save all outputs
    # ─────────────────────────────────────────────────────────────────────────

    def save_all(
        self,
        output_dir: str  = "output_analysis",
        video_name: str  = "match",
        fmt: str         = "png",          # "png" atau "jpg"
        jpg_quality: int = 95,
    ) -> dict[str, str]:
        """
        Generate dan simpan semua output ke output_dir.
        Return dict {nama: path} untuk disimpan ke database.
        """
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, video_name)

        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if fmt == "jpg" else []

        outputs = {
            "heatmap_team1":    (f"{base}_heatmap_team1.{fmt}",    self.generate_team_heatmap(1)),
            "heatmap_team2":    (f"{base}_heatmap_team2.{fmt}",    self.generate_team_heatmap(2)),
            "heatmap_combined": (f"{base}_heatmap_combined.{fmt}", self.generate_combined_heatmap()),
            "zone_control":     (f"{base}_zone_control.{fmt}",     self.generate_zone_control()),
        }

        saved_paths = {}
        for key, (path, img) in outputs.items():
            cv2.imwrite(path, img, params)
            print(f"[HeatmapAnalyzer] Saved → {path}")
            saved_paths[key] = path

        # Simpan summary stats sebagai JSON
        stats       = self.get_summary_stats()
        stats_path  = f"{base}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[HeatmapAnalyzer] Saved → {stats_path}")
        saved_paths["stats"] = stats_path

        return saved_paths