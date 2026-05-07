"""
zone_analyzer.py
Analisis zona penguasaan lapangan per tim.

Menghasilkan dua jenis visualisasi:
  1. Grid Zone Control  — lapangan dibagi N×M kotak, tiap kotak diwarnai
                          berdasarkan tim yang paling dominan di sana
  2. Voronoi Control    — tiap titik di lapangan "milik" pemain aktif terdekat,
                          sehingga area per tim terbentuk secara dinamis

Basis data:
  - Posisi pemain (position_adjusted / position) → canvas coordinates
  - Posisi bola (ball_trail_smooth) → canvas coordinates
  Keduanya digabung dengan bobot yang bisa dikonfigurasi.

Output per segmen (overall / per babak):
  - {name}_zone_grid.png
  - {name}_zone_voronoi.png
  - {name}_zone_stats.json

Cara pakai:
    from zone_analyzer import ZoneAnalyzer

    zone = ZoneAnalyzer(canvas_w=960, canvas_h=560)

    # Di dalam loop tactical frames:
    zone.collect(frame_num, tracks, homography_calc, kp, ball_trail_smooth[frame_num])

    # Setelah loop selesai:
    zone.save_all(output_dir="output_analysis", video_name="match")

    # Untuk per babak (opsional):
    zone.save_segment(
        output_dir="output_analysis",
        segment_name="match_babak1",
        frame_start=0,
        frame_end=374,
    )
"""

from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import Voronoi


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────

TEAM_BGR = {
    1: (220,  80,  30),   # biru
    2: ( 30,  80, 220),   # merah
}
TEAM_LABELS = {1: "Tim 1", 2: "Tim 2"}

# Bobot penggabungan posisi pemain vs bola
WEIGHT_PLAYER = 1.0
WEIGHT_BALL   = 2.0   # bola diberi bobot lebih tinggi karena lebih representatif


# ─────────────────────────────────────────────────────────────────────────────
# Helper: gambar lapangan
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pitch(canvas_w: int, canvas_h: int,
                bg: tuple = (30, 80, 30)) -> np.ndarray:
    img = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    lw  = max(1, canvas_w // 200)
    c   = (180, 180, 180)

    cv2.rectangle(img, (2, 2), (canvas_w - 3, canvas_h - 3), c, lw)
    mid = canvas_w // 2
    cv2.line(img, (mid, 2), (mid, canvas_h - 3), c, lw)
    cv2.circle(img, (mid, canvas_h // 2), int(canvas_h * 0.15), c, lw)
    cv2.circle(img, (mid, canvas_h // 2), 3, c, -1)

    pw = int(canvas_w * 0.13); ph = int(canvas_h * 0.40)
    py = (canvas_h - ph) // 2
    cv2.rectangle(img, (2, py), (2 + pw, py + ph), c, lw)
    cv2.rectangle(img, (canvas_w - 3 - pw, py), (canvas_w - 3, py + ph), c, lw)

    gw = int(canvas_w * 0.04); gh = int(canvas_h * 0.18)
    gy = (canvas_h - gh) // 2
    cv2.rectangle(img, (2, gy), (2 + gw, gy + gh), c, lw)
    cv2.rectangle(img, (canvas_w - 3 - gw, gy), (canvas_w - 3, gy + gh), c, lw)

    return img


def _add_bar(img: np.ndarray, text: str, sub: str = "",
             position: str = "top") -> np.ndarray:
    h = 55
    bar = np.full((h, img.shape[1], 3), (18, 18, 18), dtype=np.uint8)
    cv2.putText(bar, text, (18, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (235, 235, 235), 2)
    if sub:
        tw = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(bar, sub, (img.shape[1] - tw - 16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return np.vstack([bar, img]) if position == "top" else np.vstack([img, bar])


def _add_legend(img: np.ndarray,
                items: list[tuple[tuple, str]]) -> np.ndarray:
    h   = 48
    bar = np.full((h, img.shape[1], 3), (18, 18, 18), dtype=np.uint8)
    x   = 18
    for color, label in items:
        cv2.rectangle(bar, (x, 14), (x + 26, 34), color, -1)
        cv2.putText(bar, label, (x + 34, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (215, 215, 215), 1)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
        x += tw + 60
    return np.vstack([img, bar])


# ─────────────────────────────────────────────────────────────────────────────
# FrameRecord — satu record per frame
# ─────────────────────────────────────────────────────────────────────────────

class _FrameRecord:
    __slots__ = ("frame_num", "player_pos", "ball_pos")

    def __init__(self, frame_num: int,
                 player_pos: dict[int, list[tuple[int, int]]],
                 ball_pos: Optional[tuple[int, int]]):
        self.frame_num  = frame_num
        self.player_pos = player_pos   # {team_id: [(x,y), ...]}
        self.ball_pos   = ball_pos     # (x, y) atau None


# ─────────────────────────────────────────────────────────────────────────────
# ZoneAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ZoneAnalyzer:
    """
    Kumpulkan data posisi per frame, lalu generate grid & voronoi zone control.

    Parameters
    ----------
    canvas_w, canvas_h : int    ukuran canvas taktis (px)
    grid_cols, grid_rows: int   jumlah kolom & baris grid zona
    output_scale        : float scale faktor PNG output
    player_weight       : float bobot kontribusi posisi pemain
    ball_weight         : float bobot kontribusi posisi bola
    """

    def __init__(
        self,
        canvas_w:      int   = 960,
        canvas_h:      int   = 560,
        grid_cols:     int   = 6,     # lapangan dibagi 6 kolom × 4 baris = 24 zona
        grid_rows:     int   = 4,
        output_scale:  float = 2.0,
        player_weight: float = WEIGHT_PLAYER,
        ball_weight:   float = WEIGHT_BALL,
    ):
        self.canvas_w      = canvas_w
        self.canvas_h      = canvas_h
        self.grid_cols     = grid_cols
        self.grid_rows     = grid_rows
        self.output_scale  = output_scale
        self.player_weight = player_weight
        self.ball_weight   = ball_weight

        self._records: list[_FrameRecord] = []

    # ─────────────────────────────────────────────────────────────────────
    # Collect
    # ─────────────────────────────────────────────────────────────────────

    def collect(
        self,
        frame_num:      int,
        tracks:         dict,
        homography_calc,
        keypoints:      dict,
        ball_map_pos:   Optional[tuple[int, int]],
    ) -> None:
        """
        Kumpulkan posisi pemain & bola untuk satu frame.
        Panggil di dalam loop tactical map di main.py.
        """
        H           = homography_calc.get_homography(keypoints)
        player_pos: dict[int, list[tuple[int, int]]] = {1: [], 2: []}

        for player_id, track in tracks["player"][frame_num].items():
            team = track.get("team", -1)
            if team not in (1, 2):
                continue
            pos = track.get("position_adjusted") or track.get("position")
            if pos is None:
                continue
            mp = homography_calc.transform_point(
                pos, H,
                canvas_w=self.canvas_w, canvas_h=self.canvas_h, margin=5,
            )
            if mp is not None:
                player_pos[team].append(mp)

        for gk_id, track in tracks["goalkeeper"][frame_num].items():
            team = track.get("team", -1)
            if team not in (1, 2):
                continue
            pos = track.get("position_adjusted") or track.get("position")
            if pos is None:
                continue
            mp = homography_calc.transform_point(
                pos, H,
                canvas_w=self.canvas_w, canvas_h=self.canvas_h, margin=5,
            )
            if mp is not None:
                player_pos[team].append(mp)

        self._records.append(_FrameRecord(frame_num, player_pos, ball_map_pos))

    # ─────────────────────────────────────────────────────────────────────
    # Internal: slice records per segmen
    # ─────────────────────────────────────────────────────────────────────

    def _slice(
        self,
        frame_start: Optional[int] = None,
        frame_end:   Optional[int] = None,
    ) -> list[_FrameRecord]:
        if frame_start is None and frame_end is None:
            return self._records
        return [
            r for r in self._records
            if (frame_start is None or r.frame_num >= frame_start)
            and (frame_end   is None or r.frame_num <= frame_end)
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Internal: bangun density map gabungan per tim
    # ─────────────────────────────────────────────────────────────────────

    def _build_density(
        self,
        records: list[_FrameRecord],
    ) -> dict[int, np.ndarray]:
        """
        Return {team_id: density_map (float32, shape canvas_h×canvas_w)}.
        Posisi pemain diberi bobot player_weight, bola ball_weight.
        """
        dm = {t: np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
              for t in (1, 2)}

        for rec in records:
            # Posisi pemain
            for team, positions in rec.player_pos.items():
                for x, y in positions:
                    if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                        dm[team][y, x] += self.player_weight

            # Posisi bola: dikreditkan ke tim yang "lebih dekat"
            # (sederhana: tim dengan pemain terdekat ke bola)
            if rec.ball_pos is not None:
                bx, by = rec.ball_pos
                if 0 <= bx < self.canvas_w and 0 <= by < self.canvas_h:
                    min_dist = {t: float("inf") for t in (1, 2)}
                    for team, positions in rec.player_pos.items():
                        for px, py in positions:
                            d = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                            if d < min_dist[team]:
                                min_dist[team] = d
                    closest = min(min_dist, key=min_dist.get)
                    dm[closest][by, bx] += self.ball_weight

        # Gaussian blur untuk smooth
        k = 51
        for t in (1, 2):
            dm[t] = cv2.GaussianBlur(dm[t], (k, k), 0)

        return dm

    # ─────────────────────────────────────────────────────────────────────
    # Generate: Grid Zone Control
    # ─────────────────────────────────────────────────────────────────────

    def _generate_grid(
        self,
        records: list[_FrameRecord],
        title:   str = "Grid Zone Control",
    ) -> np.ndarray:
        dm    = self._build_density(records)
        pitch = _draw_pitch(self.canvas_w, self.canvas_h)
        out   = pitch.copy().astype(np.float32)

        cw = self.canvas_w / self.grid_cols
        ch = self.canvas_h / self.grid_rows

        zone_stats = {}   # untuk label persentase di tiap kotak

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x0, y0 = int(col * cw), int(row * ch)
                x1, y1 = int((col + 1) * cw), int((row + 1) * ch)

                s1 = float(dm[1][y0:y1, x0:x1].sum())
                s2 = float(dm[2][y0:y1, x0:x1].sum())
                total = s1 + s2

                if total < 1e-6:
                    continue

                ratio1 = s1 / total
                ratio2 = s2 / total
                winner = 1 if ratio1 >= ratio2 else 2
                dominance = abs(ratio1 - ratio2)   # 0–1

                color = np.array(TEAM_BGR[winner], dtype=np.float32)
                alpha = 0.25 + dominance * 0.50    # 0.25–0.75

                roi = out[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = roi * (1 - alpha) + color * alpha

                zone_stats[(row, col)] = {
                    "winner": winner,
                    "team1_pct": round(ratio1 * 100, 1),
                    "team2_pct": round(ratio2 * 100, 1),
                }

                # Label persentase di tengah kotak
                cx_txt = x0 + int(cw * 0.5)
                cy_txt = y0 + int(ch * 0.5)
                pct    = f"{max(ratio1, ratio2)*100:.0f}%"
                tw, th = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.putText(out.astype(np.uint8), pct,
                            (cx_txt - tw // 2, cy_txt + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)

        # Gambar grid lines di atas
        result = out.astype(np.uint8)
        for col in range(1, self.grid_cols):
            cv2.line(result, (int(col * cw), 0), (int(col * cw), self.canvas_h),
                     (200, 200, 200), 1)
        for row in range(1, self.grid_rows):
            cv2.line(result, (0, int(row * ch)), (self.canvas_w, int(row * ch)),
                     (200, 200, 200), 1)

        # Hitung persentase zona yang dikuasai tiap tim
        t1_zones = sum(1 for v in zone_stats.values() if v["winner"] == 1)
        t2_zones = sum(1 for v in zone_stats.values() if v["winner"] == 2)
        total_zones = max(t1_zones + t2_zones, 1)
        sub = (f"{TEAM_LABELS[1]}: {t1_zones}/{total_zones} zona  |  "
               f"{TEAM_LABELS[2]}: {t2_zones}/{total_zones} zona")

        result = _add_bar(result, title, sub, position="top")
        result = _add_legend(result, [
            (TEAM_BGR[1], f"{TEAM_LABELS[1]} dominan"),
            (TEAM_BGR[2], f"{TEAM_LABELS[2]} dominan"),
            ((100, 100, 100), "Warna lebih gelap = lebih dominan"),
        ])

        if self.output_scale != 1.0:
            w = int(result.shape[1] * self.output_scale)
            h = int(result.shape[0] * self.output_scale)
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Generate: Voronoi Zone Control
    # ─────────────────────────────────────────────────────────────────────

    def _generate_voronoi(
        self,
        records: list[_FrameRecord],
        title:   str = "Voronoi Zone Control",
    ) -> np.ndarray:
        """
        Voronoi berbasis posisi rata-rata pemain per tim selama segmen.
        Tiap titik canvas → milik pemain terdekat → warnai dengan warna timnya.
        """
        # Kumpulkan rata-rata posisi tiap "player slot" per tim
        # Karena track_id bisa tidak konsisten, kita pakai semua posisi
        # lalu cluster menjadi N titik representatif (centroid sederhana)
        all_pos = {1: [], 2: []}
        for rec in records:
            for team, positions in rec.player_pos.items():
                all_pos[team].extend(positions)

        # Jika data sangat sedikit, fallback ke grid
        if len(all_pos[1]) < 3 or len(all_pos[2]) < 3:
            return self._generate_grid(records, title="Voronoi (fallback ke Grid — data kurang)")

        # Ambil centroid dengan k-means sederhana (11 titik per tim = 11 pemain)
        def get_centroids(pts: list[tuple], k: int = 11) -> np.ndarray:
            arr = np.array(pts, dtype=np.float32)
            k   = min(k, len(arr))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, _, centers = cv2.kmeans(
                arr, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
            return centers

        c1 = get_centroids(all_pos[1], k=11)
        c2 = get_centroids(all_pos[2], k=11)

        # Semua centroid + label tim
        all_centers = np.vstack([c1, c2])
        all_teams   = [1] * len(c1) + [2] * len(c2)

        # Buat pixel grid canvas → cari centroid terdekat
        ys, xs = np.mgrid[0:self.canvas_h, 0:self.canvas_w]
        pts_grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

        # Brute-force nearest centroid (cukup cepat untuk 960×560)
        diffs  = pts_grid[:, None, :] - all_centers[None, :, :]   # (N, K, 2)
        dists  = (diffs ** 2).sum(axis=2)                          # (N, K)
        nearest = dists.argmin(axis=1)                              # (N,)

        team_map = np.array([all_teams[i] for i in nearest]).reshape(
            self.canvas_h, self.canvas_w
        )

        # Render
        pitch  = _draw_pitch(self.canvas_w, self.canvas_h)
        result = pitch.copy().astype(np.float32)

        for team in (1, 2):
            mask  = (team_map == team).astype(np.float32)
            color = np.array(TEAM_BGR[team], dtype=np.float32)
            for ch in range(3):
                result[:, :, ch] += mask * color[ch] * 0.45

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Gambar centroid sebagai titik
        for i, (cx, cy) in enumerate(all_centers):
            team  = all_teams[i]
            color = TEAM_BGR[team]
            cv2.circle(result, (int(cx), int(cy)), 6,  color, -1)
            cv2.circle(result, (int(cx), int(cy)), 7,  (255, 255, 255), 1)

        # Overlay garis lapangan
        lines = _draw_pitch(self.canvas_w, self.canvas_h)
        lmask = (lines > 60).any(axis=2)
        result[lmask] = (result[lmask].astype(np.float32) * 0.4 +
                         lines[lmask].astype(np.float32) * 0.6).astype(np.uint8)

        result = _add_bar(result, title,
                          f"Berbasis rata-rata posisi {len(c1)}+{len(c2)} centroid pemain",
                          position="top")
        result = _add_legend(result, [
            (TEAM_BGR[1], f"{TEAM_LABELS[1]} area"),
            (TEAM_BGR[2], f"{TEAM_LABELS[2]} area"),
            ((255, 255, 255), "Centroid pemain"),
        ])

        if self.output_scale != 1.0:
            w = int(result.shape[1] * self.output_scale)
            h = int(result.shape[0] * self.output_scale)
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Stats JSON
    # ─────────────────────────────────────────────────────────────────────

    def _compute_stats(
        self,
        records: list[_FrameRecord],
    ) -> dict:
        dm     = self._build_density(records)
        total  = dm[1] + dm[2] + 1e-9

        # Pixel-level dominance
        t1_px  = int((dm[1] > dm[2]).sum())
        t2_px  = int((dm[2] > dm[1]).sum())
        tot_px = self.canvas_w * self.canvas_h

        # Grid-level
        cw = self.canvas_w / self.grid_cols
        ch = self.canvas_h / self.grid_rows
        zone_results = []
        t1_zones = t2_zones = 0

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x0, y0 = int(col * cw), int(row * ch)
                x1, y1 = int((col + 1) * cw), int((row + 1) * ch)
                s1 = float(dm[1][y0:y1, x0:x1].sum())
                s2 = float(dm[2][y0:y1, x0:x1].sum())
                tot = s1 + s2
                if tot < 1e-6:
                    continue
                winner = 1 if s1 >= s2 else 2
                if winner == 1:
                    t1_zones += 1
                else:
                    t2_zones += 1
                zone_results.append({
                    "row": row, "col": col,
                    "winner": winner,
                    "team1_pct": round(s1 / tot * 100, 1),
                    "team2_pct": round(s2 / tot * 100, 1),
                })

        total_zones = max(t1_zones + t2_zones, 1)

        # Ball possession berbasis density
        ball_frames = sum(1 for r in records if r.ball_pos is not None)

        return {
            "frames_analyzed": len(records),
            "ball_visible_frames": ball_frames,
            "pixel_dominance": {
                "team1_pct": round(t1_px / tot_px * 100, 1),
                "team2_pct": round(t2_px / tot_px * 100, 1),
            },
            "grid_zone_dominance": {
                "team1_zones": t1_zones,
                "team2_zones": t2_zones,
                "total_zones": total_zones,
                "team1_pct": round(t1_zones / total_zones * 100, 1),
                "team2_pct": round(t2_zones / total_zones * 100, 1),
            },
            "zone_detail": zone_results,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Public: save semua output untuk satu segmen
    # ─────────────────────────────────────────────────────────────────────

    def _save_segment_internal(
        self,
        records:      list[_FrameRecord],
        output_dir:   str,
        base_name:    str,
        title_suffix: str = "",
        fmt:          str = "png",
        jpg_quality:  int = 95,
    ) -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if fmt == "jpg" else []

        title_g = f"Grid Zone Control{title_suffix}"
        title_v = f"Voronoi Zone Control{title_suffix}"

        imgs = {
            "zone_grid":    (f"{base_name}_zone_grid.{fmt}",    self._generate_grid(records, title_g)),
            "zone_voronoi": (f"{base_name}_zone_voronoi.{fmt}", self._generate_voronoi(records, title_v)),
        }

        saved = {}
        for key, (path, img) in imgs.items():
            cv2.imwrite(path, img, params)
            print(f"[ZoneAnalyzer] Saved → {path}")
            saved[key] = path

        stats      = self._compute_stats(records)
        stats_path = f"{base_name}_zone_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[ZoneAnalyzer] Saved → {stats_path}")
        saved["zone_stats"] = stats_path

        return saved

    def save_all(
        self,
        output_dir:  str = "output_analysis",
        video_name:  str = "match",
        fmt:         str = "png",
        jpg_quality: int = 95,
    ) -> dict[str, str]:
        """Simpan output untuk keseluruhan video."""
        base = os.path.join(output_dir, video_name)
        return self._save_segment_internal(
            self._records, output_dir, base,
            title_suffix="", fmt=fmt, jpg_quality=jpg_quality,
        )

    def save_segment(
        self,
        output_dir:    str,
        segment_name:  str,
        frame_start:   Optional[int] = None,
        frame_end:     Optional[int] = None,
        fmt:           str = "png",
        jpg_quality:   int = 95,
    ) -> dict[str, str]:
        """
        Simpan output untuk segmen tertentu (misal per babak).

        Parameters
        ----------
        segment_name : str   nama file output, misal "match_babak1"
        frame_start  : int   frame pertama segmen (inklusif), None = dari awal
        frame_end    : int   frame terakhir segmen (inklusif), None = sampai akhir
        """
        records = self._slice(frame_start, frame_end)
        if not records:
            print(f"[ZoneAnalyzer] WARNING: Tidak ada data untuk segmen "
                  f"{segment_name} (frame {frame_start}–{frame_end})")
            return {}

        label = ""
        if frame_start is not None or frame_end is not None:
            fs = frame_start if frame_start is not None else 0
            fe = frame_end   if frame_end   is not None else self._records[-1].frame_num
            label = f" (frame {fs}–{fe})"

        base = os.path.join(output_dir, segment_name)
        return self._save_segment_internal(
            records, output_dir, base,
            title_suffix=label, fmt=fmt, jpg_quality=jpg_quality,
        )