"""
Interactive slideshow + manual brush editor for LV cavity segmentations.

WHY THIS EXISTS
    Neither extract_lv_volume.py nor extract_lv_volume_conical.py let you
    touch mask voxels by hand. extract_lv_volume_conical.py's
    --manual-review only lets you drag-correct the valve-plane CUTOFF LINE
    before the cone is built — the resulting mask itself is never editable.
    This script runs the v2 (conical + valve-plane) segmentation from
    extract_lv_volume_conical.py exactly as-is, then opens a single
    interactive window with two things bolted together:
        1. A phase slider (+ Play button) that scrubs/animates through every
           cardiac-phase frame — the "slideshow".
        2. A paint/erase brush, usable on any of the three orthogonal
           panels, that edits the CURRENT frame's mask voxel-by-voxel.
    Edits persist per-frame as you scrub back and forth, and are written
    out on demand.

    This file does not modify extract_lv_volume.py or
    extract_lv_volume_conical.py — it only imports and calls them.

SEGMENTATION CACHING
    The per-frame call into segment_lv_cavity_conical() is the slow part
    of startup (it re-does the flood fill / cone construction for every
    frame, every time). To avoid paying that cost on every launch, the raw
    (pre-edit, pre-temporal-smoothing) masks are cached to
    <output>/segmentation_cache.nii.gz + segmentation_cache.json the first
    time they're computed. On the next run, if a cache is found for the
    same --folders, you're asked (in the terminal, before the matplotlib
    window opens) whether to load it instead of re-segmenting. The cache
    records the segmentation params used (seed, threshold, cone/sphere
    geometry) so you can see at a glance whether it still matches what
    you're about to run.

Usage:
    python mask_editor.py --seed 260 280 144

    Match the tuning of a previous extract_lv_volume_conical.py run so the
    starting (pre-edit) masks are identical to what you already reviewed:
        python mask_editor.py --seed 260 280 144 \\
            --sphere-radius 50 --apex-radius 15 --cone-length 90

    Force a full re-segmentation even if a matching cache exists:
        python mask_editor.py --seed 260 280 144 --no-cache

Controls (once the window opens):
    - "Cardiac phase" slider / "Play" button: scrub or animate through frames.
    - Axial/Coronal/Sagittal sliders: move the displayed slice on each panel.
    - Add/Erase radio + brush radius slider: left-click-drag on ANY panel to
      paint or erase a disk (in that panel's plane only) on the CURRENT
      frame's mask.
    - "Reset Frame": discard edits on the current frame, back to the
      auto-segmented mask.
    - "Save & Exit": write the edited 4D mask + a before/after volume CSV
      and plot to --output, then close.

Output:
    <output>/segmentation_cache.nii.gz      - raw (pre-edit) v2 masks, stacked
    <output>/segmentation_cache.json        - params + per-frame metadata for the cache
    <output>/segmentation_4d_edited.nii.gz  - edited masks, stacked (x,y,z,t) [on Save & Exit]
    <output>/volumes_edited.csv             - auto vs. edited volume per frame [on Save & Exit]
    <output>/volume_edited_plot.png         - auto vs. edited volume over time [on Save & Exit]
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("TkAgg")  # this tool is interactive end-to-end; this venv's
                          # Python has no working _tkinter, so TkAgg (what
                          # extract_lv_volume.py/extract_lv_volume_conical.py
                          # use) fails here — MacOSX is the native backend
                          # that actually renders on this machine.
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import ndimage

from extract_lv_volume import (
    load_nifti, collect_frames, select_seed_interactive, estimate_threshold,
    compute_volume_ml, temporal_smooth_masks, save_4d_nifti,
)
from extract_lv_volume_conical import segment_lv_cavity_conical


# ──────────────────────────────────────────────────────────────────────────────
# Brush painting
# ──────────────────────────────────────────────────────────────────────────────

# Each panel fixes one voxel axis and displays the other two. Given the
# data[...].T + imshow(origin="lower") convention used throughout this
# codebase, the plotted x-axis always corresponds to the smaller-numbered
# of the two remaining voxel axes, and the plotted y-axis to the larger one.
PANELS = [
    {"axis_fixed": 2, "a_axis": 0, "b_axis": 1, "title": "Axial"},     # fixed z, vary x,y
    {"axis_fixed": 1, "a_axis": 0, "b_axis": 2, "title": "Coronal"},   # fixed y, vary x,z
    {"axis_fixed": 0, "a_axis": 1, "b_axis": 2, "title": "Sagittal"},  # fixed x, vary y,z
]


def _plane_view(mask, axis_fixed, fixed_idx):
    """A writable 2D view into `mask` at `fixed_idx` along `axis_fixed`."""
    if axis_fixed == 0:
        return mask[fixed_idx, :, :]
    if axis_fixed == 1:
        return mask[:, fixed_idx, :]
    return mask[:, :, fixed_idx]


def paint_disk(mask, axis_fixed, fixed_idx, a_idx, b_idx, va, vb, radius_mm, add):
    """Paint (add=True) or erase (add=False) a disk of `radius_mm` centered
    at (a_idx, b_idx) within the single slice `mask` takes at `fixed_idx`
    along `axis_fixed`. Edits only that one slice — a 2D brush, not a 3D ball.
    """
    plane = _plane_view(mask, axis_fixed, fixed_idx)
    na, nb = plane.shape
    ra = int(np.ceil(radius_mm / va))
    rb = int(np.ceil(radius_mm / vb))
    a_lo, a_hi = max(0, a_idx - ra), min(na - 1, a_idx + ra)
    b_lo, b_hi = max(0, b_idx - rb), min(nb - 1, b_idx + rb)
    if a_lo > a_hi or b_lo > b_hi:
        return
    aa = np.arange(a_lo, a_hi + 1)[:, None]
    bb = np.arange(b_lo, b_hi + 1)[None, :]
    dist2 = ((aa - a_idx) * va) ** 2 + ((bb - b_idx) * vb) ** 2
    sel = dist2 <= radius_mm ** 2
    sub = plane[a_lo:a_hi + 1, b_lo:b_hi + 1]
    sub[sel] = 1 if add else 0


def _overlay_rgba(mask_plane, color=(0.0, 1.0, 1.0), alpha=0.4):
    overlay = np.zeros((*mask_plane.shape, 4))
    overlay[mask_plane.astype(bool)] = (*color, alpha)
    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation caching
# ──────────────────────────────────────────────────────────────────────────────
#
# The expensive step is segment_lv_cavity_conical() being called once per
# frame. Everything downstream of that (temporal smoothing, manual editing,
# display) is cheap. So we cache exactly the raw, pre-smoothing, pre-edit
# masks — plus enough metadata to reconstruct `all_data`/`meta`/voxel info
# without re-segmenting.

CACHE_MASK_NAME = "segmentation_cache.nii.gz"
CACHE_META_NAME = "segmentation_cache.json"


def _cache_paths(output_dir):
    return (os.path.join(output_dir, CACHE_MASK_NAME),
            os.path.join(output_dir, CACHE_META_NAME))


def save_segmentation_cache(raw_masks, meta, filepaths, voxel_sizes_list,
                             voxel_vol_list, affine, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask_path, json_path = _cache_paths(output_dir)

    save_4d_nifti(raw_masks, affine, mask_path)

    cache = {
        "params": params,
        "meta": [list(m) for m in meta],
        "filepaths": list(filepaths),
        "voxel_sizes": [list(v) for v in voxel_sizes_list],
        "voxel_vol": list(voxel_vol_list),
        "affine": np.asarray(affine).tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Segmentation cache saved: {mask_path}")


def find_segmentation_cache(output_dir, folders):
    """Return the parsed cache dict + mask path if a cache exists for these
    folders, else None. Does not check segmentation params — that's shown
    to the user separately so they can decide whether it still applies.
    """
    mask_path, json_path = _cache_paths(output_dir)
    if not (os.path.exists(mask_path) and os.path.exists(json_path)):
        return None
    with open(json_path) as f:
        cache = json.load(f)
    if sorted(cache.get("params", {}).get("folders", [])) != sorted(folders):
        print(f"Found a segmentation cache in '{output_dir}', but it was built "
              f"from different --folders than requested — ignoring it.")
        return None
    return cache, mask_path


def load_segmentation_cache(cache, mask_path):
    """Reconstruct raw_masks, meta, voxel_sizes_list, voxel_vol_list, affine,
    and all_data from a cache dict (as returned by find_segmentation_cache).
    """
    print(f"Loading cached segmentation: {mask_path}")
    masks_4d = nib.load(mask_path).get_fdata()
    n_frames = masks_4d.shape[-1]
    raw_masks = [masks_4d[..., i].astype(np.uint8) for i in range(n_frames)]

    meta = [tuple(m) for m in cache["meta"]]
    filepaths = cache["filepaths"]
    voxel_sizes_list = [tuple(v) for v in cache["voxel_sizes"]]
    voxel_vol_list = list(cache["voxel_vol"])
    affine = np.array(cache["affine"])

    print(f"Loading image intensity data for {len(filepaths)} frames...")
    all_data = []
    for filepath in filepaths:
        data, _, _, _ = load_nifti(filepath)
        all_data.append(data)

    return all_data, raw_masks, meta, voxel_sizes_list, voxel_vol_list, affine


def _describe_params(params):
    return (f"seed={params.get('seed')}, threshold={params.get('threshold')}, "
            f"sphere_radius={params.get('sphere_radius')}, "
            f"apex_radius={params.get('apex_radius')}, "
            f"cone_length={params.get('cone_length')}, "
            f"apex_sphere_radius={params.get('apex_sphere_radius')}, "
            f"valve_search_mm={params.get('valve_search_mm')}, "
            f"axis={params.get('axis')}, apex_dir={params.get('apex_dir')}")


def maybe_prompt_for_cache(args):
    """If a matching cache exists and --no-cache wasn't passed, ask the user
    (in the terminal) whether to load it. Returns the loaded tuple, or None
    if the caller should run the full segmentation instead.
    """
    if args.no_cache:
        return None
    found = find_segmentation_cache(args.output, args.folders)
    if found is None:
        return None
    cache, mask_path = found

    print("\nFound an existing segmentation cache:")
    print(f"  {mask_path}")
    print(f"  {_describe_params(cache['params'])}")
    print(f"  {len(cache['meta'])} frames")
    answer = input("Load this cached segmentation instead of re-running it? [y/N] ").strip().lower()
    if answer != "y":
        return None
    return load_segmentation_cache(cache, mask_path)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive editor
# ──────────────────────────────────────────────────────────────────────────────

class MaskEditor:
    def __init__(self, all_data, masks_auto, voxel_sizes_list, voxel_vol_list,
                 meta, brush_mm, fps, vmin=-200, vmax=600):
        self.all_data = all_data
        self.masks_auto = [m.copy() for m in masks_auto]
        self.masks = [m.copy() for m in masks_auto]
        self.voxel_sizes_list = voxel_sizes_list
        self.voxel_vol_list = voxel_vol_list
        self.meta = meta  # list of (folder, frame_idx, filename)
        self.n = len(all_data)
        self.edited = [False] * self.n
        self.vmin, self.vmax = vmin, vmax

        self.t = 0
        shape = all_data[0].shape
        c = self._centroid_or_center(self.masks[0], shape)
        self.x, self.y, self.z = c

        self.brush_mm = brush_mm
        self.mode = "Add"
        self.painting = False
        self.paint_panel = None
        self.playing = False

        self._build_figure()
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._advance_phase)

        self.draw()
        plt.show()

    @staticmethod
    def _centroid_or_center(mask, shape):
        if mask.sum() > 0:
            return tuple(int(round(c)) for c in ndimage.center_of_mass(mask))
        return (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    # ── figure / widgets ────────────────────────────────────────────────
    def _build_figure(self):
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 8.5))
        plt.subplots_adjust(bottom=0.36, top=0.90)

        self.ax_phase = plt.axes([0.10, 0.28, 0.55, 0.03])
        self.sl_phase = Slider(self.ax_phase, "Cardiac phase", 0, self.n - 1,
                                valinit=0, valstep=1)

        self.ax_play = plt.axes([0.68, 0.275, 0.09, 0.045])
        self.btn_play = Button(self.ax_play, "Play")

        self.ax_z = plt.axes([0.10, 0.21, 0.55, 0.03])
        self.sl_z = Slider(self.ax_z, "Axial (z)", 0, self.all_data[0].shape[2] - 1,
                            valinit=self.z, valstep=1)
        self.ax_y = plt.axes([0.10, 0.16, 0.55, 0.03])
        self.sl_y = Slider(self.ax_y, "Coronal (y)", 0, self.all_data[0].shape[1] - 1,
                            valinit=self.y, valstep=1)
        self.ax_x = plt.axes([0.10, 0.11, 0.55, 0.03])
        self.sl_x = Slider(self.ax_x, "Sagittal (x)", 0, self.all_data[0].shape[0] - 1,
                            valinit=self.x, valstep=1)

        self.ax_brush = plt.axes([0.10, 0.05, 0.55, 0.03])
        self.sl_brush = Slider(self.ax_brush, "Brush radius (mm)", 1, 25,
                                valinit=self.brush_mm, valstep=0.5)

        self.ax_radio = plt.axes([0.80, 0.03, 0.17, 0.14])
        self.radio_mode = RadioButtons(self.ax_radio, ["Add", "Erase"])

        self.ax_reset = plt.axes([0.80, 0.19, 0.17, 0.045])
        self.btn_reset = Button(self.ax_reset, "Reset Frame")

        self.ax_save = plt.axes([0.80, 0.24, 0.17, 0.045])
        self.btn_save = Button(self.ax_save, "Save && Exit")

        self.sl_phase.on_changed(self._on_phase)
        self.sl_z.on_changed(self._on_slice)
        self.sl_y.on_changed(self._on_slice)
        self.sl_x.on_changed(self._on_slice)
        self.sl_brush.on_changed(self._on_brush)
        self.radio_mode.on_clicked(self._on_mode)
        self.btn_play.on_clicked(self._on_play)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ── drawing ─────────────────────────────────────────────────────────
    def draw(self):
        data = self.all_data[self.t]
        mask = self.masks[self.t]
        vol = compute_volume_ml(mask, self.voxel_vol_list[self.t])
        folder, _, filename = self.meta[self.t]
        flag = " [EDITED]" if self.edited[self.t] else ""
        self.fig.suptitle(
            f"{folder}/{filename}  —  volume = {vol:.2f} mL{flag}   "
            f"(mode: {self.mode}, brush: {self.brush_mm:.1f} mm)",
            fontsize=12,
        )

        idx_along = {2: self.z, 1: self.y, 0: self.x}
        for ax, cfg in zip(self.axes, PANELS):
            ax.clear()
            f = cfg["axis_fixed"]
            fixed_idx = idx_along[f]
            slc = _plane_view(data, f, fixed_idx).T
            mplane = _plane_view(mask, f, fixed_idx).T
            ax.imshow(slc, cmap="gray", origin="lower", vmin=self.vmin, vmax=self.vmax)
            ax.imshow(_overlay_rgba(mplane), origin="lower")
            if mplane.sum() > 0:
                ax.contour(mplane, levels=[0.5], colors=["cyan"], linewidths=1)

            # crosshairs showing where the other two slices are
            other = {0: self.x, 1: self.y, 2: self.z}
            a_val, b_val = other[cfg["a_axis"]], other[cfg["b_axis"]]
            ax.axvline(a_val, color="yellow", lw=0.5, alpha=0.5)
            ax.axhline(b_val, color="yellow", lw=0.5, alpha=0.5)

            ax.set_title(f"{cfg['title']} (idx={fixed_idx})")
            ax.axis("off")

        self.fig.canvas.draw_idle()

    # ── widget callbacks ────────────────────────────────────────────────
    def _on_phase(self, _):
        self.t = int(self.sl_phase.val)
        self.draw()

    def _on_slice(self, _):
        self.z = int(self.sl_z.val)
        self.y = int(self.sl_y.val)
        self.x = int(self.sl_x.val)
        self.draw()

    def _on_brush(self, _):
        self.brush_mm = float(self.sl_brush.val)
        self.draw()

    def _on_mode(self, label):
        self.mode = label
        self.draw()

    def _on_reset(self, _):
        self.masks[self.t] = self.masks_auto[self.t].copy()
        self.edited[self.t] = False
        self.draw()

    def _on_play(self, _):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text("Pause")
            self.timer.start()
        else:
            self.btn_play.label.set_text("Play")
            self.timer.stop()

    def _advance_phase(self):
        self.sl_phase.set_val((self.t + 1) % self.n)

    def _on_save(self, _):
        self.timer.stop()
        plt.close(self.fig)

    # ── brush painting via mouse ────────────────────────────────────────
    def _panel_index(self, event):
        for i, ax in enumerate(self.axes):
            if event.inaxes is ax:
                return i
        return None

    def _paint_at(self, panel_idx, xdata, ydata):
        cfg = PANELS[panel_idx]
        f = cfg["axis_fixed"]
        fixed_idx = {2: self.z, 1: self.y, 0: self.x}[f]
        voxel_sizes = self.voxel_sizes_list[self.t]
        a_idx, b_idx = int(round(xdata)), int(round(ydata))
        va, vb = voxel_sizes[cfg["a_axis"]], voxel_sizes[cfg["b_axis"]]
        paint_disk(self.masks[self.t], f, fixed_idx, a_idx, b_idx, va, vb,
                   self.brush_mm, add=(self.mode == "Add"))
        self.edited[self.t] = True

    def _on_press(self, event):
        if event.button != 1:
            return
        panel_idx = self._panel_index(event)
        if panel_idx is None or event.xdata is None or event.ydata is None:
            return
        self.painting = True
        self.paint_panel = panel_idx
        self._paint_at(panel_idx, event.xdata, event.ydata)
        self.draw()

    def _on_motion(self, event):
        if not self.painting or self.paint_panel is None:
            return
        if event.inaxes is not self.axes[self.paint_panel]:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._paint_at(self.paint_panel, event.xdata, event.ydata)
        self.draw()

    def _on_release(self, _event):
        self.painting = False
        self.paint_panel = None


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def save_edited_outputs(editor, affine, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    save_4d_nifti(editor.masks, affine,
                  os.path.join(out_dir, "segmentation_4d_edited.nii.gz"))

    csv_path = os.path.join(out_dir, "volumes_edited.csv")
    with open(csv_path, "w") as f:
        f.write("folder,frame_idx,filename,volume_ml_auto,volume_ml_edited,"
                "diff_ml,edited\n")
        for i, (folder, frame_idx, filename) in enumerate(editor.meta):
            v_auto = compute_volume_ml(editor.masks_auto[i], editor.voxel_vol_list[i])
            v_edit = compute_volume_ml(editor.masks[i], editor.voxel_vol_list[i])
            f.write(f"{folder},{frame_idx},{filename},{v_auto:.4f},{v_edit:.4f},"
                    f"{v_edit - v_auto:+.4f},{editor.edited[i]}\n")
    print(f"Edited volumes CSV saved: {csv_path}")

    plot_path = os.path.join(out_dir, "volume_edited_plot.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    folders_seen = []
    for folder, _, _ in editor.meta:
        if folder not in folders_seen:
            folders_seen.append(folder)
    frames_x, auto_vals, edit_vals, boundaries = [], [], [], []
    offset = 0
    for folder in folders_seen:
        idxs = [i for i, m in enumerate(editor.meta) if m[0] == folder]
        fxs = [offset + editor.meta[i][1] for i in idxs]
        frames_x += fxs
        auto_vals += [compute_volume_ml(editor.masks_auto[i], editor.voxel_vol_list[i]) for i in idxs]
        edit_vals += [compute_volume_ml(editor.masks[i], editor.voxel_vol_list[i]) for i in idxs]
        boundaries.append((offset, folder))
        offset = max(fxs) + 1
    ax.plot(frames_x, auto_vals, "o--", color="tab:gray", label="auto (v2)")
    ax.plot(frames_x, edit_vals, "o-", color="tab:cyan", label="manually edited")
    for x, _ in boundaries[1:]:
        ax.axvline(x - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Sequential frame index")
    ax.set_ylabel("LV cavity volume (mL)")
    ax.set_title("LV Volume Over Time: auto vs. manually edited")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Edited volume plot saved: {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Slideshow through cardiac phases + manual brush editing "
                     "of the v2 (conical + valve-plane) LV segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folders", nargs="+", default=["retro_A_frames", "retro_B_frames"])
    parser.add_argument("--seed", type=int, nargs=3, metavar=("X", "Y", "Z"),
                         help="Seed voxel coordinate inside the LV cavity (skip interactive selection)")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override intensity threshold (HU).")
    parser.add_argument("--sphere-radius", type=float, default=70,
                         help="Cone base radius at the valve plane (mm). Default: 70")
    parser.add_argument("--apex-radius", type=float, default=15,
                         help="Cone radius at the apex end of the taper (mm). Default: 15")
    parser.add_argument("--cone-length", type=float, default=90,
                         help="Distance (mm) over which the cone tapers. Default: 90")
    parser.add_argument("--apex-sphere-radius", type=float, default=25,
                         help="Radius (mm) of the apex sphere bulge unioned with the "
                              "cone. Default: 25")
    parser.add_argument("--valve-search-mm", type=float, default=60,
                         help="How far (mm) from the seed to search for the valve-plane waist. Default: 60")
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                         help="LV long axis in voxel space. Default: 1")
    parser.add_argument("--apex-dir", type=int, default=1, choices=[-1, 1],
                         help="+1 or -1: which direction along --axis is the apex. Default: 1")
    parser.add_argument("--temporal-window", type=int, default=1,
                         help="Temporal smoothing window applied to the auto masks before editing.")
    parser.add_argument("--brush-radius", type=float, default=5.0,
                         help="Initial brush radius (mm). Default: 5.0")
    parser.add_argument("--fps", type=float, default=4.0,
                         help="Slideshow playback speed (frames/sec). Default: 4.0")
    parser.add_argument("--output", default="results_conical",
                         help="Output directory (default: results_conical)")
    parser.add_argument("--no-cache", action="store_true",
                         help="Ignore any existing segmentation cache and force a full re-segmentation.")
    args = parser.parse_args()

    # ── try the cache first: if the user accepts it, we skip straight to
    # the temporal smoothing + editor, never touching segment_lv_cavity_conical.
    cached = maybe_prompt_for_cache(args)

    if cached is not None:
        all_data, raw_masks, meta, voxel_sizes_list, voxel_vol_list, shared_affine = cached
    else:
        frames = collect_frames(args.folders)
        if not frames:
            print("ERROR: No NIfTI files found in specified folders.")
            return
        print(f"Found {len(frames)} frames across {len(args.folders)} folder(s)")

        _, first_path = frames[0]
        print(f"\nLoading first frame: {first_path}")
        first_data, first_affine, _, _ = load_nifti(first_path)

        if args.seed:
            seed = tuple(args.seed)
            print(f"Using provided seed: {seed}")
        else:
            seed = select_seed_interactive(first_data)

        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using provided threshold: {threshold:.0f} HU")
        else:
            threshold = estimate_threshold(first_data, seed)
            print(f"Auto-detected threshold: {threshold:.0f} HU")

        all_data, raw_masks, voxel_sizes_list, voxel_vol_list, meta = [], [], [], [], []
        filepaths = []
        shared_affine = first_affine

        for i, (folder_name, filepath) in enumerate(frames):
            filename = os.path.basename(filepath)
            frame_idx = int(filename.split("frame_")[1].split(".")[0])
            print(f"\n[{i + 1}/{len(frames)}] {folder_name}/{filename}")

            data, _, voxel_sizes, voxel_vol = load_nifti(filepath)

            mask, _, valve_idx = segment_lv_cavity_conical(
                data, seed, voxel_sizes, threshold=threshold,
                axis=args.axis, apex_dir=args.apex_dir,
                base_radius_mm=args.sphere_radius, apex_radius_mm=args.apex_radius,
                cone_length_mm=args.cone_length, valve_search_mm=args.valve_search_mm,
                apex_sphere_radius_mm=args.apex_sphere_radius,
            )
            print(f"  v2 volume: {compute_volume_ml(mask, voxel_vol):.2f} mL "
                  f"(valve plane @ {valve_idx})")

            all_data.append(data)
            raw_masks.append(mask)
            voxel_sizes_list.append(voxel_sizes)
            voxel_vol_list.append(voxel_vol)
            meta.append((folder_name, frame_idx, filename))
            filepaths.append(filepath)

        params = {
            "folders": list(args.folders),
            "seed": list(seed),
            "threshold": threshold,
            "sphere_radius": args.sphere_radius,
            "apex_radius": args.apex_radius,
            "cone_length": args.cone_length,
            "apex_sphere_radius": args.apex_sphere_radius,
            "valve_search_mm": args.valve_search_mm,
            "axis": args.axis,
            "apex_dir": args.apex_dir,
        }
        save_segmentation_cache(raw_masks, meta, filepaths, voxel_sizes_list,
                                 voxel_vol_list, shared_affine, params, args.output)

    if args.temporal_window > 0:
        print(f"\nTemporal smoothing (window={args.temporal_window})...")
        masks_auto = temporal_smooth_masks(raw_masks, window=args.temporal_window)
    else:
        masks_auto = raw_masks

    print("\nOpening editor — scrub the 'Cardiac phase' slider or hit Play for "
          "the slideshow; left-click-drag on any panel to paint/erase; "
          "'Save & Exit' when done.")
    editor = MaskEditor(all_data, masks_auto, voxel_sizes_list, voxel_vol_list,
                         meta, brush_mm=args.brush_radius, fps=args.fps)

    save_edited_outputs(editor, shared_affine, args.output)


if __name__ == "__main__":
    main()