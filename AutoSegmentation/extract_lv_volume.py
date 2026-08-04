"""
Extract left ventricle cavity volume over time from ex vivo cardiac CT NIfTI sequences.

Pipeline:
    1. User provides a seed point inside the LV cavity (interactive or via --seed)
    2. Multi-Otsu thresholding separates cavity (bright) from myocardium (gray)
    3. Sphere-constrained region growing from seed captures the cavity
    4. Morphological close->open solidifies the cavity and breaks the catheter
    5. Catheter remnants are removed by detecting the narrow neck below the cavity
    6. Volume is computed from the cleaned mask and voxel spacing

Usage:
    Interactive (click seed on first frame):
        python extract_lv_volume.py

    Non-interactive (provide seed voxel coordinate):
        python extract_lv_volume.py --seed 260 280 144

    Override intensity threshold:
        python extract_lv_volume.py --seed 260 280 144 --threshold 267

    Process only one folder:
        python extract_lv_volume.py --folders retro_A_frames
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from glob import glob
from scipy import ndimage
from skimage.morphology import ball
from skimage.filters import threshold_multiotsu


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_nifti(path):
    """Load a NIfTI file, return (data, affine, voxel_sizes_mm, voxel_volume_mm3)."""
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    voxel_sizes = nib.affines.voxel_sizes(affine)
    voxel_volume_mm3 = np.prod(voxel_sizes)
    return data, affine, voxel_sizes, voxel_volume_mm3


def collect_frames(folders):
    """Collect all NIfTI files from the given folders, sorted by name.

    Returns list of (folder_name, filepath) tuples.
    """
    frames = []
    for folder in folders:
        files = sorted(glob(os.path.join(folder, "*.nii*")))
        folder_name = os.path.basename(folder)
        for f in files:
            frames.append((folder_name, f))
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Interactive seed selection
# ──────────────────────────────────────────────────────────────────────────────

def select_seed_interactive(data):
    """Show a 3-plane viewer. User clicks to place seed, then confirms.

    Returns (x, y, z) voxel coordinate.
    """
    seed = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]
    confirmed = [False]

    vmin, vmax = -200, 600

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Click on the LV cavity to place seed, then press 'Confirm'",
        fontsize=13,
    )
    plt.subplots_adjust(bottom=0.25)

    ax_sl_ax = plt.axes([0.15, 0.12, 0.2, 0.03])
    ax_sl_cor = plt.axes([0.42, 0.12, 0.2, 0.03])
    ax_sl_sag = plt.axes([0.69, 0.12, 0.2, 0.03])
    sl_ax = Slider(ax_sl_ax, "Axial", 0, data.shape[2] - 1, valinit=seed[2], valstep=1)
    sl_cor = Slider(ax_sl_cor, "Coronal", 0, data.shape[1] - 1, valinit=seed[1], valstep=1)
    sl_sag = Slider(ax_sl_sag, "Sagittal", 0, data.shape[0] - 1, valinit=seed[0], valstep=1)

    ax_btn = plt.axes([0.42, 0.03, 0.16, 0.05])
    btn = Button(ax_btn, "Confirm Seed")

    def draw():
        for ax in axes:
            ax.clear()
        axes[0].imshow(data[:, :, seed[2]].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0].axhline(seed[1], color="g", lw=0.5, alpha=0.5)
        axes[0].axvline(seed[0], color="y", lw=0.5, alpha=0.5)
        axes[0].set_title(f"Axial z={seed[2]}")
        axes[0].plot(seed[0], seed[1], "r+", markersize=12, markeredgewidth=2)

        axes[1].imshow(data[:, seed[1], :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[1].axhline(seed[2], color="r", lw=0.5, alpha=0.5)
        axes[1].axvline(seed[0], color="y", lw=0.5, alpha=0.5)
        axes[1].set_title(f"Coronal y={seed[1]}")
        axes[1].plot(seed[0], seed[2], "r+", markersize=12, markeredgewidth=2)

        axes[2].imshow(data[seed[0], :, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[2].axhline(seed[2], color="r", lw=0.5, alpha=0.5)
        axes[2].axvline(seed[1], color="g", lw=0.5, alpha=0.5)
        axes[2].set_title(f"Sagittal x={seed[0]}")
        axes[2].plot(seed[1], seed[2], "r+", markersize=12, markeredgewidth=2)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == axes[0]:
            seed[0] = int(round(event.xdata))
            seed[1] = int(round(event.ydata))
            sl_sag.set_val(seed[0])
            sl_cor.set_val(seed[1])
        elif event.inaxes == axes[1]:
            seed[0] = int(round(event.xdata))
            seed[2] = int(round(event.ydata))
            sl_sag.set_val(seed[0])
            sl_ax.set_val(seed[2])
        elif event.inaxes == axes[2]:
            seed[1] = int(round(event.xdata))
            seed[2] = int(round(event.ydata))
            sl_cor.set_val(seed[1])
            sl_ax.set_val(seed[2])
        else:
            return
        draw()

    def on_slider(_):
        seed[2] = int(sl_ax.val)
        seed[1] = int(sl_cor.val)
        seed[0] = int(sl_sag.val)
        draw()

    def on_confirm(_):
        confirmed[0] = True
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    sl_ax.on_changed(on_slider)
    sl_cor.on_changed(on_slider)
    sl_sag.on_changed(on_slider)
    btn.on_clicked(on_confirm)

    draw()
    plt.show()

    if not confirmed[0]:
        raise RuntimeError("Seed selection was cancelled (window closed without confirming).")

    print(f"Seed selected: ({seed[0]}, {seed[1]}, {seed[2]})")
    print(f"Intensity at seed: {data[seed[0], seed[1], seed[2]]:.1f} HU")
    return tuple(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Threshold estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_threshold(data, seed, radius_vox=75):
    """Use multi-Otsu on a local ROI around the seed to find cavity threshold.

    The highest Otsu class boundary separates bright cavity from gray myocardium.
    """
    x, y, z = seed
    roi = data[
        max(0, x - radius_vox): x + radius_vox,
        max(0, y - radius_vox): y + radius_vox,
        max(0, z - radius_vox): z + radius_vox,
    ]
    tissue = roi[roi > -200].ravel()
    thresholds = threshold_multiotsu(tissue, classes=3)
    # The upper threshold separates myocardium from cavity
    thresh = thresholds[-1]
    print(f"  Multi-Otsu thresholds: {[f'{t:.0f}' for t in thresholds]} -> using {thresh:.0f} HU")
    return thresh


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def find_nearest_above_threshold(data, seed, threshold, search_radius=15):
    """If seed is below threshold, find the nearest voxel above threshold."""
    if data[seed[0], seed[1], seed[2]] > threshold:
        return seed
    r = search_radius
    patch = data[
        max(0, seed[0]-r):seed[0]+r+1,
        max(0, seed[1]-r):seed[1]+r+1,
        max(0, seed[2]-r):seed[2]+r+1,
    ]
    above = patch > threshold
    if not above.any():
        return seed
    # Find closest above-threshold voxel (Euclidean distance in voxel units)
    coords = np.array(np.where(above)).T
    center = np.array([seed[0] - max(0, seed[0]-r),
                        seed[1] - max(0, seed[1]-r),
                        seed[2] - max(0, seed[2]-r)])
    dists = np.sum((coords - center) ** 2, axis=1)
    closest = coords[np.argmin(dists)]
    adjusted = (
        max(0, seed[0]-r) + int(closest[0]),
        max(0, seed[1]-r) + int(closest[1]),
        max(0, seed[2]-r) + int(closest[2]),
    )
    return adjusted


def segment_lv_cavity(data, seed, voxel_sizes, threshold, sphere_radius_mm=50,
                      extend_directions=((1, +1), (0, +1), (0, -1))):
    """Segment the LV cavity from a single CT frame.

    Steps:
        1. Snap seed to nearest above-threshold voxel if needed
        2. Restrict to sphere around seed
        3. Region growing from seed
        4. Close (fill internal gaps) then adaptive open (break catheter)
        5. Remove catheter remnants by position (narrow-neck detection)
        6. Extend mask along `extend_directions` to recover voxels lost to
           morphology. Default is +y (apex) and ±x (lateral walls). Omits
           -y (catheter side) so the catheter doesn't regrow. Pass an
           empty tuple () to skip.

    Returns (mask, centroid).
    """
    print(f"  Threshold: {threshold:.0f} HU")

    # 1. Snap seed if below threshold
    orig_seed = seed
    seed = find_nearest_above_threshold(data, seed, threshold)
    if seed != orig_seed:
        print(f"  Seed snapped from {orig_seed} ({data[orig_seed]:.0f} HU) to "
              f"{seed} ({data[seed]:.0f} HU)")

    # 2. Sphere constraint around seed
    zz, yy, xx = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    dist = np.sqrt(
        ((zz - seed[0]) * voxel_sizes[0]) ** 2
        + ((yy - seed[1]) * voxel_sizes[1]) ** 2
        + ((xx - seed[2]) * voxel_sizes[2]) ** 2
    )
    sphere = dist <= sphere_radius_mm

    # 3. Region growing within sphere
    in_range = ((data > threshold) & sphere).astype(np.uint8)
    labeled, _ = ndimage.label(in_range)
    seed_label = labeled[seed[0], seed[1], seed[2]]

    if seed_label == 0:
        raise RuntimeError(
            f"Seed ({seed}) not in any above-threshold component even after snapping. "
            "Try a lower threshold."
        )

    mask = (labeled == seed_label).astype(np.uint8)
    print(f"  Region growing: {mask.sum()} voxels")

    # 4. Morphological close (fill gaps) then adaptive open (break catheter)
    #    Close uses small ball to avoid bridging to nearby structures.
    #    Open uses larger ball to break thin connections; falls back to
    #    smaller ball if the cavity itself gets erased.
    closed = ndimage.binary_closing(mask, structure=ball(2)).astype(np.uint8)

    final_mask = None
    for r in [4, 3, 2]:
        opened = ndimage.binary_opening(closed, structure=ball(r)).astype(np.uint8)
        lab, n = ndimage.label(opened)
        if n > 0 and opened.sum() > 1000:
            sizes = ndimage.sum(opened, lab, range(1, n + 1))
            core = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)
            m = ndimage.binary_dilation(core, structure=ball(r)).astype(np.uint8)
            m = (m & closed).astype(np.uint8)
            final_mask = m
            print(f"  Cleanup: close(ball(2)) + open(ball({r}))")
            break

    if final_mask is None:
        raise RuntimeError("Morphological cleanup eroded the cavity completely.")

    mask = final_mask

    # 5. Remove catheter remnants by detecting narrow neck
    mask = remove_catheter_by_position(mask)

    if mask.sum() == 0:
        raise RuntimeError("Segmentation produced empty mask after catheter removal.")

    # 6. Extend mask to recover voxels lost by morphological opening
    if extend_directions:
        mask = extend_mask_constrained(mask, data, threshold,
                                        directions=list(extend_directions))

    # 7. Fill holes robustly. 3D fill_holes handles cavities fully enclosed
    #    in 3D; per-slice fill_holes (in all 3 orientations) catches gaps
    #    that have thin 1-voxel escape paths in 3D but are clearly
    #    interior when viewed as a slice.
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    for z in range(mask.shape[2]):
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    for y in range(mask.shape[1]):
        mask[:, y, :] = ndimage.binary_fill_holes(mask[:, y, :])
    for x in range(mask.shape[0]):
        mask[x, :, :] = ndimage.binary_fill_holes(mask[x, :, :])

    # 8. Light 3D closing to smooth the boundary
    mask = ndimage.binary_closing(mask, structure=ball(2)).astype(np.uint8)

    centroid = ndimage.center_of_mass(mask)
    centroid = tuple(int(round(c)) for c in centroid)
    return mask, centroid


def _shift_one(mask, axis, direction):
    """Shift a bool mask by one voxel along (axis, direction), zero-filling."""
    shifted = np.zeros_like(mask)
    if axis == 0:
        if direction > 0:
            shifted[1:, :, :] = mask[:-1, :, :]
        else:
            shifted[:-1, :, :] = mask[1:, :, :]
    elif axis == 1:
        if direction > 0:
            shifted[:, 1:, :] = mask[:, :-1, :]
        else:
            shifted[:, :-1, :] = mask[:, 1:, :]
    else:  # axis == 2
        if direction > 0:
            shifted[:, :, 1:] = mask[:, :, :-1]
        else:
            shifted[:, :, :-1] = mask[:, :, 1:]
    return shifted


def extend_mask_constrained(mask, data, threshold, directions, max_steps=50):
    """Iteratively grow the mask along the given axis/direction pairs.

    The LV apex and lateral walls often get eroded by morphological opening.
    This function recovers them by taking the current mask's edge in each
    allowed direction and growing outward one voxel at a time, keeping any
    new voxel that is (a) adjacent to the current mask and (b) above
    threshold.

    Only grows in directions listed in `directions` — omit the catheter
    direction so it doesn't regrow.

    Args:
        mask: 3D uint8/bool mask
        data: 3D CT volume
        threshold: HU threshold — voxels above this can be added
        directions: list of (axis, direction) tuples, e.g. [(1, +1), (0, +1),
                    (0, -1)] for +y, +x, -x
        max_steps: cap on the number of 1-voxel growth steps

    Returns:
        Extended uint8 mask
    """
    extended = mask.astype(bool).copy()
    bright = data > threshold
    counts = {d: 0 for d in directions}

    for _ in range(max_steps):
        added_this_step = False
        for axis, direction in directions:
            shifted = _shift_one(extended, axis, direction)
            new_voxels = shifted & bright & (~extended)
            n_new = int(new_voxels.sum())
            if n_new > 0:
                extended |= new_voxels
                counts[(axis, direction)] += n_new
                added_this_step = True
        if not added_this_step:
            break

    parts = []
    for (axis, direction), n in counts.items():
        name = {0: "x", 1: "y", 2: "z"}[axis]
        parts.append(f"{'+' if direction > 0 else '-'}{name}:+{n}")
    print(f"  Mask extension ({', '.join(parts)})")
    return extended.astype(np.uint8)


def remove_catheter_by_position(mask):
    """Remove catheter by detecting the narrow neck below the LV cavity.

    Scans along each axis from the centroid outward. Where the cross-sectional
    area drops below 15% of the maximum, everything beyond is removed.
    """
    if mask.sum() == 0:
        return mask

    com = ndimage.center_of_mass(mask)

    # Check all 3 axes for catheter-like extensions
    for axis in range(3):
        com_idx = int(round(com[axis]))
        # Compute per-slice area along this axis
        areas = np.array([
            np.take(mask, i, axis=axis).sum()
            for i in range(mask.shape[axis])
        ])

        if areas.max() == 0:
            continue

        max_area = areas[max(0, com_idx - 10):min(len(areas), com_idx + 10)].max()
        cutoff_thresh = 0.15 * max_area

        # Scan downward from centroid
        cutoff_lo = 0
        for i in range(com_idx, -1, -1):
            if areas[i] < cutoff_thresh:
                cutoff_lo = i + 1
                break

        # Scan upward from centroid
        cutoff_hi = mask.shape[axis]
        for i in range(com_idx, mask.shape[axis]):
            if areas[i] < cutoff_thresh:
                cutoff_hi = i
                break

        # Only crop if we'd remove a small fraction (catheter, not half the cavity)
        total_in_range = sum(areas[cutoff_lo:cutoff_hi])
        if total_in_range > 0.7 * mask.sum():
            sl_lo = [slice(None)] * 3
            sl_lo[axis] = slice(0, cutoff_lo)
            mask[tuple(sl_lo)] = 0

            sl_hi = [slice(None)] * 3
            sl_hi[axis] = slice(cutoff_hi, mask.shape[axis])
            mask[tuple(sl_hi)] = 0

    # Keep largest component after trimming
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Volume computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_volume_ml(mask, voxel_volume_mm3):
    """Compute volume in mL from a binary mask."""
    return mask.sum() * voxel_volume_mm3 / 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# QC output
# ──────────────────────────────────────────────────────────────────────────────

def save_overlay(data, mask, centroid, frame_name, out_path, vmin=-200, vmax=600):
    """Save a 3-plane overlay image with segmentation contour."""
    cx, cy, cz = centroid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(frame_name, fontsize=12)

    for ax, slc, msk, title in [
        (axes[0], data[:, :, cz].T, mask[:, :, cz].T, f"Axial z={cz}"),
        (axes[1], data[:, cy, :].T, mask[:, cy, :].T, f"Coronal y={cy}"),
        (axes[2], data[cx, :, :].T, mask[cx, :, :].T, f"Sagittal x={cx}"),
    ]:
        ax.imshow(slc, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if msk.sum() > 0:
            ax.contour(msk, levels=[0.5], colors=["red"], linewidths=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_montage(all_data, all_masks, all_centroids, all_names, out_path, vmin=-200, vmax=600):
    """Save a tiled montage of axial mid-slices with contours."""
    n = len(all_data)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for i in range(n):
        cz = all_centroids[i][2]
        axes_flat[i].imshow(all_data[i][:, :, cz].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if all_masks[i][:, :, cz].sum() > 0:
            axes_flat[i].contour(all_masks[i][:, :, cz].T, levels=[0.5], colors=["red"], linewidths=1)
        axes_flat[i].set_title(all_names[i], fontsize=9)
        axes_flat[i].axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_volume_plot(results, out_path):
    """Plot volume (mL) vs sequential frame index.

    Folders are concatenated along x: folder 0's frames come first, then
    folder 1's frames continue from where folder 0 ended, etc.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Preserve the order folders first appear in `results`
    folders = []
    for r in results:
        if r["folder"] not in folders:
            folders.append(r["folder"])

    offset = 0
    for folder in folders:
        subset = [r for r in results if r["folder"] == folder]
        frames = [offset + r["frame_idx"] for r in subset]
        volumes = [r["volume_ml"] for r in subset]
        ax.plot(frames, volumes, "o-", label=folder, markersize=6)
        # Next folder starts right after this one's last frame
        offset = max(frames) + 1

    ax.set_xlabel("Sequential frame index")
    ax.set_ylabel("LV cavity volume (mL)")
    ax.set_title("LV Volume Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Volume plot saved: {out_path}")


def save_pv_loop(results, pressure_path, out_path):
    """Plot LV volume vs pressure (PV loop) alongside P and V vs time.

    Reads pressure from a two-column (time, pressure) file whose first
    line is a header like '200 1'. The 20 frame volumes are mapped to
    pressure samples evenly spaced along the time axis — i.e. frame i
    corresponds to sample index round(i * (N_samples - 1) / (N_frames - 1)).

    Pressure in the .dat is assumed to be in Pascals; display converts
    to mmHg (divide by 133.322).
    """
    if not os.path.exists(pressure_path):
        print(f"Pressure file not found: {pressure_path} — skipping PV loop")
        return

    with open(pressure_path) as f:
        f.readline()  # discard header
        pdata = np.loadtxt(f)
    if pdata.ndim != 2 or pdata.shape[1] != 2:
        print(f"Unexpected pressure file shape {pdata.shape} — skipping PV loop")
        return

    volumes = np.array([r["volume_ml"] for r in results])
    n_frames = len(volumes)
    if n_frames < 2:
        return

    indices = np.linspace(0, len(pdata) - 1, n_frames).astype(int)
    times = pdata[indices, 0]
    pressures_Pa = pdata[indices, 1]
    pressures_mmHg = pressures_Pa / 133.322

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P & V vs time
    ax1 = axes[0]
    l1 = ax1.plot(times, pressures_mmHg, "o-", color="tab:orange", label="Pressure")[0]
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax2 = ax1.twinx()
    l2 = ax2.plot(times, volumes, "o-", color="tab:blue", label="Volume")[0]
    ax2.set_ylabel("LV volume (mL)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Pressure & Volume vs time")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2], ["Pressure", "Volume"], loc="upper right")

    # Right: PV loop, colored by frame index to show direction
    ax3 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
    for i in range(n_frames - 1):
        ax3.plot(pressures_mmHg[i:i+2], volumes[i:i+2],
                 "-", color=colors[i], linewidth=1.5)
    sc = ax3.scatter(pressures_mmHg, volumes,
                     c=range(n_frames), cmap="viridis", s=50, zorder=5)
    plt.colorbar(sc, ax=ax3, label="Frame index")
    ax3.set_xlabel("Pressure (mmHg)")
    ax3.set_ylabel("LV volume (mL)")
    ax3.set_title("LV P-V loop")
    ax3.grid(True, alpha=0.3)
    ax3.annotate("start", (pressures_mmHg[0], volumes[0]),
                 textcoords="offset points", xytext=(5, 5))
    ax3.annotate("end", (pressures_mmHg[-1], volumes[-1]),
                 textcoords="offset points", xytext=(5, 5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PV loop saved: {out_path}")


def temporal_smooth_masks(masks, window=1):
    """Smooth masks temporally via majority vote over a sliding window.

    For each frame i, take the binary mask of the 2*window+1 frames centered
    at i and keep voxels that appear in > half of them. This removes voxels
    that flicker in/out between consecutive frames.
    """
    if window <= 0 or len(masks) < 3:
        return masks
    smoothed = []
    n = len(masks)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        stack = np.stack(masks[lo:hi], axis=0).astype(np.uint8)
        votes = stack.sum(axis=0)
        thresh = (hi - lo) / 2.0
        smoothed.append((votes > thresh).astype(np.uint8))
    return smoothed


def save_4d_nifti(masks, affine, out_path):
    """Stack masks along a 4th (time) axis and save as NIfTI.

    Some ParaView builds don't include a NIfTI reader; in that case use
    `save_paraview_series()` which writes native VTI + PVD files instead.
    """
    arr = np.stack(masks, axis=-1).astype(np.uint8)
    img = nib.Nifti1Image(arr, affine)
    img.header["pixdim"][4] = 1.0  # 1 time step per frame (unitless)
    nib.save(img, out_path)
    print(f"4D segmentation (NIfTI) saved: {out_path} (shape {arr.shape})")


def save_paraview_series(masks, affine, out_dir, basename="segmentation"):
    """Write masks as a ParaView-native VTI series + PVD index file.

    VTI is ParaView's XML ImageData format — always loadable without any
    reader plugins. A .pvd file is a small XML index that lists each VTI
    as a time step, so ParaView treats the whole series as time-varying
    data and shows a VCR/timeline UI automatically.

    To open in ParaView: File → Open → pick the .pvd file.
    """
    import vtk

    os.makedirs(out_dir, exist_ok=True)

    # Spacing and origin from the affine. The affine's sign may be negative
    # (radiological orientation) — VTK ImageData expects positive spacing,
    # so take absolute values and set the origin accordingly.
    spacing = [abs(affine[i, i]) for i in range(3)]
    origin = [affine[i, 3] for i in range(3)]

    n_frames = len(masks)
    shape = masks[0].shape  # (nx, ny, nz)

    vti_rel_paths = []
    for t, mask in enumerate(masks):
        image = vtk.vtkImageData()
        image.SetDimensions(shape[0], shape[1], shape[2])
        image.SetSpacing(*spacing)
        image.SetOrigin(*origin)

        # VTK expects Fortran-ordered flat array for SetArray; use numpy
        # ascontiguousarray in Fortran order to match the VTK layout.
        arr = np.ascontiguousarray(mask.astype(np.uint8).ravel(order="F"))
        from vtk.util.numpy_support import numpy_to_vtk
        vtk_arr = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_arr.SetName("segmentation")
        image.GetPointData().SetScalars(vtk_arr)

        vti_name = f"{basename}_{t:04d}.vti"
        vti_path = os.path.join(out_dir, vti_name)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(vti_path)
        writer.SetInputData(image)
        writer.SetDataModeToBinary()
        writer.Write()
        vti_rel_paths.append(vti_name)

    # Write PVD index
    pvd_path = os.path.join(out_dir, f"{basename}.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for t, name in enumerate(vti_rel_paths):
            f.write(f'    <DataSet timestep="{t}" group="" part="0" '
                    f'file="{name}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
    print(f"4D segmentation (VTI series) saved: {pvd_path} + {n_frames} VTI files")


def save_cycle_video(all_data, all_masks, all_centroids, all_names, out_path,
                     fps=4, vmin=-200, vmax=600):
    """Render a 3-plane video over the cardiac cycle.

    Each frame shows the axial / coronal / sagittal slices through the
    centroid of that frame's mask, with the segmentation contour in red.
    Writes an MP4 using moviepy's ImageSequenceClip (ffmpeg backend).
    """
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("moviepy not available — skipping video")
            return

    tmp_dir = os.path.join(os.path.dirname(out_path), "_video_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    n = len(all_data)
    img_paths = []
    for i in range(n):
        cx, cy, cz = all_centroids[i]
        data = all_data[i]
        mask = all_masks[i]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.suptitle(all_names[i], fontsize=11)

        for ax, slc, msk, title in [
            (axes[0], data[:, :, cz].T, mask[:, :, cz].T, f"Axial z={cz}"),
            (axes[1], data[:, cy, :].T, mask[:, cy, :].T, f"Coronal y={cy}"),
            (axes[2], data[cx, :, :].T, mask[cx, :, :].T, f"Sagittal x={cx}"),
        ]:
            ax.imshow(slc, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if msk.sum() > 0:
                ax.contour(msk, levels=[0.5], colors=["red"], linewidths=1.2)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        img_path = os.path.join(tmp_dir, f"frame_{i:03d}.png")
        plt.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close()
        img_paths.append(img_path)

    clip = ImageSequenceClip(img_paths, fps=fps)
    # Ensure even dimensions for h264
    clip = clip.resized(lambda t: (clip.w // 2 * 2, clip.h // 2 * 2))
    clip.write_videofile(
        out_path, codec="libx264", audio=False, preset="ultrafast",
        ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None,
    )
    # Clean up intermediate PNGs
    for p in img_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    print(f"Cycle video saved: {out_path}")


def save_csv(results, out_path):
    """Save results to CSV."""
    with open(out_path, "w") as f:
        f.write("folder,frame_idx,filename,volume_ml,seed_x,seed_y,seed_z,voxel_count\n")
        for r in results:
            f.write(
                f"{r['folder']},{r['frame_idx']},{r['filename']},"
                f"{r['volume_ml']:.4f},{r['seed'][0]},{r['seed'][1]},{r['seed'][2]},"
                f"{r['voxel_count']}\n"
            )
    print(f"CSV saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract LV cavity volume over time from ex vivo cardiac CT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folders", nargs="+", default=["retro_A_frames", "retro_B_frames"],
        help="Folders containing NIfTI frame sequences",
    )
    parser.add_argument(
        "--seed", type=int, nargs=3, metavar=("X", "Y", "Z"),
        help="Seed voxel coordinate inside the LV cavity (skip interactive selection)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override intensity threshold (HU). If not set, auto-detected via multi-Otsu.",
    )
    parser.add_argument(
        "--sphere-radius", type=float, default=50,
        help="Search sphere radius in mm around seed (default: 50)",
    )
    parser.add_argument(
        "--pressure-file", default="pressure_interpolated.dat",
        help="Two-column (time, pressure) file for PV loop plot. "
             "Set to empty string to skip.",
    )
    parser.add_argument(
        "--temporal-window", type=int, default=1,
        help="Temporal smoothing window (frames on each side). 0 to disable. "
             "Default 1 = majority vote over 3 frames.",
    )
    parser.add_argument(
        "--video-fps", type=int, default=4,
        help="Frames per second for the cycle video (default: 4).",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (default: results)",
    )
    args = parser.parse_args()

    matplotlib.use("Agg")  # non-interactive backend for batch processing

    # Collect all frames
    frames = collect_frames(args.folders)
    if not frames:
        print("ERROR: No NIfTI files found in specified folders.")
        return

    print(f"Found {len(frames)} frames across {len(args.folders)} folder(s)")

    # Output directories
    os.makedirs(args.output, exist_ok=True)
    qc_dir = os.path.join(args.output, "qc_overlays")
    os.makedirs(qc_dir, exist_ok=True)

    # Load first frame
    first_folder, first_path = frames[0]
    print(f"\nLoading first frame: {first_path}")
    first_data, first_affine, first_voxel_sizes, first_voxel_vol = load_nifti(first_path)
    print(f"Shape: {first_data.shape}, voxel sizes: {first_voxel_sizes} mm, "
          f"voxel volume: {first_voxel_vol:.6f} mm3")

    # Determine seed
    if args.seed:
        seed = tuple(args.seed)
        print(f"Using provided seed: {seed}")
    else:
        matplotlib.use("TkAgg")  # switch to interactive backend
        seed = select_seed_interactive(first_data)
        matplotlib.use("Agg")  # switch back

    # Estimate threshold from first frame if not provided
    if args.threshold is not None:
        global_threshold = args.threshold
        print(f"Using provided threshold: {global_threshold:.0f} HU")
    else:
        global_threshold = estimate_threshold(first_data, seed)
        print(f"Auto-detected threshold: {global_threshold:.0f} HU")

    # ── Pass 1: segment every frame, collect masks + data ─────────────
    all_data = []
    all_masks = []  # pre-smoothing
    all_meta = []   # (folder, frame_idx, filename)
    prev_folder = None
    shared_affine = None

    for i, (folder_name, filepath) in enumerate(frames):
        filename = os.path.basename(filepath)
        frame_idx = int(filename.split("frame_")[1].split(".")[0])

        if folder_name != prev_folder:
            prev_folder = folder_name
            print(f"\n{'=' * 60}")
            print(f"Processing folder: {folder_name}")
            print(f"{'=' * 60}")

        print(f"\n[{i + 1}/{len(frames)}] {filename}")

        data, affine, voxel_sizes, voxel_vol = load_nifti(filepath)
        if shared_affine is None:
            shared_affine = affine
            shared_voxel_vol = voxel_vol

        mask, centroid = segment_lv_cavity(
            data, seed, voxel_sizes,
            threshold=global_threshold,
            sphere_radius_mm=args.sphere_radius,
        )
        volume_ml = compute_volume_ml(mask, voxel_vol)
        print(f"  Volume: {volume_ml:.2f} mL ({int(mask.sum())} voxels)")

        all_data.append(data)
        all_masks.append(mask)
        all_meta.append((folder_name, frame_idx, filename))

    # ── Pass 2: temporal smoothing across the full sequence ────────────
    if args.temporal_window > 0:
        print(f"\nTemporal smoothing (window={args.temporal_window})...")
        all_masks = temporal_smooth_masks(all_masks, window=args.temporal_window)

    # ── Pass 3: compute centroids, volumes, per-folder groupings ──────
    results = []
    all_centroids = []
    all_data_by_folder = {}
    all_masks_by_folder = {}
    all_centroids_by_folder = {}
    all_names_by_folder = {}

    for (folder_name, frame_idx, filename), data, mask in zip(
        all_meta, all_data, all_masks
    ):
        if mask.sum() > 0:
            centroid = tuple(int(round(c)) for c in ndimage.center_of_mass(mask))
        else:
            centroid = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)
        all_centroids.append(centroid)

        volume_ml = compute_volume_ml(mask, shared_voxel_vol)
        voxel_count = int(mask.sum())
        results.append({
            "folder": folder_name,
            "frame_idx": frame_idx,
            "filename": filename,
            "volume_ml": volume_ml,
            "seed": centroid,
            "voxel_count": voxel_count,
        })

        # Per-frame QC overlay
        overlay_name = f"{folder_name}_frame_{frame_idx:02d}_overlay.png"
        save_overlay(data, mask, centroid, filename, os.path.join(qc_dir, overlay_name))

        if folder_name not in all_data_by_folder:
            all_data_by_folder[folder_name] = []
            all_masks_by_folder[folder_name] = []
            all_centroids_by_folder[folder_name] = []
            all_names_by_folder[folder_name] = []
        all_data_by_folder[folder_name].append(data)
        all_masks_by_folder[folder_name].append(mask)
        all_centroids_by_folder[folder_name].append(centroid)
        all_names_by_folder[folder_name].append(f"frame_{frame_idx:02d}")

    # ── Save all outputs ──────────────────────────────────────────────
    save_csv(results, os.path.join(args.output, "volumes.csv"))
    save_volume_plot(results, os.path.join(args.output, "volume_plot.png"))
    if args.pressure_file:
        save_pv_loop(results, args.pressure_file,
                     os.path.join(args.output, "pv_loop.png"))

    for folder_name in all_data_by_folder:
        montage_path = os.path.join(args.output, f"montage_{folder_name}.png")
        save_montage(
            all_data_by_folder[folder_name],
            all_masks_by_folder[folder_name],
            all_centroids_by_folder[folder_name],
            all_names_by_folder[folder_name],
            montage_path,
        )
        print(f"Montage saved: {montage_path}")

    # 4D NIfTI + ParaView-native VTI/PVD series
    save_4d_nifti(all_masks, shared_affine,
                  os.path.join(args.output, "segmentation_4d.nii.gz"))
    save_paraview_series(all_masks, shared_affine,
                         os.path.join(args.output, "paraview"))

    # Cardiac cycle video
    all_video_names = [f"{m[0]}/frame_{m[1]:02d}" for m in all_meta]
    save_cycle_video(all_data, all_masks, all_centroids, all_video_names,
                     os.path.join(args.output, "cycle.mp4"),
                     fps=args.video_fps)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['folder']}/frame_{r['frame_idx']:02d}: {r['volume_ml']:.2f} mL")
    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
