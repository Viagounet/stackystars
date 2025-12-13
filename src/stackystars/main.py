import click
import ffmpeg
import numpy as np

from pathlib import Path
from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

@click.group()
@click.version_option()
def cli():
    """StaryStacks

    This command line tool allows you to stack & analyze astronomy images.
    """

@cli.group()
@click.version_option()
def frames():
    """Main command group for any operation on frames
    """

@frames.command()
@click.argument("video", type=Path)
@click.option(
    "-o",
    "--output-dir",
    type=Path,
    help="Directory where extracted frames will be written (defaults to <video>_frames)",
)
@click.option(
    "-f",
    "--format",
    "image_format",
    type=click.Choice(["png", "jpg", "jpeg"], case_sensitive=False),
    default="png",
    show_default=True,
    help="Image format for the extracted frames",
)
def extract(video: Path, output_dir: Path | None, image_format: str):
    """Extract individual frames from a video using ffmpeg."""

    resolved_video = video.expanduser().resolve()
    if not resolved_video.exists() or not resolved_video.is_file():
        raise click.BadParameter(f"Video '{video}' does not exist", param_hint="video")

    if output_dir is None:
        output_dir = resolved_video.parent / f"{resolved_video.stem}_frames"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / f"frame_%06d.{image_format.lower()}"

    # Suppress ffmpeg noise to keep CLI output readable.
    job = (
        ffmpeg
        .input(str(resolved_video))
        .output(str(output_pattern), format="image2", vsync="0")
        .global_args("-loglevel", "error")
    )

    try:
        click.echo("Starting frames extraction.")
        ffmpeg.run(job, capture_stdout=True, capture_stderr=True, overwrite_output=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else str(exc)
        raise click.ClickException(f"ffmpeg failed to extract frames: {stderr.strip()}") from exc

    generated_frames = sorted(output_dir.glob(f"frame_*.{image_format.lower()}"))
    frame_count = len(generated_frames)
    if frame_count == 0:
        raise click.ClickException("ffmpeg reported success but no frames were produced")

    click.echo(f"Wrote {frame_count} frame(s) to {output_dir}")

def optimize_constrast(img):
    den = denoise_tv_chambolle(img, weight=0.05)

    h, w = den.shape
    m = int(0.05 * min(h, w))                  # ignore 5% border
    inner = np.zeros_like(den, bool)
    inner[m:h-m, m:w-m] = True

    thr = np.percentile(den[inner], 99.99)     # << key change
    out = den.copy()
    out[(den > thr) & inner] = 1.0
    return out

@frames.command()
@click.argument("folder", type=Path)
@click.argument("objects", type=int, default=1)
def stack(folder: Path, objects: int):
    """Stack frames into a single image."""
    if objects != 1:
        raise ValueError("For now, image stacking is limited to a single object.")

    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import rgb2gray
    from skimage import img_as_float, img_as_ubyte, io
    from skimage.transform import resize
    from skimage.feature import blob_doh

    from glob import glob

    imgs_paths = glob(f"{folder}/*")[:75]
    imgs_arrays = []
    imgs_arrays_resized = []
    RESIZE_FACTOR = 2
    click.echo("Iterating and resizing frames.")
    for img_path in tqdm(imgs_paths):
        img = io.imread(img_path)
        h, w = img.shape[:2]
        new_shape = (h // RESIZE_FACTOR, w // RESIZE_FACTOR) + (() if img.ndim == 2 else (img.shape[-1],))
        img_resize = resize(img, new_shape, anti_aliasing=False)
        imgs_arrays.append(img)
        imgs_arrays_resized.append(img_resize)
        
    imgs_arrays_resized_contrast = []

    click.echo("Optimizing contrast.")
    for i, img_array in tqdm(enumerate(imgs_arrays_resized)):
        img_gray = rgb2gray(img_as_float(img_array))
        img_gray_high_contrast = optimize_constrast(img_gray)
        imgs_arrays_resized_contrast.append(img_gray_high_contrast)
        # print(img_gray_high_contrast)
        # io.imsave(f"{i}.png", img_as_ubyte(img_gray_high_contrast))
    
    click.echo("Detecting bounding boxes.")
    boxes = []
    for i, img_array in tqdm(enumerate(imgs_arrays_resized_contrast)):
        blob_doh_result = blob_doh(img_array, max_sigma=30, threshold=0.01)
        if blob_doh_result.shape[0] == objects:
            if objects == 1:
                boxes.append(blob_doh_result[0])
            else:
                pass # To implement
        else:
            boxes.append(None)

    CROP_SIGMA_MULT = 3
    saved = 0

    kept_cropped_images = []
    sum_middle_luminosity = []

    click.echo("Cropping images.")
    for i, (bounding_box, img_array) in tqdm(enumerate(zip(boxes, imgs_arrays))):
        if bounding_box is None:
            continue

        y_r, x_r, sigma_r = bounding_box  # resized coords
        y = float(y_r) * RESIZE_FACTOR
        x = float(x_r) * RESIZE_FACTOR
        sigma = float(sigma_r) * RESIZE_FACTOR

        # Define crop radius (pixels) in original resolution
        radius = int(max(8, (CROP_SIGMA_MULT * sigma) / 2.0))

        h, w = img_array.shape[:2]
        cy, cx = int(round(y)), int(round(x))

        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius)
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius)
        # Guard against degenerate crops
        if (y1 - y0) < 2 or (x1 - x0) < 2:
            continue

        crop = img_array[y0:y1, x0:x1]  # keeps channels if present
        h, w = crop.shape[:2]
        y = h // 2
        row = crop[y, :]
        kept_cropped_images.append(crop)
        sum_middle_luminosity.append(crop.max())
    
    click.echo("Filtering out dark images.")
    sum_middle_luminosity = np.asarray(sum_middle_luminosity)
    if sum_middle_luminosity.size == 0:
        kept_cropped_images_last_quartile = []
    else:
        thr = np.quantile(sum_middle_luminosity, 0.75)  # 75th percentile
        kept_cropped_images_last_quartile = [
            crop for crop, v in zip(kept_cropped_images, sum_middle_luminosity) if v >= thr
        ]
