import click
import ffmpeg
from pathlib import Path

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

@frames.command()
@click.argument("folder", type=Path)
def stack(folder: Path):
    """Stack frames into a single image."""
    pass