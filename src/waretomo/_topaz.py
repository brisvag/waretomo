import contextlib
import io
import multiprocessing
import re
import time
from concurrent import futures

from rich import print
from topaz.commands.denoise3d import denoise, load_model, set_device, train_model
from topaz.torch import set_num_threads


def _run_and_update_progress(progress, task, func, *args, **kwargs):
    std = io.StringIO()
    with futures.ThreadPoolExecutor(1) as executor:
        with contextlib.redirect_stderr(std), contextlib.redirect_stdout(std):
            job = executor.submit(func, *args, **kwargs)

            last_read_pos = 0
            while not job.done():
                time.sleep(0.5)
                std.seek(last_read_pos)
                last = std.read()
                if match := re.search(r"(\d+\.\d+)%", last):
                    last_read_pos = std.tell()
                    progress.update(task, completed=float(match.group(1)))

        try:
            return job.result()
        except RuntimeError as e:
            if "CUDA out of memory." in e.args[0]:
                raise RuntimeError(
                    "Not enough GPU memory. "
                    "Try to lower --topaz-tile-size or --topaz-patch-size"
                ) from e
            raise


def topaz_batch(
    progress,
    tilt_series,
    outdir,
    even,
    odd,
    model_name="unet-3d-10a",
    train=False,
    tile_size=32,
    patch_size=32,
    dry_run=False,
    verbose=False,
    overwrite=False,
):
    set_num_threads(0)
    if train:
        task = progress.add_task(description="Training...")
        model, _ = _run_and_update_progress(
            progress,
            task,
            train_model,
            even_path=even,
            odd_path=odd,
            save_prefix=str(outdir / "trained_models" / model_name),
            save_interval=10,
            device=-2,
            tilesize=patch_size,
            base_kernel_width=11,
            num_workers=multiprocessing.cpu_count(),
        )
    else:
        model = load_model(model_name, base_kernel_width=11)
    model.eval()
    model, use_cuda, num_devices = set_device(model, -2)

    inputs = [ts["recon"] for ts in tilt_series]

    if verbose:
        if len(inputs) > 2:
            print(f"denoising: [{inputs[0]} [...] {inputs[-1]}]")
        else:
            print(f"denoising: {inputs}")
        print(f"output: {outdir}")

    if not dry_run:
        for path in progress.track(inputs, description="Denoising..."):
            subtask = progress.add_task(description=path.name)
            _run_and_update_progress(
                progress,
                subtask,
                denoise,
                model=model,
                path=path,
                outdir=str(outdir),
                batch_size=num_devices,
                patch_size=patch_size,
                padding=patch_size // 2,
                suffix="",
            )
            progress.update(subtask, visible=False)
