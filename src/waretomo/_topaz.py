import contextlib
import io
import re
import time
from concurrent import futures

import pkg_resources
import torch
import torch.nn as nn
from rich import print
from topaz.commands.denoise3d import denoise
from topaz.denoise import UDenoiseNet3D
from topaz.torch import set_num_threads


def topaz_batch_train(progress):
    pass


def topaz_batch(
    progress,
    tilt_series,
    outdir,
    train=False,
    patch_size=32,
    dry_run=False,
    verbose=False,
    overwrite=False,
):
    set_num_threads(0)
    model = UDenoiseNet3D(base_width=7)
    f = pkg_resources.resource_stream(
        "topaz", "pretrained/denoise/unet-3d-10a-v0.2.4.sav"
    )
    state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.cuda()

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
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with futures.ThreadPoolExecutor(1) as executor:
                    job = executor.submit(
                        denoise,
                        model=model,
                        path=path,
                        outdir=str(outdir),
                        batch_size=torch.cuda.device_count(),
                        patch_size=patch_size,
                        padding=patch_size // 2,
                        suffix="",
                    )

                    last_read_pos = 0
                    while not job.done():
                        time.sleep(0.5)
                        stderr.seek(last_read_pos)
                        last = stderr.read()
                        if match := re.search(r"(\d+.\d+)%", last):
                            last_read_pos = stderr.tell()
                            progress.update(subtask, completed=float(match.group(1)))

                    progress.update(subtask, visible=False)

                    try:
                        job.result()
                    except RuntimeError as e:
                        if "CUDA out of memory." in e.args[0]:
                            raise RuntimeError(
                                "Not enough GPU memory. Try a lower --topaz-patch-size"
                            ) from e
                        raise
