import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from time import sleep

import GPUtil

from ._threaded import run_threaded


@contextmanager
def _cd(dir):
    prev = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(prev)


def _aretomo(
    input_,
    rawtlt,
    aln,
    xf,
    output,
    warp_mdoc_basename,
    suffix="",
    cmd="AreTomo",
    tilt_axis=0,
    patches=None,
    roi_file=None,
    tilt_corr=True,
    thickness_align=1200,
    thickness_recon=0,
    binning=4,
    px_size=0,
    kv=0,
    dose=0,
    cs=0,
    defocus=0,
    reconstruct=False,
    gpu_queue=None,
    dry_run=False,
    overwrite=False,
):
    log = logging.getLogger("waretomo")
    # cwd dance is necessary cause aretomo messes up paths otherwise
    # need to use os.path.relpath cause pathlib cannot handle non-subpath relative paths
    # https://stackoverflow.com/questions/38083555/using-pathlibs-relative-to-for-directories-on-the-same-level
    cwd = output.parent.absolute()
    input_ = Path(os.path.relpath(input_, cwd))
    rawtlt = Path(os.path.relpath(rawtlt, cwd))
    aln = Path(os.path.relpath(aln, cwd))
    xf = Path(os.path.relpath(xf, cwd))
    output = Path(os.path.relpath(output, cwd))
    if not reconstruct:
        output = output.with_stem(output.stem + "_aligned").with_suffix(".st")
    # LogFile is broken, so we do it ourselves
    aretomolog = output.with_suffix(".aretomolog")
    with _cd(cwd):
        if not overwrite and output.exists():
            raise FileExistsError(output)

    # only one job per gpu
    gpu = 0 if gpu_queue is None else gpu_queue.get()

    options = {
        "InMrc": input_,
        "OutMrc": output,
        # 'LogFile': input_.with_suffix('.log').relative_to(cwd),  # currently broken
        "OutBin": binning,
        "Gpu": gpu,
        "DarkTol": 0,
    }

    if reconstruct:
        options.update(
            {
                "AlnFile": aln,
                "VolZ": thickness_recon,
                "PixSize": px_size,
                "Kv": kv,
                "Cs": cs,
                "Defoc": defocus,
                "FlipVol": 1,
                "WBP": 1,
            }
        )
        if dose:
            options["ImgDose"] = dose

    else:
        options.update(
            {
                "AngFile": rawtlt,
                "AlignZ": thickness_align,
                "TiltCor": int(tilt_corr),
                "VolZ": 0,
                "TiltAxis": f"{tilt_axis} 1" if tilt_axis is not None else "0 1",
                "OutImod": 2,
            }
        )
        if roi_file is not None:
            options["RoiFile"] = roi_file
        if patches is not None:
            options["Patch"] = f"{patches} {patches}"

    # run aretomo with basic settings
    aretomo_cmd = f"{cmd} {' '.join(f'-{k} {v}' for k, v in options.items())}"

    log.info(aretomo_cmd)
    if not reconstruct:
        log.info(f'mv {xf} {warp_mdoc_basename + ".xf"}')

    if not dry_run:
        with _cd(cwd):
            try:
                proc = subprocess.run(
                    aretomo_cmd.split(), capture_output=True, check=False, cwd=cwd
                )
            finally:
                aretomolog.write_bytes(proc.stdout + proc.stderr)
                if gpu_queue is not None:
                    gpu_queue.put(gpu)
            proc.check_returncode()
        if not reconstruct:
            # move xf file so warp can see it (needs full ts name + .xf)
            shutil.move(cwd / xf, cwd / (warp_mdoc_basename + ".xf"))
    else:
        sleep(0.1)
        if gpu_queue is not None:
            gpu_queue.put(gpu)


def aretomo_batch(
    progress, tilt_series, suffix="", label="", cmd="AreTomo", gpus=None, **kwargs
):
    log = logging.getLogger("waretomo")
    if not shutil.which(cmd):
        raise FileNotFoundError(f"{cmd} is not available on the system")
    if gpus is None:
        gpus = [gpu.id for gpu in GPUtil.getGPUs()]
    if not gpus:
        raise RuntimeError("you need at least one GPU to run AreTomo")
    log.info(f"Running AreTomo in parallel on {len(gpus)} GPUs.")

    # use a queue to hold gpu ids to ensure we only run one job per gpu
    gpu_queue = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    partials = []
    for ts in tilt_series:
        partials.append(
            lambda ts=ts: _aretomo(
                input_=ts["stack" + suffix],
                rawtlt=ts["rawtlt"],
                aln=ts["aln"],
                xf=ts["xf"],
                roi_file=ts["roi"],
                output=ts["recon" + suffix],
                gpu_queue=gpu_queue,
                cmd=cmd,
                warp_mdoc_basename=ts["mdoc"].stem,
                **ts["aretomo_kwargs"],
                **kwargs,
            )
        )

    run_threaded(progress, partials, label=label, max_workers=len(gpus), **kwargs)
