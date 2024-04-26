from enum import Enum, auto

import click

import waretomo


class ProcessingStep(str, Enum):
    """Enum for step selection."""

    align = auto()
    tilt_mdocs = auto()
    reconstruct = auto()
    stack_halves = auto()
    reconstruct_halves = auto()
    denoise = auto()

    def __str__(self):
        """String."""
        return self.name


@click.command(
    name="waretomo",
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument(
    "warp_dir", type=click.Path(exists=True, dir_okay=True, resolve_path=True)
)
@click.option(
    "-m",
    "--mdoc-dir",
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    help="directory containing mdoc files [default: same as WARP_DIR]",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    help="output directory for all the processing "
    "[default: WARP_DIR/waretomo_processing]",
)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="only print some info, without running the commands.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="level of verbosity; can be passed multiple times.",
)
@click.option(
    "-j",
    "--just",
    type=str,
    multiple=True,
    help="reconstruct just this tomogram (can be passed multiple times)",
)
@click.option(
    "-e",
    "--exclude",
    type=str,
    multiple=True,
    help="exclude this tomogram from the run (can be passed multiple times)",
)
@click.option(
    "-t",
    "--sample-thickness",
    type=int,
    default=400,
    help="unbinned thickness of the SAMPLE (ice or lamella) used for alignment",
)
@click.option(
    "-z",
    "--z-thickness",
    type=int,
    default=1200,
    help="unbinned thickness of the RECONSTRUCTION.",
)
@click.option(
    "-b",
    "--binning",
    type=int,
    default=4,
    help="binning for aretomo reconstruction (relative to warp pre-processed binning)",
)
@click.option(
    "--dose",
    type=float,
    help="exposure dose (e/A^2/tilt_image). If not passed, guess from mdocs.",
)
@click.option(
    "-a", "--tilt-axis", type=float, help="starting tilt axis for AreTomo, if any"
)
@click.option(
    "-p",
    "--patches",
    type=int,
    help="number of patches for local alignment in aretomo (NxN), if any",
)
@click.option(
    "-r",
    "--roi-dir",
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    help="directory containing ROI files. "
    "Extension does not matter, but names should be same as TS.",
)
@click.option(
    "-f", "--overwrite", is_flag=True, help="overwrite any previous existing run"
)
@click.option(
    "--train", is_flag=True, default=False, help="whether to train a new denosing model"
)
@click.option(
    "--topaz-tile-size",
    type=int,
    default=64,
    help="tile size for training topaz model.",
)
@click.option(
    "--topaz-patch-size",
    type=int,
    default=64,
    help="patch size for denoising in topaz.",
)
@click.option(
    "--topaz-model",
    type=str,
    default="unet-3d-10a",
    help="topaz model for denoising. If --train was not given, this must to be the "
    "name of a pre-trained model ('unet-3d-10a', 'unet-3d-20a') or the path of a "
    "previously generated model. If instead --train was given, a new model will"
    "be generated with the given name. New models are saved inside "
    "'OUTPUT_DIR/trained_models/.",
)
@click.option(
    "--start-from",
    type=click.Choice(ProcessingStep.__members__),
    default="align",
    help="use outputs from a previous run, starting processing at this step",
)
@click.option(
    "--stop-at",
    type=click.Choice(ProcessingStep.__members__),
    default="denoise",
    help="terminate processing after this step",
)
@click.option("--aretomo", type=str, default="AreTomo", help="aretomo executable")
@click.option(
    "--gpus",
    type=str,
    help="Comma separated list of gpus to use for aretomo. Default to all.",
)
@click.option(
    "--tiltcorr/--no-tiltcorr", default=True, help="do not correct sample tilt"
)
@click.version_option(waretomo.__version__, "-V", "--version")
def cli(
    warp_dir,
    mdoc_dir,
    output_dir,
    dry_run,
    verbose,
    just,
    exclude,
    sample_thickness,
    z_thickness,
    binning,
    dose,
    tilt_axis,
    patches,
    roi_dir,
    overwrite,
    train,
    topaz_tile_size,
    topaz_patch_size,
    topaz_model,
    start_from,
    stop_at,
    aretomo,
    gpus,
    tiltcorr,
):
    """
    Run aretomo in batch on data preprocessed in warp.

    Needs to be ran after imod stacks were generated.
    Requires AreTomo>=1.3.0.

    Assumes the default Warp directory structure with generated imod stacks.
    """
    import logging
    import sys
    from datetime import datetime
    from inspect import cleandoc
    from pathlib import Path

    from rich import print
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import Progress

    from ._parse import parse_data

    logging.basicConfig(
        level=40 - max(verbose * 10, 0),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("waretomo")

    if gpus is not None:
        gpus = [int(gpu) for gpu in gpus.split(",")]

    warp_dir = Path(warp_dir)
    if mdoc_dir is None:
        mdoc_dir = warp_dir
    mdoc_dir = Path(mdoc_dir)
    if output_dir is None:
        output_dir = warp_dir / "waretomo_processing"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if roi_dir is not None:
        roi_dir = Path(roi_dir)

    pretrained_models = ("unet-3d-10a", "unet-3d-20a")
    if train:
        if topaz_model in pretrained_models:
            raise click.UsageError(
                f"Model '{topaz_model}' already exists. Provide a new name."
            )
        elif "/" in topaz_model:
            raise click.UsageError(
                f"Model '{topaz_model}' seems to be a path. "
                "Only a new name should be given."
            )
    else:
        if not Path(topaz_model).exists() and topaz_model not in pretrained_models:
            raise click.UsageError(
                f"Model '{topaz_model}' does not exist. "
                "Provide a path to an existing model "
                f"or one of {pretrained_models}."
            )

    with Progress() as progress:
        tilt_series, tilt_series_excluded, tilt_series_unprocessed = parse_data(
            progress,
            warp_dir,
            mdoc_dir=mdoc_dir,
            output_dir=output_dir,
            roi_dir=roi_dir,
            just=just,
            exclude=exclude,
            train=train,
            dose=dose,
        )

        aretomo_kwargs = {
            "cmd": aretomo,
            "tilt_axis": tilt_axis,
            "patches": patches,
            "thickness_align": sample_thickness,
            "thickness_recon": z_thickness,
            "binning": binning,
            "gpus": gpus,
            "tilt_corr": tiltcorr,
        }

        meta_kwargs = {
            "overwrite": overwrite,
            "dry_run": dry_run,
        }

        topaz_kwargs = {
            "train": train,
            "model_name": topaz_model,
            "gpus": gpus,
            "tile_size": topaz_tile_size,
            "patch_size": topaz_patch_size,
        }

        start_from = ProcessingStep[start_from]
        stop_at = ProcessingStep[stop_at]

        steps = {
            step: start_from <= val <= stop_at
            for step, val in ProcessingStep.__members__.items()
        }
        if not train:
            steps["stack_halves"] = False
            steps["reconstruct_halves"] = False

        nl = "\n"

        not_ready_log = "".join(
            f'{nl}{" " * 12}- {ts}' for ts in tilt_series_unprocessed
        )
        ready_log = "".join(f'{nl}{" " * 12}- {ts["name"]}' for ts in tilt_series)
        excluded = "".join(f'{nl}{" " * 12}- {ts}' for ts in tilt_series_excluded)
        steps_log = "".join(
            f'{nl}{" " * 12}- '
            f'[{"green" if v else "red"}]{k}[/{"green" if v else "red"}] '
            for k, v in steps.items()
        )
        opts_log = "".join(f'{nl}{" " * 12}- {k}: {v}' for k, v in meta_kwargs.items())
        aretomo_opts_log = "".join(
            f'{nl}{" " * 12}- {k}: {v}' for k, v in aretomo_kwargs.items()
        )
        topaz_log = "".join(
            f'{nl}{" " * 12}- {k}: {v}' for k, v in topaz_kwargs.items()
        )

        summary = cleandoc(
            f"""
            [bold]Warp directory[/bold]: {warp_dir}
            [bold]Mdoc directory[/bold]: {mdoc_dir}
            [bold]Tilt series - NOT READY[/bold]: {not_ready_log}
            [bold]Tilt series - READY[/bold]: {ready_log}
            [bold]Tilt series - EXCLUDED[/bold]: {excluded}
            [bold]Processing steps[/bold]: {steps_log}
            [bold]Run options[/bold]: {opts_log}
            [bold]AreTomo options[/bold]: {aretomo_opts_log}
            [bold]Topaz options[/bold]: {topaz_log}
            """
        )
        print(Panel(summary))

        ts = tilt_series[0]
        ts_name = ts["name"]
        ts_info = {k: v for k, v in ts.items() if k not in ("even", "odd")}
        ts_info = {
            k: (str(v.relative_to(warp_dir)) if isinstance(v, Path) else v)
            for k, v in ts_info.items()
            if k
        }
        ts_info.update(ts_info.pop("aretomo_kwargs"))
        ts_info = "".join(f'{nl}{" " * 12}- {k}: {v}' for k, v in ts_info.items())
        first_ts_summary = cleandoc(
            f"""
            Double check that these values make sense for {ts_name}:
            {ts_info}
            """
        )

        print(Panel(first_ts_summary))

        if not dry_run:
            with open(output_dir / "waretomo.log", "a") as f:
                print("=" * 80, file=f)
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), file=f)
                print(f'Command: {" ".join(sys.argv)}', file=f)
                print(summary, "\n", file=f)

        if steps["align"]:
            from ._aretomo import aretomo_batch

            log.info("Aligning with AreTomo...")
            aretomo_batch(
                progress,
                tilt_series,
                label="Aligning",
                **aretomo_kwargs,
                **meta_kwargs,
            )

        if steps["tilt_mdocs"]:
            if not tiltcorr:
                log.info("No need to tilt mdocs!")
            else:
                from ._fix_mdoc import tilt_mdocs_batch

                log.info("Tilting mdocs...")
                (mdoc_dir / "mdoc_tilted").mkdir(parents=True, exist_ok=True)
                tilt_mdocs_batch(
                    progress,
                    tilt_series,
                    **meta_kwargs,
                )

        if steps["reconstruct"]:
            from ._aretomo import aretomo_batch

            log.info("Reconstructing with AreTomo...")
            aretomo_batch(
                progress,
                tilt_series,
                reconstruct=True,
                label="Reconstructing",
                **aretomo_kwargs,
                **meta_kwargs,
            )

        if steps["stack_halves"]:
            from ._stack import prepare_half_stacks

            for half in ("even", "odd"):
                log.info(f"Preparing {half} stacks for denoising...")
                prepare_half_stacks(progress, tilt_series, half=half, **meta_kwargs)

        if steps["reconstruct_halves"]:
            from ._aretomo import aretomo_batch

            for half in ("even", "odd"):
                log.info(f"Reconstructing {half} tomograms for deonoising...")
                half_dir = output_dir / half
                half_dir.mkdir(parents=True, exist_ok=True)
                aretomo_batch(
                    progress,
                    tilt_series,
                    suffix=f"_{half}",
                    reconstruct=True,
                    label=f"Reconstructing {half} halves",
                    **aretomo_kwargs,
                    **meta_kwargs,
                )
                # remove leftovers from aretomo otherwise topaz dies later
                for f in half_dir.glob("*_projX?.mrc"):
                    f.unlink(missing_ok=True)
                for f in half_dir.glob("*.aretomolog"):
                    f.unlink(missing_ok=True)

        if steps["denoise"]:
            from ._topaz import topaz_batch

            log.info("Denoising tomograms...")
            outdir_denoised = output_dir / "denoised"
            outdir_denoised.mkdir(parents=True, exist_ok=True)
            topaz_batch(
                progress,
                tilt_series,
                outdir=outdir_denoised,
                even=str(output_dir / "even"),
                odd=str(output_dir / "odd"),
                **topaz_kwargs,
                **meta_kwargs,
            )
