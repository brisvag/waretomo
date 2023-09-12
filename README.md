# waretomo

[![License](https://img.shields.io/pypi/l/waretomo.svg?color=green)](https://github.com/brisvag/waretomo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/waretomo.svg?color=green)](https://pypi.org/project/waretomo)
[![Python Version](https://img.shields.io/pypi/pyversions/waretomo.svg?color=green)](https://python.org)
[![CI](https://github.com/brisvag/waretomo/actions/workflows/ci.yml/badge.svg)](https://github.com/brisvag/waretomo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/brisvag/waretomo/branch/main/graph/badge.svg)](https://codecov.io/gh/brisvag/waretomo)

Overengineered batch processing script for tomography data with Warp and aretomo.

_NOTE: This was tested with AreTomo 1.3.4, i's recommended to use the same version._

# Installation and usage

```bash
pip install waretomo
```

```bash
waretomo -h
```

Every time you run waretomo, a log will be appended to an `waretomo.log` file in the output directory.

# Walkthrough

Here's a short summary of how I recommend using this script:

- pre-process your dataset in Warp, up to `Create stack for imod`. Make sure to deselect bad tilts from the warp interface; they will be skipped in subsequent steps
- run `waretomo` in `dry-run` mode (`-d`), to make sure everything is set up correctly. Pass any other options you'd like (see `waretomo -h` for a complete list and shorthands):
```bash
waretomo . --mdoc-dir ./mdocs  -b 8 -t 800 -d
```
- follow error messages if any arise (you may have to provide the path to the `AreTomo` executable with `--aretomo`, for example)
- once the above command works, it will give you a summary of inputs and of the upcoming pipeline. It's time to try to run it on a single tomogram, using `-j` (`--just`). _Note that the name to provide to `-j` is the tomogram name given at the top of the mdoc file (`ImageFile = <something>`), and not the input file (this is because warp actually uses this value for all subsequent outputs)._
```bash
waretomo . --mdoc-dir ./mdocs  -b 8 -t 800 -j tiltseries_23.mrc
```
- this will run the full pipeline on that tomogram. Keep an eye out for error messages, they might give you tips for solving them.
- if everything works, you will now have a few outputs to check out:
    - `./waretomo_processing/tiltseries_23.mrc`: raw aretomo reconstruction.
    - `./waretomo_processing/denoised/tiltseries_23.mrc`: same as above, denoised with topaz for better annotation
    - `<mdoc-dir>/mdoc_tilted/tiltseries_23.mrc.mdoc`: mdoc file updated with skipped tilt and with adjusted tilt angles from aretomo's `TiltAlign` option (e.g: to align lamellae to the XY plane)
    - `./waretomo_processing/tiltseries_23.xf`: alignment metadata, used by warp together with above mdocs for reconstruction.
- check out the reconstructions and make sure everything looks as you want. If anything is wrong, adjust parameters as you see fit. You can also run only parts of the script by using the `--start-from` and `--stop-at` options (both are inclusive). Use `-f` if you want to overwrite existing outputs.
- once you're happy, remove the `-j` option to process the full dataset.

At this point, you're ready to go back to Warp. Here, you can simply `import tilt series from IMOD`. Don't forget to provide the `mdoc_tilted` directory instead of the original mdocs. Set `waretomo_processing` as the `Root folder with IMOD processing results`, and Warp will find the `xf` files located there. Provide the pixel size of the binned aretomo reconstructions (find out with e.g: `header waretomo_processing/tiltseries_23.mrc.mrc`). You might have to provide the dose per tilt, depending on the origin/correctness of your mdocs.

You can now proceed with reconstructing with Warp, use M, and so on, while having matching AreTomo reconstructions with their local patch alignments and subsequent denoising to maximize readability for picking and annotation.
