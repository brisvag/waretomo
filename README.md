# waretomo

[![License](https://img.shields.io/pypi/l/waretomo.svg?color=green)](https://github.com/brisvag/waretomo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/waretomo.svg?color=green)](https://pypi.org/project/waretomo)
[![Python Version](https://img.shields.io/pypi/pyversions/waretomo.svg?color=green)](https://python.org)
[![CI](https://github.com/brisvag/waretomo/actions/workflows/ci.yml/badge.svg)](https://github.com/brisvag/waretomo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/brisvag/waretomo/branch/main/graph/badge.svg)](https://codecov.io/gh/brisvag/waretomo)

Batch processing for tomography data with Warp and aretomo.

# Installation

```bash
pip install waretomo
```

# Usage

Assuming we're in a pre-processed Warp directory containing an `./imod/` directory with averaged and stacked tilt series, and mdoc files inside `./mdocs/`:

```
waretomo . --mdoc-dir mdocs --binning 8
```

_TIP: run first with the dry-run `-d` option as above to get a summary of all the steps that will be performed._

For detailed help:

```bash
waretomo -h
```
