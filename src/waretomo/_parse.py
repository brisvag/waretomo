import logging
from pathlib import Path, PureWindowsPath
from xml.etree import ElementTree

from mdocfile.data_models import Mdoc


def parse_data(
    progress,
    warp_dir,
    mdoc_dir,
    output_dir,
    roi_dir,
    just=(),
    exclude=(),
    train=False,
    dose=None,
):
    log = logging.getLogger("waretomo")

    imod_dir = warp_dir / "imod"
    if not imod_dir.exists():
        raise FileNotFoundError("warp directory does not have an `imod` subdirectory")

    if just:
        mdocs = [
            p
            for ts_name in just
            if (p := (Path(mdoc_dir) / (ts_name + ".mdoc"))).exists()
        ]
    else:
        mdocs = sorted(Path(mdoc_dir).glob("*.mdoc"))

    if not mdocs:
        raise FileNotFoundError("could not find any mdoc files")

    odd_dir = warp_dir / "average" / "odd"
    even_dir = warp_dir / "average" / "even"

    tilt_series = []
    tilt_series_excluded = []
    tilt_series_unprocessed = []

    for mdoc_file in progress.track(mdocs, description="Reading mdocs..."):
        log.info(f"Parsing {mdoc_file}.")
        mdoc = Mdoc.from_file(mdoc_file)

        # warp uses mdoc name in many places, but some times the ts_name is important
        mdoc_name = mdoc_file.stem
        ts_name = mdoc.global_data.ImageFile.stem
        stack = imod_dir / mdoc_name / (mdoc_name + ".st")

        if ts_name in exclude or mdoc_name in exclude:
            tilt_series_excluded.append(ts_name)
            continue
        # skip if not preprocessed in warp
        if not stack.exists():
            tilt_series_unprocessed.append(ts_name)
            continue

        # extract even/odd paths
        tilts = [
            warp_dir / PureWindowsPath(tilt.SubFramePath).name
            for tilt in mdoc.section_data
        ]
        skipped_tilts = []
        odd = []
        even = []
        valid_xml = None
        for i, tilt in enumerate(tilts):
            if not tilt.exists():
                log.warning(
                    f"{tilt.name} is listed in an mdoc file, "
                    "but the file does not exists. "
                    "The tilt will be skipped, "
                    "but you may want to check your data."
                )
                skipped_tilts.append(i)
                continue

            xml = ElementTree.parse(tilt.with_suffix(".xml")).getroot()
            if xml.attrib["UnselectManual"] == "True":
                skipped_tilts.append(i)
            else:
                valid_xml = xml
                odd.append(odd_dir / (tilt.stem + ".mrc"))
                even.append(even_dir / (tilt.stem + ".mrc"))

        if valid_xml is None:
            tilt_series_unprocessed.append(ts_name)
            continue

        if train:
            for img in odd + even:
                if not img.exists():
                    raise FileNotFoundError(img)

        # extract metadata from warp xmls
        # (we assume the last xml has the same data as the others)
        for param in valid_xml.find("OptionsCTF"):
            if param.get("Name") == "BinTimes":
                binning = float(param.get("Value"))
            elif param.get("Name") == "Voltage":
                kv = int(param.get("Value"))
            elif param.get("Name") == "Cs":
                cs = float(param.get("Value"))
        for param in xml.find("CTF"):
            if param.get("Name") == "Defocus":
                defocus = (
                    float(param.get("Value")) * 1e4
                )  # defocus for aretomo is in Angstrom

        if roi_dir is not None:
            roi_files = list(roi_dir.glob(f"{ts_name}*"))
            if len(roi_files) == 1:
                roi_file = roi_files[0]
            else:
                roi_file = None
        else:
            roi_file = None

        dose = mdoc.section_data[0].ExposureDose if dose is None else dose
        if not dose:
            log.error("Exposure dose not present in mdoc! Setting to 0.")
        px_size_raw = mdoc.section_data[0].PixelSpacing
        if not px_size_raw:
            log.error("Pixel spacing not present in mdoc! Setting to 1.")

        # aretomo being weird about paths and names splitting needs extra care...
        ts_aligned = ts_name.split(".")[0] + "_aligned"
        alignment_result_dir = output_dir / (ts_aligned + ".st_Imod")

        tilt_series.append(
            {
                "name": ts_name,
                "mdoc": mdoc_file,
                "stack": stack,
                "rawtlt": stack.with_suffix(".rawtlt"),
                "xf": alignment_result_dir / (ts_aligned + ".xf"),
                "tlt": alignment_result_dir / (ts_aligned + ".tlt"),
                "aln": output_dir / (mdoc_name + ".st.aln"),
                "skipped_tilts": skipped_tilts,
                "roi": roi_file,
                "odd": odd,
                "even": even,
                "stack_odd": output_dir / (ts_name + "_odd.st"),
                "stack_even": output_dir / (ts_name + "_even.st"),
                "recon_odd": output_dir / "odd" / (ts_name + ".mrc"),
                "recon_even": output_dir / "even" / (ts_name + ".mrc"),
                "recon": output_dir / (ts_name + ".mrc"),
                "aretomo_kwargs": {
                    "dose": dose,
                    "px_size": px_size_raw * 2**binning,
                    "cs": cs,
                    "kv": kv,
                    "defocus": defocus,
                },
            }
        )

    return tilt_series, tilt_series_excluded, tilt_series_unprocessed
