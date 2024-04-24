import logging
import os
import subprocess
from concurrent import futures


def run_threaded(
    progress,
    partials,
    label="",
    max_workers=None,
    dry_run=False,
    **kwargs,
):
    max_workers = max_workers or min(32, os.cpu_count() + 4)  # see concurrent docs

    log = logging.getLogger("waretomo")

    with futures.ThreadPoolExecutor(max_workers) as executor:
        main_task = progress.add_task(f"{label}...", total=len(partials))

        jobs = []
        for fn in partials:
            job = executor.submit(fn)
            jobs.append(job)

        tasks = []
        for i, _thread in enumerate(executor._threads):
            tasks.append(progress.add_task(f"thread #{i}...", start=False))

        exist = 0
        errors = []
        for job in futures.as_completed(jobs):
            try:
                job.result()
            except FileExistsError:
                exist += 1
            except subprocess.CalledProcessError as e:
                errors.append(e)
                log.warning("Subprocess failed with:")
            progress.update(main_task, advance=1)

        for t in tasks:
            progress.update(t, total=1, completed=1, visible=False)

        if exist:
            log.warn(f"{label}: {exist} files already exist and were not overwritten")

        if errors:
            log.error(f"{label}: {len(errors)} commands have failed:")
            for err in errors:
                log.error(
                    f'{" ".join(err.cmd)} ' f"failed with:\n{err.stderr.decode()}"
                )
