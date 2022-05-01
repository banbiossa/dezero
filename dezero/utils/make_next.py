"""Look through the files in the directory, and copy the last one to the next one

"""
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_last(target_dir: Path):
    logger.info(f"Look through {target_dir}")

    all_files = os.listdir(target_dir)
    step_files = [f for f in all_files if f.startswith("step")]

    last = sorted(step_files)[-1]

    return last


def make_next(target_dir: Path):
    last = get_last(target_dir)

    last_num = int(last[4:6])
    logger.info(f"Last file is {last_num}")

    copy_from = target_dir / last
    assert copy_from.exists()
    copy_to = target_dir / f"step{last_num+1:02d}.py"

    logger.info(f"copy from {copy_from} to {copy_to}")
    shutil.copy(copy_from, copy_to)

    logger.info("All done :)")


def delete_last(target_dir: Path):
    last = get_last(target_dir)
    logger.info(f"Remove {last=}")
    os.remove(target_dir / last)


def get_target_dir():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    current = Path(__file__)
    target_dir = current.parents[1] / "steps"
    return target_dir


def delete_last_cli():
    target_dir = get_target_dir()
    delete_last(target_dir)


def make_next_cli():
    target_dir = get_target_dir()
    make_next(target_dir)


if __name__ == "__main__":
    run()
