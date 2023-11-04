from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import os


def setup_mmlu(session):
    # setup_mmlu
    if "MMLU_DATA_DIR" not in os.environ:
        td = TemporaryDirectory()
        tmp_path = Path(td.name)
        del td

        os.mkdir(tmp_path)
        urllib.request.urlretrieve(
            "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            tmp_path / f"mmlu.tar",
        )

        os.system(f"tar -xf {tmp_path}/mmlu.tar -C {tmp_path}")
        os.environ["MMLU_DATA_DIR"] = os.path.join(str(tmp_path), "data")
        session.__MMLU_CACHE = str(tmp_path)
    else:
        session.__MMLU_CACHE = None


def teardown_mmlu(session):
    # remove the cached images
    import shutil

    if session.__MMLU_CACHE is not None:
        shutil.rmtree(session.__MMLU_CACHE, ignore_errors=True)


def pytest_sessionstart(session):
    setup_mmlu(session)


def pytest_sessionfinish(session, exitstatus):
    teardown_mmlu(session)
