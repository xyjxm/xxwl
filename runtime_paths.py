import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
RTMODELS_DIR = os.path.join(PROJECT_ROOT, "rtmodels")


def configure(change_cwd=True):
    """Use the bundled Quanser libraries without requiring global env vars."""
    if os.path.isdir(LOCAL_PYTHON_DIR) and LOCAL_PYTHON_DIR not in sys.path:
        sys.path.insert(0, LOCAL_PYTHON_DIR)

    pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [path for path in pythonpath.split(os.pathsep) if path]
    if LOCAL_PYTHON_DIR not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([LOCAL_PYTHON_DIR] + paths)

    os.environ["QAL_DIR"] = PROJECT_ROOT
    os.environ["RTMODELS_DIR"] = RTMODELS_DIR
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, ".ultralytics"))
    os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

    if change_cwd:
        os.chdir(PROJECT_ROOT)

    return PROJECT_ROOT
