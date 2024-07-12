import os

import pytest
import torch

from cube_detection import CubeSegmentation


@pytest.fixture(scope='session')
def datadir():
    datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'image_out')
    return datadir

@pytest.fixture(scope='module')
def model_yolo():
    from roboflow import Roboflow
    api_key = os.environ.get('ROBO_KEY')
    assert api_key is not None, 'enter api key here'
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("my_detect_rubik")
    model_yolo = project.version(1).model
    return model_yolo

@pytest.fixture(scope='module')
def cube_seg():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cube_sec = CubeSegmentation(device=device)
    return cube_sec