import os
from pyecca.replay import ULogReplay
from pyecca import uros

script_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(script_dir, "data")


def test_replay():
    core = uros.Core()
    ULogReplay(core, os.path.join(data_dir, "19_01_20.ulg"))
    core.run()
