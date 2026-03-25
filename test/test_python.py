import numpy as np
import matplotlib.pyplot as plt
from src.vicsek_pkg_example.test2 import VicsekModel

def test_main():
    assert(True)

def test_distance_type(r=np.random.rand(3,2)):
    assert(type(VicsekModel.distances(r, 2))==type(np.array(1)))