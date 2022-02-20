## Source Panel Method for 2D Airfoil Analysis  ##
# Author: William Kemp

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

class surface:
    def __init__(self, filename):
        self.points = None
        self.lengths = None
        self.angles = None
        self.filename = filename

        self.import_surface()
        self.find_lengths()
        self.def_cllcpts()
        self.def_srcpts()

    def import_surface(self):
        data_folder = Path("surfaces/")
        file_to_open = data_folder / self.filename
        data = np.genfromtxt(file_to_open, delimiter=',', skip_header=1)
        self.points = data  # x in the first column, z in the second column

    def find_lengths(self):
        l = np.sqrt( np.sum( np.diff(self.points, axis=0) ** 2, axis=1 ) )
        self.lengths = l

    def def_cllcpts(self):
        # Defining the collocation points at 3/4 the length of the panel
        x_c = self.points[:-1, 0] + (3/4) * np.diff(self.points[:, 0])
        y_c = self.points[:-1, 1] + (3/4) * np.diff(self.points[:, 1])
        print(x_c)
        print(y_c)
        # Problem, the panels begin at the trailing edge with x being positive

    def def_srcpts(self):
        # Defining the collocation points at 3/4 the length of the panel
        x_s = self.points[:-1, 0] + (1 / 4) * np.diff(self.points[:, 0])
        y_s = self.points[:-1, 1] + (1 / 4) * np.diff(self.points[:, 1])
        print(x_s)
        print(y_s)

# Getting surface points
naca0012 = surface("circle4.csv")
# Finding panel lengths

# Finding panel normal and tangential
