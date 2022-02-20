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
        self.lowermask = None
        self.filename = filename

        self.import_surface()
        self.find_lengths()
        self.def_cllcpts()
        self.def_srcpts()
        self.show()

    def import_surface(self):
        data_folder = Path("surfaces/")
        file_to_open = data_folder / self.filename
        data = np.genfromtxt(file_to_open, delimiter=',', skip_header=1)
        self.points = np.concatenate((data, [data[0, :]]), axis=0)  # x in the first column, z in the second column

        # Mask telling us what is the lower surface, important for collocation pt definition
        # Simple negative points in Z mean lower surface, this only works for symmetrical airfoils!!!!!
        self.lowermask = self.points[1:, 1] < 0
        self.lowermask[-1] = True  # The last point must be on the lower surface

    def find_lengths(self):
        l = np.sqrt( np.sum( np.diff(self.points, axis=0) ** 2, axis=1 ) )
        self.lengths = l

    def def_cllcpts(self):
        # Defining the collocation points at 3/4 the length of the panel for upper surface,
        # Defining the collocation points at 1/4 the length of the panel for lower surface

        x_cl = self.points[:-1, 0] + (3/4) * np.diff(self.points[:, 0])
        z_cl = self.points[:-1, 1] + (3/4) * np.diff(self.points[:, 1])
        x_cu = self.points[:-1, 0] + (1/4) * np.diff(self.points[:, 0])
        z_cu = self.points[:-1, 1] + (1/4) * np.diff(self.points[:, 1])

        x_c = np.append(x_cu[~self.lowermask], x_cl[self.lowermask])
        z_c = np.append(z_cu[~self.lowermask], z_cl[self.lowermask])
        self.clcpts = np.stack((x_c, z_c), axis=1)
        # Problem, the panels begin at the trailing edge with x being positive, then moves counter clockwise

    def def_srcpts(self):
        # Defining the source points at 1/4 the length of the panel for upper surface
        # Defining the source points at 3/4 the length of the panel for lower surface

        x_sl = self.points[:-1, 0] + (1 / 4) * np.diff(self.points[:, 0])
        z_sl = self.points[:-1, 1] + (1 / 4) * np.diff(self.points[:, 1])
        x_su = self.points[:-1, 0] + (3 / 4) * np.diff(self.points[:, 0])
        z_su = self.points[:-1, 1] + (3 / 4) * np.diff(self.points[:, 1])

        x_s = np.append(x_su[~self.lowermask], x_sl[self.lowermask])
        z_s = np.append(z_su[~self.lowermask], z_sl[self.lowermask])
        self.srcpts = np.stack((x_s, z_s), axis=1)

    def show(self):
        plt.figure()
        plt.plot(self.points[:, 0], self.points[:, 1], linestyle='', marker='.', label="Nodes")
        plt.plot(self.srcpts[:, 0], self.srcpts[:, 1], linestyle='', marker='x', label="Sources")
        plt.plot(self.clcpts[:, 0], self.clcpts[:, 1], linestyle='', marker='*', label="Collocations")
        plt.legend()
        plt.show()
# Getting surface points
circle4 = surface("circle4.csv")
circle8 = surface("circle8.csv")
circle32 = surface("circle32.csv")
naca0012 = surface("naca0012.csv")

