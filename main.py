## Source Panel Method for 2D Airfoil Analysis  ##
# Author: William Kemp

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def build_matrices(surface, op_point):
    x_clc = surface.clcpts[:, 0]
    x_src = surface.srcpts[:, 0]
    mXj, mXi = np.meshgrid(x_clc, x_src)

    z_clc = surface.clcpts[:, 1]
    z_src = surface.srcpts[:, 1]
    mZj, mZi = np.meshgrid(z_clc, z_src)

    dZ = mZj - mZi
    dX = mXj - mXi
    R = np.sqrt( dX**2 + dZ**2 )  # Finding the distance from ith source to jth collocation point
    print(R)

    Theta = np.arctan(dZ / dX)
    print(Theta)

    Qx = (2*np.pi*R)**-1 * np.cos(Theta)
    Qz = (2*np.pi*R)**-1 * np.sin(Theta)

    Nx, no = np.meshgrid(surface.n_x, surface.n_x)
    Nz, no = np.meshgrid(surface.n_z, surface.n_z)

    ## Building D Matrix
    D = Nx*Qx + Nz*Qz

    ## Building B Matrix
    Ux_inf = op_point.U_inf * np.cos(op_point.AoA)
    Uz_inf = op_point.U_inf * np.sin(op_point.AoA)

    B = np.transpose(surface.n_x * Ux_inf + surface.n_z * Uz_inf)
    return D, B

class surface:
    def __init__(self, filename):
        self.points = None
        self.lengths = None
        self.epsilon = None  # Angle between x axis and panel from first point to second
        self.lowermask = None
        self.filename = filename

        self.import_surface()
        self.def_cllcpts()
        self.def_srcpts()
        self.show()

    def import_surface(self):
        data_folder = Path("surfaces/")
        file_to_open = data_folder / self.filename
        data = np.genfromtxt(file_to_open, delimiter=',', skip_header=1)
        self.points = np.concatenate((data, [data[0, :]]), axis=0)  # x in the first column, z in the second column

        self.xdiffs = np.diff(self.points[:, 0])
        self.zdiffs = np.diff(self.points[:, 1])
        self.find_lengths()
        # Mask telling us what is the lower surface, important for collocation pt definition
        # Simple negative points in Z mean lower surface, this only works for symmetrical airfoils!!!!!
        self.lowermask = self.points[1:, 1] < 0
        self.lowermask[-1] = True  # The last point must be on the lower surface

        # Finding the tangential vector
        t_x = self.xdiffs / self.lengths
        t_z = self.zdiffs / self.lengths

        # Finding the normal vector
        self.n_x = t_z
        self.n_z = -t_x

        self.epsilon = np.arctan(self.n_x / self.n_z)

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

        x_sl = self.points[:-1, 0] + (1 / 2) * np.diff(self.points[:, 0])
        z_sl = self.points[:-1, 1] + (1 / 2) * np.diff(self.points[:, 1])
        x_su = self.points[:-1, 0] + (1 / 2) * np.diff(self.points[:, 0])
        z_su = self.points[:-1, 1] + (1 / 2) * np.diff(self.points[:, 1])

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

class op_point:

    def __init__(self, U_inf, AoA):
        self.U_inf = U_inf  # Freestream velocity in meters per second
        self.AoA = AoA  # Angle of attack in radians

# Defining FreeStream Condition

U_inf = 10  # in m/s
alpha = 0  # AoA in radians

operating_point = op_point(U_inf, alpha)

# Getting surface points
circle4 = surface("circle4.csv")
circle8 = surface("circle8.csv")
circle32 = surface("circle32.csv")
naca0012 = surface("naca0012.csv")

# Constructing matrix to solve
D, B = build_matrices(circle4, operating_point)
print(D)
print(B)
# Solving for Q's, or source strengths
Q = np.linalg.solve(D, B)
print(Q)