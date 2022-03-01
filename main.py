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

    dZ = mZj - mZi  # Zij matrix (Contains all displacements from source i to collocation j
    dX = mXj - mXi  # Xij matrix
    R = np.sqrt( dX**2 + dZ**2 )  # Finding the distance from ith source to jth collocation point

    Ls, Lj = np.meshgrid(surface.lengths, surface.lengths)  # Length dSi in matrix form
    EPs, EPj = np.meshgrid(surface.epsilon, surface.epsilon)  # Incidence angles epsilon at source i

    #  Finding local displacements from rotation of global displacements by epsilon
    XQ = np.multiply(dX, np.cos(EPs)) - np.multiply(dZ, np.sin(EPs))
    ZQ = np.multiply(dX, np.sin(EPs)) + np.multiply(dZ, np.cos(EPs))

    #  Finding all local velocities from the local displacements XQ, ZQ
    UQ = (-4*np.pi)**-1 * np.log( ((XQ - (0.5*Ls))**2 + ZQ**2)/((XQ + (0.5*Ls))**2 + ZQ**2) )
    VQ = (-2*np.pi)**-1 * ( np.arctan((XQ-(0.5*Ls))/ZQ) - np.arctan((XQ+(0.5*Ls))/ZQ) )

    #  Rotating all local velocities back into the global coordinate system
    U = np.multiply(UQ, np.cos(EPs)) + np.multiply(VQ, np.sin(EPs))
    V = -1*np.multiply(UQ, np.sin(EPs)) + np.multiply(VQ, np.cos(EPs))

    #  Matrixizing the normal vector components
    Nxi, Nxj = np.meshgrid(surface.n_x, surface.n_x)
    Nzi, Nzj = np.meshgrid(surface.n_z, surface.n_z)

    ## Building D Matrix (n dot V)ij = nj dot Vij
    D = np.multiply(Nxi, U) + np.multiply(Nzi, V)

    ## Building B Matrix
    Ux_inf = op_point.U_inf * np.cos(op_point.AoA)
    Uz_inf = op_point.U_inf * np.sin(op_point.AoA)

    B = np.transpose([surface.n_x * Ux_inf + surface.n_z * Uz_inf])
    return D, B
def panel_vel(sigma, length, x, z):
    Vx = -sigma * (4*np.pi)**-1 * np.log( ((x-0.5*length)**2+z**2)/((x+0.5*length)**2+z**2) )
    Vz = -sigma * (2*np.pi)**-1 * (np.arctan((x-(0.5*length))/z) - np.arctan((x+(0.5*length))/z))
    return Vx, Vz

class surface:
    def __init__(self, filename):
        self.points = None
        self.srcpts = None
        self.clcpts = None
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

        # Epsilon needs 180 degree rotation when on bottom surface
        self.epsilon = np.arctan(self.n_x / self.n_z)
        self.epsilon = np.append(self.epsilon[~self.lowermask], self.epsilon[self.lowermask] - np.pi)
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
U_inf = 1  # in m/s
alpha = 0  # AoA in radians
operating_point = op_point(U_inf, alpha)

#print(panel_vel(10*2*np.pi, 2*2, -1, -1))
#print(panel_vel(10*2*np.pi, 2*2, 9, -1))

# Getting surface points
circle4 = surface("circle4.csv")
circle8 = surface("circle8.csv")
circle16 = surface("circle16.csv")
circle32 = surface("circle32.csv")
naca0012 = surface("naca0012.csv")

# Constructing matrix to solve
D, B = build_matrices(circle32, operating_point)
# Solving for Q's, or source strengths
Q = np.linalg.solve(D, B)
print(D)
print(B)
print(Q)

plt.plot(Q, marker='.', linestyle='')
plt.show()

def panel_vel(sigma, length, x, z):
    Vx = -sigma * (4*np.pi)**-1 * np.log( ((x-0.5*length)**2+z**2)/((x+0.5*length)**2+z**2) )
    Vz = -sigma * (2*np.pi)**-1 * (np.arctan((x-0.5*length)/z) - np.arctan((x+0.5*length)/z))
    return Vx, Vz