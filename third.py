## Source Panel Method for 2D Airfoil Analysis  ##
# Author: William Kemp

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


plt.close("all")
def build_matrices(surface, op_point):
    x_clc = surface.clcpts[:, 0]
    x_src = surface.srcpts[:, 0]
    mXj, mXi = np.meshgrid(x_clc, x_src)

    z_clc = surface.clcpts[:, 1]
    z_src = surface.srcpts[:, 1]
    mZj, mZi = np.meshgrid(z_clc, z_src)
    
    theta = np.arctan2(surface.srcpts[:, 1], surface.srcpts[:, 0]) * (180/np.pi)

    dZ = mZj - mZi  # Zij matrix (Contains all displacements from source i to collocation j
    dX = mXj - mXi  # Xij matrix
    
    dxtest = x_clc - x_src[1]
    
    R = np.sqrt( np.square(dX) + np.square(dZ) )  # Finding the distance from ith source to jth collocation point

    Ls, Lj = np.meshgrid(surface.lengths, surface.lengths)  # Length dSi in matrix form
    EPc, EPs = np.meshgrid(surface.epsilon, surface.epsilon)  # Incidence angles epsilon at source i

    #  Finding local displacements from rotation of global displacements by epsilon
    XQ = np.multiply(dX, np.cos(EPs)) - np.multiply(dZ, np.sin(EPs))
    ZQ = np.multiply(dX, np.sin(EPs)) + np.multiply(dZ, np.cos(EPs))
    

    #  Finding all local velocities from the local displacements XQ, ZQ
    UQ = (-2)**-1 * np.log( np.divide( (np.square(XQ + (0.5*Lj)) + np.square(ZQ)), (np.square(XQ - (0.5*Lj)) + np.square(ZQ)) ) )
    VQ = (-1)**-1 * ( np.arctan((XQ+(0.5*Lj))/ZQ) - np.arctan((XQ-(0.5*Lj))/ZQ) )
    
    # Checking for panels that are the same
    VQ = np.where(np.logical_and(np.multiply(ZQ,ZQ) < 1e-10, np.multiply(XQ,XQ) < 1e-10), np.pi, VQ)
    UQ = np.where(np.logical_and(np.multiply(ZQ,ZQ) < 1e-10, np.multiply(XQ,XQ) < 1e-10), 0, UQ)

    #  Rotating all local velocities back into the global coordinate system
    U = np.multiply(UQ, np.cos(EPs)) + np.multiply(VQ, np.sin(EPs))
    V = np.multiply(-1*UQ, np.sin(EPs)) + np.multiply(VQ, np.cos(EPs))

    #  Matrixizing the normal vector components
    Nxi, Nxj = np.meshgrid(surface.n_x, surface.n_x)
    Nzi, Nzj = np.meshgrid(surface.n_z, surface.n_z)

    ## Building D Matrix (n dot V)ij = nj dot Vij
    D = np.multiply(Nxj, U) + np.multiply(Nzj, V)

    ## Building B Matrix
    Ux_inf = -1*op_point.U_inf * np.cos(op_point.AoA)
    Uz_inf = -1*op_point.U_inf * np.sin(op_point.AoA)

    B = np.transpose([np.multiply(surface.n_x, Ux_inf) + np.multiply(surface.n_z, Uz_inf)])
    Q = np.linalg.solve(D, B)
    
    # Finding the surface velocities using U, V matrices with new Q values
    u_act = np.dot(-U, Q) + np.ones(np.shape(Q))*Ux_inf
    v_act = np.dot(-V, Q) + np.ones(np.shape(Q))*Uz_inf
    
    Cp = 1 - (op_point.U_inf**-2 * (np.square(u_act) + np.square(v_act)))
    plt.figure()
    plt.xlabel("Distance from TE to LE")
    plt.ylabel("Coefficient of Pressure (Cp)")
    plt.plot(Cp)
    plt.show()
    
    ## Finding Streamlines
    
    if surface.filename == "naca0012.csv":
        xv = np.linspace(-0.25, 1.25, 500)
        zv = np.linspace(-0.2, 0.2, 500)
    else:
        xv = np.linspace(-8, 8, 500)
        zv = np.linspace(-8, 8, 500)
    
    Xs, Zs = np.meshgrid(xv, zv)
    Ug = np.zeros(np.shape(Xs))
    Vg = np.zeros(np.shape(Zs))   
    
    for i, x in enumerate(xv):
        for j, z in enumerate(zv):
            # Getting distance from all source pts to point of interest
            dX = x - surface.srcpts[:,0]
            dZ = z - surface.srcpts[:,1]
            # Rotating each distance by the source angle epsilon
            dXq = np.multiply(dX, np.cos(surface.epsilon)) - np.multiply(dZ, np.sin(surface.epsilon))
            dZq = np.multiply(dX, np.sin(surface.epsilon)) + np.multiply(dZ, np.cos(surface.epsilon))
            # Determining velocity contribution
            # This is where its going wrong!!!!!
            k1 = np.log( np.divide( (np.square(dXq + (0.5*surface.lengths)) + np.square(dZq)), (np.square(dXq - (0.5*surface.lengths)) + np.square(dZq))))
            k1 = np.reshape(k1, np.shape(Q))
            k2 = np.arctan((dXq+(0.5*surface.lengths))/dZq) - np.arctan((dXq-(0.5*surface.lengths))/dZq)
            k2 = np.reshape(k2, np.shape(Q))
            
            Uq = (-2)**-1 * np.multiply(Q, k1)
            Vq = (-1)**-1 * np.multiply(Q, k2)
            
            # Rotating Back
            ut = np.multiply(Uq, np.reshape(np.cos(surface.epsilon), np.shape(Uq))) + np.multiply(Vq, np.reshape(np.sin(surface.epsilon), np.shape(Vq)))
            vt = np.multiply(-1*Uq, np.reshape(np.sin(surface.epsilon), np.shape(Uq))) + np.multiply(Vq, np.reshape(np.cos(surface.epsilon), np.shape(Vq)))
            # Adding all the elements
            ut = np.sum(ut)
            vt = np.sum(vt)
            # Storing in the matrix
            Ug[j,i] = -ut - Ux_inf
            Vg[j,i] = -vt - Uz_inf
    
    plt.figure()
    plt.streamplot(Xs, Zs, Ug, Vg, density=3)
    plt.scatter(surface.clcpts[:,0], surface.clcpts[:,1], c='r')
    plt.show()
    return Q, Cp, theta

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
    def import_surface(self):
        data_folder = Path("surfaces/")
        file_to_open = data_folder / self.filename
        data = np.genfromtxt(file_to_open, delimiter=',', skip_header=1)
        
        if self.filename != "naca0012.csv":
            self.points = np.concatenate((data, [data[0, :]]), axis=0)  # x in the first column, z in the second column
        else:
            self.points = data

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
        self.epsilon = np.arctan2(self.n_x,self.n_z)
        
    def find_lengths(self):
        l = np.sqrt( np.sum( np.diff(self.points, axis=0) ** 2, axis=1 ) )
        self.lengths = l
    def def_cllcpts(self):
        # Defining the collocation points at 3/4 the length of the panel for upper surface,
        # Defining the collocation points at 1/4 the length of the panel for lower surface

        x_cl = self.points[:-1, 0] + (1/2) * np.diff(self.points[:, 0])
        z_cl = self.points[:-1, 1] + (1/2) * np.diff(self.points[:, 1])
        x_cu = self.points[:-1, 0] + (1/2) * np.diff(self.points[:, 0])
        z_cu = self.points[:-1, 1] + (1/2) * np.diff(self.points[:, 1])

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
        self.AoA = AoA * (np.pi/180) # Angle of attack in radians

# Defining FreeStream Condition
U_inf = 1  # in m/s
alpha = 0  # AoA in radians
operating_point_0 = op_point(U_inf, 0)
operating_point_5 = op_point(U_inf, 5)
operating_point_10 = op_point(U_inf, 10)
operating_point_15 = op_point(U_inf, 15)

#print(panel_vel(10*2*np.pi, 2*2, -1, -1))
#print(panel_vel(10*2*np.pi, 2*2, 9, -1))

# Getting surface points
circle4 = surface("circle4.csv")
circle8 = surface("circle8.csv")
circle16 = surface("circle16.csv")
circle32 = surface("circle32.csv")
naca = surface("naca0012.csv")

# Constructing matrix to solve
# Q, Cp_8, theta8 = build_matrices(circle8, operating_point_0)
# Q, Cp_16, theta16 = build_matrices(circle16, operating_point_0)
# Q, Cp_32, theta32 = build_matrices(circle32, operating_point_0)

# Analysitcal
# thetaa = np.linspace(-np.pi, np.pi, 100)
# Cpa = 1 - 4*np.sin(thetaa)**2
# plt.figure(dpi=150, figsize=(3,8))
# plt.plot(theta8, Cp_8, label="8 Panel Cylinder", marker='v', linestyle='')
# plt.plot(theta16, Cp_16, label="16 Panel Cylinder",  marker='^', linestyle='')
# plt.plot(theta32, Cp_32, label="32 Panel Cylinder", marker='o', linestyle='')
# plt.plot(thetaa * (180/np.pi), Cpa, label = "Analytical Solution")
# plt.xlabel("Theta")
# plt.ylabel("Coefficient of Pressure (Cp)")
# plt.legend()
# plt.show()

Q, Cp0, theta0 = build_matrices(naca, operating_point_0)
Q, Cp5, theta5 = build_matrices(naca, operating_point_5)
Q, Cp10, theta10 = build_matrices(naca, operating_point_10)
Q, Cp15, theta15 = build_matrices(naca, operating_point_15)

plt.figure(dpi=150, figsize=(5,3))
plt.gca().invert_yaxis()
plt.plot(Cp0, label="0 Degree AoA", marker='v', linestyle='')
plt.plot(Cp5, label="5 Degree AoA",  marker='^', linestyle='')
plt.plot(Cp10, label="10 Degree AoA", marker='o', linestyle='')
plt.plot(Cp15, label="15 Degree AoA", marker='x', linestyle='')
plt.xlabel("Fraction Along Chord (x/c)")
plt.ylabel("Coefficient of Pressure (Cp)")
plt.legend()
plt.show()