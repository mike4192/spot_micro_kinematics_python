#!/usr/bin/env python

""" Functions for computing 3D rotation, transformation, and homogenenous transformation matrices """

import numpy as np
from math import cos, sin


def rotx(ang):
    """Create a 3x3 numpy rotation matrix about the x axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the x axis
    """
    rotxMatrix = np.array(
        [   [1,             0,              0],
            [0,      cos(ang),      -sin(ang)],
            [0,      sin(ang),       cos(ang)]  ])

    return rotxMatrix


def roty(ang):
    """Create a 3x3 numpy rotation matrix about the y axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the y axis
    """
    rotyMatrix = np.array(
        [   [ cos(ang),      0,       sin(ang)],
            [       0,       1,              0],
            [-sin(ang),      0,       cos(ang)]  ])

    return rotyMatrix


def rotz(ang):
    """Create a 3x3 numpy rotation matrix about the z axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the z axis
    """
    rotzMatrix = np.array(
        [   [cos(ang),   -sin(ang),             0],
            [sin(ang),    cos(ang),             0],
            [       0,           0,             1]  ])

    return rotzMatrix

def rotxyz(x_ang,y_ang,z_ang):
    """Creates a 3x3 numpy rotation matrix from three rotations done in the order
    of x, y, and z in the local coordinate frame as it rotates.

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians

    Returns:
        The 3D rotation matrix for a x, y, z rotation
    """
    return rotx(z_ang) @ roty(y_ang) @ rotz(z_ang)


def homog_rotxyz(x_ang,y_ang,z_ang):
    """Creates a 4x4 numpy homogeneous rotation matrix from three rotations
    done in the order x, y, and z in the local coordinate frame as it rotates. This is
    the same as the output of homog_trans except with no translation

    The three columns and rows represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians

    Returns:
        The homogenous transformation matrix for a x, y, z rotation and translation
    """
    return np.block( [  [rotxyz(x_ang,y_ang,z_ang), np.array([[0],[0],[0]])], [np.array([0,0,0,1])]  ]  )

def homog_transxyz(x,y,z):
    """Creates a 4x4 numpy linear transformation matrix

    Args:
        x: translation in x
        y: translation in y
        z: translation in z

    Returns:
        4x4 numpy array for a linear translation on a 4x1 vector
    """

    return np.block([ [np.eye(3,3), np.array([[x],[y],[z]]) ],  [np.array([0,0,0,1])]  ]   )

def homog_transform(x_ang,y_ang,z_ang,x_t,y_t,z_t):
    """Creates a 4x4 numpy rotation and transformation matrix from three rotations
    done in the order x, y, and z in the local coordinate frame as it rotates, then 
    a transformation in x, y, and z in that rotate coordinate frame.

    The three columns and rows represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix. The last column and three rows
    represents the translation in the rotated coordinate frame

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians
        x_t: linear translation in x
        y_t: linear translation in y
        z_t: linear translation in z

    Returns:
        The homogenous transformation matrix for a x, y, z rotation and translation
    """
    return homog_rotxyz(x_ang,y_ang,z_ang) @ homog_transxyz(x_t,y_t,z_t)

def ht_inverse(ht):
    '''Calculate the inverse of a homogeneous transformation matrix

    Args
        ht: Input 4x4 nump matrix homogeneous transformation

    Returns:
        A 4x4 numpy matrix that is the inverse of the inputted transformation
    '''

    # Invert the rotation part of the homogeneous transform

    temp_rot = ht[0:3,0:3]

    temp_vec = -1*ht[0:3,3]

    temp_rot_ht = np.block([ [temp_rot.transpose(),   np.zeros((3,1))],
                             [np.zeros((1,3))     ,         np.eye(1)] ])


    temp_vec_ht = np.eye(4)
    temp_vec_ht[0:3,3] = temp_vec

    return temp_rot_ht @ temp_vec_ht
