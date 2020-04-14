"""Kinematic transformations for a quadriped robot. 

Equations from:
Şen, Muhammed Arif & Bakırcıoğlu, Veli & Kalyoncu, Mete. (2017). 
Inverse Kinematic Analysis Of A Quadruped Robot. 
International Journal of Scientific & Technology Research. 6. 
"""

from . import transformations
from math import pi, cos, sin
import numpy as np

def t_rightback(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the rightback leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the right back leg
    '''
    temp_homog_transf = np.block( [ [ transformations.roty(pi/2), np.array([[-l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_m @ temp_homog_transf

def t_rightfront(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the rightfront leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the right front leg
    '''
    temp_homog_transf = np.block( [ [ transformations.roty(pi/2), np.array([[l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_m @ temp_homog_transf

def t_leftfront(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the left front leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the left front leg
    '''
    temp_homog_transf = np.block( [ [ transformations.roty(-pi/2), np.array([[l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_m @ temp_homog_transf

def t_leftback(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the left back leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the left back leg
    '''
    temp_homog_transf = np.block( [ [ transformations.roty(-pi/2), np.array([[-l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_m @ temp_homog_transf


def t_0_to_1(theta1,l1):
    '''Create the homogeneous transformation matrix for joint 0 to 1 for a quadriped leg.

    Args:
        theta1: Rotation angle in radians of the hip joint
        l1: Length of the hip joint link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 0 to 1
    '''
    # I believe there is a typo in the paper. The paper lists this matrix as:
    # 
    # t =    [ cos(theta1)     -sin(theta1)    0        -l1*cos(theta1);
    #          sin(theta1)     -cos(theta1)    0        -l1*sin(theta1);
    #                    1                0    0                      0;
    #                    0                0    0                      1;]
    # 
    # However I believe index [2],[0] should be zero, and index [2],[2] should be 1 instead.
    # If not, then the rotated z axis disapears?? And from the diagram, it appears the transformed
    # axis's z axis is the same as the original. So I think the matrix should be:
    # 
    # t =    [ cos(theta1)     -sin(theta1)    0        -l1*cos(theta1);
    #          sin(theta1)     -cos(theta1)    0        -l1*sin(theta1);
    #                    0                0    1                      0;d
    #                    0                0    0                      1;]
    
    t_01 = np.block( [ [ transformations.rotz(theta1), np.array([[-l1*cos(theta1)],[-l1*sin(theta1)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_01


def t_1_to_2():
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.

    Args:
        None

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 1 to 2
    '''
    # I believe there is a typo in the paper. The paper lists this matrix as:
    # 
    # t =    [           0                0        -1                      0;
    #                   -1                0         0                      0;
    #                    0                0         1                      0;
    #                    0                0         0                      1;]
    # 
    # However I believe index [1],[2] should be 1, and index [2],[2] should be 0.
    # If not, then the rotated y axis disapears?? So I think the matrix should be:
    # 
    # t =    [           0                0        -1                      0;
    #                   -1                0         0                      0;
    #                    0                1         0                      0;
    #                    0                0         0                      1;]
    # 
    t_12 = np.array([[ 0,  0, -1,  0],
                     [-1,  0,  0,  0],
                     [ 0,  1,  0,  0],
                     [ 0,  0,  0,  1]])
    return t_12

def t_2_to_3(theta2,l2):
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.

    Args:
        theta2: Rotation angle in radians of the leg joint
        l2: Length of the upper leg link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 2 to 3
    '''

    t_23 = np.block( [ [ transformations.rotz(theta2), np.array([[l2*cos(theta2)],[l2*sin(theta2)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_23

def t_3_to_4(theta3,l3):
    '''Create the homogeneous transformation matrix for joint 3 to 4 for a quadriped leg.

    Args:
        theta3: Rotation angle in radians of the knee joint
        l3: Length of the lower leg link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 3 to 4
    '''

    t_34 = np.block( [ [ transformations.rotz(theta3), np.array([[l3*cos(theta3)],[l3*sin(theta3)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_34

def t_0_to_4(theta1, theta2, theta3, l1, l2, l3):
    '''Create the homogeneous transformation matrix from joint 0 to 4 of a quadriped leg

    Args:
        theta1: Rotation angle in radians of joint 1
        theta2: Rotation angle in radians of joint 2
        theta3: Rotation angle in radians of joint 3
        l1: Length of leg link 1, the hip length
        l2: Length of leg link 2, the uppper leg length
        l3: Length of leg link 3, the lower leg
    
    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 0 to 4
    '''
    return t_0_to_1(theta1,l1) @ t_1_to_2() @ t_2_to_3(theta2,l2) @ t_3_to_4(theta3,l3)