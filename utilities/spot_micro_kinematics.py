"""Forward and inverse kinematic transformations for a quadriped robot. 

Equations from:
Şen, Muhammed Arif & Bakırcıoğlu, Veli & Kalyoncu, Mete. (2017). 
Inverse Kinematic Analysis Of A Quadruped Robot. 
International Journal of Scientific & Technology Research. 6. 
"""

from . import transformations
from math import pi, cos, sin, atan2, sqrt
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

def ikine(x4,y4,z4,l1,l2,l3,legs12=True):
    '''Use inverse kinematics fo calculate the leg angles for a leg to achieve a desired
    leg end point position (x4,y4,z4)

    Args:
        x4: x position of leg end point relative to leg start point coordinate system.
        y4: y position of leg end point relative to leg start point coordinate system.
        z4: z position of leg end point relative to leg start point coordinate system.
        l1: leg link 1 length
        l2: leg link 2 length
        l3: leg link 3 length
        legs12: Optional input, boolean indicating whether equations are for legs 1 or 2. 
                If false, then equation for legs 3 and 4 is used

    Returns:
        A length 3 tuple of leg angles in the order (q1,q2,q3)
    '''

    # Supporting variable D
    D = (x4**2 + y4**2 + z4**2 - l1**2 - l2**2 - l3**2)/(2*l2*l3)

    if legs12 == True:
        q3 = atan2(sqrt(1-D**2),D)
    else:
        q3 = atan2(-sqrt(1-D**2),D)
    
    q2 = atan2(z4, sqrt(x4**2 + y4**2 - l1**2)) - atan2(l3*sin(q3), l2 + l3*cos(q3) )  

    # After using the equations, there seem to be two errors:
    #   1. The first y4 should not have a negative sign
    #   2. The entire equation should be multiplied by -1
    # The equation for q1 below reflects these changes 
    q1 = atan2(y4, x4) + atan2(sqrt(x4**2 + y4**2 - l1**2), -l1)

    return (q1,q2,q3)

def get_leg_delta_coords(ht_body,p4_coords,l,w):
    '''Get the delta coordinates for each leg's coordinate system from a body pose, and a set
    of foot points in absolute coordinate frame. The delate coordinates are used to go into the 
    inverse kinematics equations.

    Args:
        ht_body: Desired pose of the body. A 4x4 homogeneous transformation numpy matrix
        p4_coords: A 4x3 numpy matrix of desired (x4,y4,z4) positions for the end point (point 4) of each of
                    the four legs. I.e., the foot.
                    Leg order: rigthback, rightfront, leftfront, leftback. Example input:
                        np.array( [ [x4_rb,y4_rb,z4_rb],
                                    [x4_rf,y4_rf,z4_rf],
                                    [x4_lf,y4_lf,z4_lf],
                                    [x4_lb,y4_lb,z4_lb] ])
        l: Length of robot base
        w: Width of robot base
        
    Returns:
        A 4x3 numpy matrix of the foot coordinates for each leg, in the legs' coordinate system
        Order is rightback, rightfront, leftfront, leftback 
    '''

    # First get the pose (the homogeneous transformation matices) for each leg's start point coordinate system
    ht_rb_leg = t_rightback(ht_body,l,w)
    ht_rf_leg = t_rightfront(ht_body,l,w)
    ht_lf_leg = t_leftfront(ht_body,l,w)
    ht_lb_leg = t_leftback(ht_body,l,w)

    # Get inverse transformation matrices for each leg
    ht_rb_leg_inv = transformations.ht_inverse(ht_rb_leg)
    ht_rf_leg_inv = transformations.ht_inverse(ht_rf_leg)
    ht_lf_leg_inv = transformations.ht_inverse(ht_lf_leg)
    ht_lb_leg_inv = transformations.ht_inverse(ht_lb_leg)

    # Transform global (x4,y4,z4) coordinates for each leg, into each leg's starting point coordinate system
    # Use the inverse of each leg's start point homogeneous transform. E.g.:
    # foot_point_in_legs_coord_system   =   transpose(ht_rb_leg) * foot_point_in_global_coord_sys 
    
    # Create length 4 vector of p4 points to use with homogeneous transformatin matrices
    rb_leg_p4_global_coords = np.block( [p4_coords[0,:], 1] )
    rf_leg_p4_global_coords = np.block( [p4_coords[1,:], 1] )
    lf_leg_p4_global_coords = np.block( [p4_coords[2,:], 1] )
    lb_leg_p4_global_coords = np.block( [p4_coords[3,:], 1] )

    # Calculate foot coordinates in each leg's coordinate system
    rb_p4_in_leg_coord_sys = ht_rb_leg_inv.dot(rb_leg_p4_global_coords) 
    rf_p4_in_leg_coord_sys = ht_rf_leg_inv.dot(rf_leg_p4_global_coords) 
    lf_p4_in_leg_coord_sys = ht_lf_leg_inv.dot(lf_leg_p4_global_coords) 
    lb_p4_in_leg_coord_sys = ht_lb_leg_inv.dot(lb_leg_p4_global_coords) 

    # Prepare return value
    ret_val = np.block([    [rb_p4_in_leg_coord_sys[:-1]],
                            [rf_p4_in_leg_coord_sys[:-1]],
                            [lf_p4_in_leg_coord_sys[:-1]],
                            [lb_p4_in_leg_coord_sys[:-1]]  ])
    return ret_val

def ikine_body_pose(ht_body,p4_coords,body_length,body_width,l1,l2,l3):
    '''Caculate the leg joint angles to achieve a desired body pose and set of absolute foot positions

    Args:
        ht_body: Desired pose of the body. A 4x4 homogeneous transformation numpy matrix
        p4_coords: A 4x3 numpy matrix of desired (x4,y4,z4) positions for the end point (point 4) of each of
                    the four legs. I.e., the foot.
                    Leg order: rigthback, rightfront, leftfront, leftback. Example input:
                        np.array( [ [x4_rb,y4_rb,z4_rb],
                                    [x4_rf,y4_rf,z4_rf],
                                    [x4_lf,y4_lf,z4_lf],
                                    [x4_lb,y4_lb,z4_lb] ])
        body_length: Length of robot base
        body_width: Width of robot base
        l1: leg hip length
        l2: upper leg length
        l3: lower leg length

    Returns
        A 4x3 tuple of joint angles for each pf the four legs
        Leg order: rightback, rightfront, leftfront, leftback 
        Joint angle order: q1, q2, q3 (or hip, leg, knee)
    '''

    # Get the delta coordinates for each leg's coordinate system
    leg_delta_coords = get_leg_delta_coords(ht_body,p4_coords,body_length,body_width)

    # Run the inverse kinematics for each leg's set of coordinates
    rb_leg_angs = ikine(leg_delta_coords[0,0],leg_delta_coords[0,1],leg_delta_coords[0,2],
                        l1,l2,l3,legs12=True)

    rf_leg_angs = ikine(leg_delta_coords[1,0],leg_delta_coords[1,1],leg_delta_coords[1,2],
                        l1,l2,l3,legs12=True)

    lf_leg_angs = ikine(leg_delta_coords[2,0],leg_delta_coords[2,1],leg_delta_coords[2,2],
                        l1,l2,l3,legs12=False)
    
    lb_leg_angs = ikine(leg_delta_coords[3,0],leg_delta_coords[3,1],leg_delta_coords[3,2],
                        l1,l2,l3,legs12=False)

    return (rb_leg_angs,
            rf_leg_angs,
            lf_leg_angs,
            lb_leg_angs)