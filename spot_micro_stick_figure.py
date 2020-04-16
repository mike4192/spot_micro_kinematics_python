import math
from math import pi, sin, cos
import matplotlib.pyplot as plt
import numpy as np
from .utilities import spot_micro_kinematics as smk
from .utilities import transformations

d2r = pi/180
r2d = 180/pi

class SpotMicroLeg(object):
    '''Encapsulates a spot micro leg that consists of 3 links and 3 joint angles
    
    Attributes:
        _q1: Rotation angle in radians of hip joint
        _q2: Rotation angle in radians of upper leg joint
        _q3: Rotation angle in radians of lower leg joint
        _l1: Length of leg link 1 (i.e.: hip joint)
        _l2: Length of leg link 2 (i.e.: upper leg)
        _l3: Length of leg link 3 (i.e.: lower leg)
        _ht_leg: Homogeneous transformation matrix of leg starting 
                 position and coordinate system relative to robot body.
                 4x4 np matrix    
    '''

    def __init__(self,q1,q2,q3,l1,l2,l3,ht_leg_start):
        '''Constructor'''
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._ht_leg_start = ht_leg_start

        # Create homogeneous transformation matrices for each joint
        self._t01 = smk.t_0_to_1(self._q1,self._l1)
        self._t12 = smk.t_1_to_2()
        self._t23 = smk.t_2_to_3(self._q2,self._l2)
        self._t34 = smk.t_3_to_4(self._q3,self._l3)


    def set_angles(self,q1,q2,q3):
        '''Set the three leg angles and update transformation matrices as needed'''
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3
        self._t01 = smk.t_0_to_1(self._q1,self._l1)
        self._t23 = smk.t_2_to_3(self._q2,self._l2)
        self._t34 = smk.t_3_to_4(self._q3,self._l3)
    
    def set_homog_transf(self,ht_leg_start):
        '''Set the homogeneous transformation of the leg start position'''
        self._ht_leg_start = ht_leg_start

    def get_leg_points(self):
        '''Get coordinates of 4 points that define a wireframe of the leg:
            Point 1: hip/body point
            Point 2: upper leg/hip joint
            Point 3: Knee, (upper/lower leg joint)
            Point 4: Foot, leg end
        
        Returns:
            A length 4 tuple consisting of 4 length 3 numpy arrays representing the 
            x,y,z coordinates in the global frame of the 4 leg points
        '''
        # Build up the total homogeneous transformation incrementally, saving each leg
        # point along the way
        # The total homogeneous transformation builup is:
        # ht = ht_leg_start @ t01 @ t12 @ t23 @ t34 
        p1 = self._ht_leg_start[0:3,3]

        ht_buildup = self._ht_leg_start @ self._t01 @ self._t12

        p2 = ht_buildup[0:3,3]

        ht_buildup = ht_buildup @ self._t23

        p3 = ht_buildup[0:3,3]

        ht_buildup = ht_buildup @ self._t34

        p4 = ht_buildup[0:3,3]

        return (p1,p2,p3,p4)

class SpotMicroStickFigure(object):
    """Encapsulates an 12 DOF spot micro stick figure  

    Encapuslates a 12 DOF spot micro stick figure. The 12 degrees of freedom represent the 
    twelve joint angles. Contains inverse kinematic capabilities
    
    Attributes:
        hip_length: Length of the hip joint
        upper_leg_length: length of the upper leg link
        lower_leg_length: length of the lower leg length
        body_width: width of the robot body
        body_height: length of the robot body

        x: x position of body center
        y: y position of body center
        z: z position of body center

        phi: roll angle in radians of body
        theta: pitch angle in radians of body
        psi: yaw angle in radians of body

        ht_body: homogeneous transformation matrix of the body

        rightback_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        rightfront_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        leftfront_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        leftback_leg_angles: length 3 list of joint angles. Order: hip, leg, knee

        leg_rightback
        leg_rightfront
        leg_leftfront
        leg_leftback
        
    """
    def __init__(self,x=0,y=.18,z=0,phi=0,theta=0,psi=0):
        '''constructor'''
        self.hip_length = 0.055
        self.upper_leg_length = 0.1075
        self.lower_leg_length = 0.130
        self.body_width = 0.078
        self.body_length = 0.186

        self.x = x
        self.y = y
        self.z = z
        
        self.phi = phi
        self.theta = theta
        self.psi = psi   

        # self.ht_body = transformations.homog_transform(self.phi,self.psi,self.theta,
        #                                                self.x,self.y,self.z)

        #TODO: make initialization of body pose clear
        # linear transformation then rotation to achieve a position, and body orientation
        self.ht_body = transformations.homog_transxyz(self.x,self.y,self.z) @ transformations.homog_rotxyz(self.phi,self.psi,self.theta)

        # Intialize all leg angles to 0, 30, 30 degrees
        self.rb_leg_angles   = [0,-30*d2r,60*d2r]
        self.rf_leg_angles   = [0,-30*d2r,60*d2r]
        self.lf_leg_angles   = [0,30*d2r,-60*d2r]
        self.lb_leg_angles   = [0,30*d2r,-60*d2r]

        self.leg_rightback = SpotMicroLeg(self.rb_leg_angles[0],self.rb_leg_angles[1],self.rb_leg_angles[2],
                                          self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                          smk.t_rightback(self.ht_body,self.body_length,self.body_width))
        
        self.leg_rightfront = SpotMicroLeg(self.rf_leg_angles[0],self.rf_leg_angles[1],self.rf_leg_angles[2],
                                          self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                          smk.t_rightfront(self.ht_body,self.body_length,self.body_width))
                                                  
        self.leg_leftfront = SpotMicroLeg(self.lf_leg_angles[0],self.lf_leg_angles[1],self.lf_leg_angles[2],
                                          self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                          smk.t_leftfront(self.ht_body,self.body_length,self.body_width))

        self.leg_leftback = SpotMicroLeg(self.lb_leg_angles[0],self.lb_leg_angles[1],self.lb_leg_angles[2],
                                          self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                          smk.t_leftback(self.ht_body,self.body_length,self.body_width)) 

    def get_leg_coordinates(self):
        '''Return coordinates of each leg as a tuple of 4 sets of 4 leg points'''

        leg_rightback_coords    = self.leg_rightback.get_leg_points()
        leg_rightfront_coords   = self.leg_rightfront.get_leg_points()
        leg_leftfront_coords    = self.leg_leftfront.get_leg_points()
        leg_leftback_coords     = self.leg_leftback.get_leg_points()
        
        return (leg_rightback_coords,leg_rightfront_coords,leg_leftfront_coords,leg_leftback_coords)

    def set_leg_angles(self,leg_angs):
        ''' Set the leg angles for all four legs

        Args:
            leg_angs: Tuple of 4 lists of leg angles. Legs in the order rightback
                      rightfront, leftfront, leftback. ANgles in the order q1,q2,q3.
                      An example input:
                        ((rb_q1,rb_q2,rb_q3),
                         (rf_q1,rf_q2,rf_q3),
                         (lf_q1,lf_q2,lf_q3),
                         (lb_q1,lb_q2,lb_q3))

        Returns:
            Nothing
        '''
        self.leg_rightback.set_angles(leg_angs[0][0],leg_angs[0][1],leg_angs[0][2])
        self.leg_rightfront.set_angles(leg_angs[1][0],leg_angs[1][1],leg_angs[1][2])
        self.leg_leftfront.set_angles(leg_angs[2][0],leg_angs[2][1],leg_angs[2][2])
        self.leg_leftback.set_angles(leg_angs[3][0],leg_angs[3][1],leg_angs[3][2])            

    def print_leg_angles(self):
        ''' Print the joint angles for alll four legs'''
        return None