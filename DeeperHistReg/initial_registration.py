import sys
current_file = sys.modules[__name__]

import DeeperHistReg.utils as u

import DeeperHistReg.rotated_landmark_based_combination as rlbc

### Note ###
# It is assumed that all algorithms in the initial_registration.py return 2x3 transformation matrix in the PyTorch format
############

### Algorithms ###

def identity_initial_registration(source, target, params):
    return u.create_identity_displacement_field(source)

def rotated_landmark_based_combination(source, target, params):
    return rlbc.rotated_landmark_based_combination(source, target, params)

### Utility ###

def get_function(function_name):
    return getattr(current_file, function_name)