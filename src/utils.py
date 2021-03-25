# Taiyu Long@mLab
# 03/19/2021 
# some util functions

import editdistance

def manual_rule(name):
    if name == 'ring' : return 'amazon'
    if name == 'nest' : return 'google'
    if name == 'phillips' : return 'philips'
    if name == 'wyzelab' : return 'wyze'
    return name

def is_equivalent(inferred, expected, threshold=2):
    if inferred == expected:
        return 1
    
    if len(expected) > 4 and len(inferred) > 4:
        if editdistance.eval(expected, inferred) <= threshold:
            return 1
    return 0 
