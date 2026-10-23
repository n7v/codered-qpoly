import bkz

import numpy as np

# Special class to manipulate auxilary structure S(B) described in [GS25, $3.1].
# 
# [GS25] Ghentiyala S., Stephens-Davidowitz N. - More basis reduction for linear codes - backward reduction, BKZ, slide reduction, and more (2024).pdf 

class S_aux:

    def __init__(self, B):
        self.S = zero_matrix(B.base_ring(), B.nrows()+1, B.ncols())
        #self.S[0] = zero_vector(B.base_ring(), B.ncols())
        self.i_stored = 0
    
    def update(self, B, i):
        for i0 in range(self.i_stored, i):
            if B.base_ring() == GF(2):
                self.S[i0+1] = bkz.or_vector_np(self.S[i0], B[i0])
            else:
                self.S[i0+1] = bkz.or_vector(self.S[i0], B[i0])
        self.i_stored = i
    
    def set_pos(self, i):
        if i < 0:
            self.i_stored = 0
        elif i <= self.i_stored:
            self.i_stored = i
        else:
            raise Exception(f"Index {i} is too big, call update() first")
    
    def __getitem__(self, key):
        return self.S[key]
