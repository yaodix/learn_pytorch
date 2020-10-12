import numpy as np

ratios = [0.5,1,2]
scales = [8,16,32]
input_size = [800,800]
sub_sample = 16

anchor_base = np.zeros([len(ratios)*len(scales),4])

ctr_x = sub_sample/2.
ctr_y = sub_sample/2.


