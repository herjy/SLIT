import Test_center as tc
import sys
import numpy as np




variable = sys.argv[1]
shift = sys.argv[2]

if variable == 'center':
    tc.test_center(np.float(shift))
if variable == 'slope':
    tc.test_slope(np.float(shift))

