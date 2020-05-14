import numpy as np
def L2_Distance(feature1,feature2):
    return np.sqrt(np.sum(np.square(feature1-feature2)))
def L1_Distance(feature1,feature2):
    return np.sum(np.abs(feature1-feature2))