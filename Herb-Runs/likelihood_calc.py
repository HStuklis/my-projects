# coding: utf-8
import numpy as np
from scipy import stats
from scipy import optimize
data = np.random.negative_binomial(1,0.3,size=30)
print(data)
def negLikelihood(theta):
    output = 1
    for item in data:
      output = output*stats.nbinom.pmf(item,1,theta,loc=0)
    return -output
    
optimize.minimize_scalar(negLikelihood,bounds=(0,1),method = 'Bounded')
