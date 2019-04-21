import numpy as np
from scipy import stats
from scipy import optimize

data = np.random.negative_binomial(1,0.3,size=1000)

def negLogLikelihood(theta):
    return -np.sum(np.log(stats.nbinom.pmf(data,1,theta,loc=0)))

res = optimize.minimize_scalar(negLogLikelihood,bounds=(0,1),method='Bounded')
print(res.x)