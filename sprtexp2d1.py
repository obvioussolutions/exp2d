#SPRT for exp2d distribution

#libraries used
import math
import numpy as np
from scipy import special
from exp2d import *

#parameters
alpha = .05
beta = .15

#lower and upper intercepts
lower_intercept = np.log(beta/(1-alpha))
upper_intercept = np.log((1-beta)/alpha)

#hypotheses
#null hypothesis
a_null = 1.2
b_null = 1.3
c_null = .01

#alternative hypothesis
a_hyp = 1.2
b_hyp = 1.3
c_hyp = .1

#get the slope of the barriers
q_null = (a_null*b_null)/c_null
q_hyp = (a_hyp*b_hyp)/c_hyp
slope = np.log(c_null/c_hyp)-(q_null - q_hyp)-np.log((special.exp1(q_null))/special.exp1(q_hyp))

#pre-loop
lrt = 0
n = 0
result = 'continue'

while result == 'continue':
    x,y = rexp2d(1,a_null,b_null,c_null)
    n += 1
    lrt += (a_null-a_hyp)*x+(b_null-b_hyp)*y+(c_null-c_hyp)*x*y
    lower_bound, upper_bound = (lower_intercept,upper_intercept) + slope*n
    if lrt < lower_bound:
        result = 'accept null'
    if lrt > upper_bound:
        result = 'accept alternative'

    
