import math
import numpy as np

def gexpdrvs(n,a,c,slack):
    #check parameters
    if (n <= 0) or (type(n) != int): raise Exception('n must be a positive integer')
    if (np.min(a) <= 0) or (np.min(c) < 0):
        raise Exception('a is positive, c is nonnegative')
    d = np.size(a); g = np.size(c)
    if g != (d*(d-1)/2): raise Exception('c must map to a complete edge graph of d')
    
    #use rejection sampling
    x = np.zeros((n,d))
    for i in range(n):
        flag = 0;v = np.zeros(d); u = np.random.uniform(); pratio = 1; expsum = 0; ct = 0
        while flag == 0:
            for j in range(d):
                v[j] = np.random.exponential(scale = a[j])
            ct = 0; expsum = 0
            for j in range(d-1):
                for k in range(j+1,d):
                    expsum += v[j]*v[k]*c[ct]
                    ct += 1
            pratio = np.exp(-expsum)
            if u <= pratio/slack: flag = 1
        x[i] = v
    
    #return statement
    return x
