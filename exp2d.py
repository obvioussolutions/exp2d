#exp2d function in python
import math
import numpy as np
from scipy import special, integrate
#exp2d
class exp2d:
    #define the class for the distribution
    def __init__(self,a = (1,1),c = 0):
        #basic variables
        self.a = a[0]; self.b = a[1]; self.c = c
        #derived variables
        self.nc = self.a*self.b
        if self.c > 0:
            self.q = self.a*self.b/self.c; self.enq = np.exp(-self.q); self.e1q = special.exp1(self.q); self.nc = self.c*self.enq/self.e1q
        
    #check that the parameters are within the parameter space
    def in_Param_Space(self):
        if (self.a <= 0) or (self.b <= 0) or (self.c < 0):
             raise Exception('a,b must be positive and c must at least 0')
    
    #computes weighted dot product of the active function and some 2-tuple     
    def wsum(self,x,w):
        return (self.a*x[0]*w[0] + self.b*x[1]*w[1] + self.c*x[0]*x[1]*w[2])
    
    #joint pdf, takes in 2d vector x, returns pdf, check if in bounds 
    def pdf(self,x):
        if min(x) >= 0:
            return self.nc*np.exp(self.wsum(x,[-1,-1,-1]))
        else:
            return 0
    
    #joint cdf, takes in 2d vector x, returns cdf, check if in bounds also
    def cdf(self,x):
        if min(x) >= 0:
            if self.c > 0:
                e1 = special.exp1(self.q + self.wsum(x,[1,1,1]))
                e2 = special.exp1(self.q + self.wsum(x,[1,0,0]))
                e3 = special.exp1(self.q + self.wsum(x,[0,1,0]))
                return 1 + (e1 - e2 - e3)/self.e1q
            else:
                e1 = np.exp(-self.a*x[0]); e2 = np.exp(-self.b*x[1])
                return ((1-e1)*(1-e2))
        else:
            return 0
        
    #random variate generator function, takes in n, returns n rvs , via rejection sampling
    def check_posint(self,n):
        if (n <= 0) or (type(n) != int): raise Exception('n must be a positive integer')
    def rvs(self,n = 1,slack = 2):
        self.check_posint(n)
        x = np.zeros((n,2))
        x[:,0] = np.random.exponential(scale = self.a,size = n)
        x[:,1] = np.random.exponential(scale = self.b,size = n)
        
        #use rejection sampling
        if self.c > 0:
            u = 0; v = 0; w = 0; flag = 0; pratio = 0
            for i in range(n):
                flag = 0
                while flag == 0:
                    v = np.random.exponential(scale = self.a); w = np.random.exponential(scale = self.b)
                    u = np.random.uniform(); pratio = np.exp(-self.c*v*w)
                    if u <= pratio/slack: flag = 1
                x[i,0] = v
                x[i,1] = w
                
        return x

    #moments for the random variable, xyz is a char 'x','y','z'
    def moment(self,xyz,n):
        #check if it is a positive integer, xyz is either x y or z
        xy = ['x','y','z']
        self.check_posint(n)
        if xyz not in xy:
            raise Exception('xyz argument must be either x y or z char')
        
        #case for moment of x
        if xyz == 'x':
            if self.c > 0:
                return (special.gamma(n+1)*special.expn(n+1,self.q)/((self.a**n)*self.e1q))
            else:
                return special.gamma(n+1)/(self.a**n)
        
        #case for moment of y
        if xyz == 'y':
            if self.c > 0:
                return (special.gamma(n+1)*special.expn(n+1,self.q)/((self.b**n)*self.e1q))
            else:
                return special.gamma(n+1)/(self.b**n)
            
        #case for moment of z = x*y
        if xyz == 'z':
            if self.c == 0:
                return(self.moment('x',n)*self.moment('y',n))
            else:
                ans = 0
                for i in range(n+1):
                    ans += ((-1)**i)*special.binom(n,i)*special.expn(i+1,self.q)
                ans *= special.gamma(n+1)/((self.c**n)*self.e1q)
                return ans
        
    #other properties
    def covar(self):
        ans = np.zeros((2,2))
        #variances
        ans[0,0] = self.moment('x',2)-(self.moment('x',1))**2
        ans[1,1] = self.moment('y',2)-(self.moment('y',1))**2
        #covariance
        ans[0,1] = self.moment('z',1)-self.moment('x',1)*self.moment('y',1)
        ans[1,0] = ans[0,1]
        return ans
    #correlation
    def corr(self):
        return (self.covar()[0,1]/(self.covar()[0,0]*self.covar()[1,1])**(1/2))

    #marginal pdfs
    def xmarginalpdf(self,x):
        if x >= 0:
            return (self.nc*np.exp(-self.a*x)/(self.b+self.c*x))
        else:
            return 0
    def ymarginalpdf(self,y):
        if y >= 0:
            return(self.nc*np.exp(-self.b*y)/(self.a+self.c*y))
        else:
            return 0
    def zmarginalpdf(self,z):
        if z > 0:
            return(2*self.nc*np.exp(-self.c*z)*special.k0(2*np.sqrt(self.a*self.b*z)))
        else:
            return 0

    #marginal CDFs
    def xmarginalcdf(self,x):
        if x >= 0:
            return (1 - special.exp1(self.q + self.a*x)/self.e1q)
        else:
            return 0
    def ymarginalcdf(self,y):
        if y >= 0:
            return (1 - special.exp1(self.q + self.b*y)/self.e1q)
        else:
            return 0
    def zmarginalcdf(self,z):
        if z >= 0:
            return integrate.quad(self.zmarginalpdf,0,z)[0]
        else:
            return 0

    #marginal PPFs
    def quantxep(self,q):
        if (q <= 0) or (q >= 1):
            raise Exception('q must be between 0 and 1')
    def initial_upper_bound(self,xyz,q):
        xy = ['x','y','z']
        if xyz not in xy:
            raise Exception('xyz argument must be either x y or z char')
        if xyz == 'x':
            return (special.lambertw(1/((1-q)*self.e1q))/self.a - (self.b/self.c))
        if xyz == 'y':
            return (special.lambertw(1/((1-q)*self.e1q))/self.b - (self.a/self.c))
        if xyz == 'z':
            return (self.moment('z',1)/(1-q))
    def xmarginalppf(self,q,eps = 1e-6):
        self.quantxep(q)
        if self.c == 0:
            return (-np.log(1-q)/self.a)
        x = self.initial_upper_bound('x',q)
        while abs(self.xmarginalcdf(x)-q) > eps:
            x -= (self.xmarginalcdf(x)-q)/(self.xmarginalpdf(x) + (self.xmarginalcdf(x)-q)*((self.a*self.b+self.c+self.a*self.c*x)/(self.b+self.c*x)))
        return x.real
    def ymarginalppf(self,q,eps = 1e-6):
        self.quantxep(q)
        if self.c == 0:
            return (-np.log(1-q)/self.b)
        y = self.initial_upper_bound('y',q)
        while abs(self.ymarginalcdf(y)-q) > eps:
            y -= (self.ymarginalcdf(y)-q)/(self.ymarginalpdf(y) + (self.ymarginalcdf(y)-q)*((self.a*self.b+self.c+self.b*self.c*y)/(self.a+self.c*y)))
        return y.real
    def zmarginalppf(self,q,eps = 1e-6):
        self.quantxep(q)
        z = self.initial_upper_bound('z',q)
        t = 0
        while abs(self.zmarginalcdf(z)-q) > eps:
            t = (self.c + np.sqrt(self.a*self.b/z)*special.kn(1,2*np.sqrt(self.a*self.b*z))/special.kn(0,2*np.sqrt(self.a*self.b*z)))
            z -= (self.zmarginalcdf(z)-q)/(self.zmarginalpdf(z)+(self.zmarginalcdf(z)-q)*t)
        return z.real
    
    #rvs
    def xmarginalrvs(self,n = 1,slack = 2):
        self.check_posint(n)     
        x = np.random.exponential(scale = self.a,size = n)
        
        if self.c > 0:
            u = 0; v = 0; pratio = 0; flag = 0
            for i in range(n):
                flag = 0
                while flag == 0:
                    v = np.random.exponential(scale = self.a)
                    u = np.random.uniform(); pratio = 1/(1+self.c*v/self.b)
                    if u <= pratio/slack: flag = 1
            x[i] = v
        return x
    def ymarginalrvs(self,n = 1,slack = 2):
        self.check_posint(n)
        y = np.random.exponential(scale = self.b,size = n)

        if self.c > 0:
            u = 0; v = 0; pratio = 0; flag = 0
            for i in range(n):
                flag = 0
                while flag == 0:
                    v = np.random.exponential(scale = self.b)
                    u = np.random.uniform(); pratio = 1/(1+self.c*v/self.a)
                    if u <= pratio/slack: flag = 1
            y[i] = v
        return y
    def zmarginalrvs(self,n = 1,slack=2):
        self.check_posint(n)
        z = np.random.exponential(scale = self.a,size = n)*np.random.exponential(scale = self.b,size = n)
        
        if self.c > 0:
            u = 0; v= 0; w = 0; pratio = 0; flag = 0
            for i in range(n):
                flag = 0
                while flag == 0:
                    v = np.random.exponential(scale = self.a)
                    w = np.random.exponential(scale = self.b)
                    u = np.random.uniform(); pratio = np.exp(-self.c*v*w)
                    if u <= pratio/slack: flag = 1
            z[i] = v*w
        return z
        
