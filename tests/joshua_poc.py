from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.integrate import quad

s = float(input("Value to use for shower age: "))

num_ele = int(input("Number of values to generate: "))

# Set value for E_0 according to s
def E_0(s):
    if s >= 0.4:
        E_0 = 44 - 17*(s - 1.46)**2
        return E_0

    elif s < 0.4:
        E_0 = 26
        return E_0

    elif s is not float:
        print("Shower age needs to be a number.")


E = np.linspace(10, 10**4, num_ele)

# Create a function to hold the CDF
def T(E, E_0, s):
    return (((((0.89*E_0) - 1.2) / ((E_0 + E)))**s) * (1 + (10**(-4)) * s*E)**(-2))

# Create a function which is T but replacing E = 10**x
def F(x, E_0, s):
    return (((((0.89*E_0) - 1.2) / ((E_0 + 10**x)))**s) * (1 + (10**(-4)) * s*10**x)**(-2))

# Create a function to hold the PDF
def PDF(E, E_0, s):
    return - s * T(E, E_0, s) * ((1/(E_0 + E)) + 2/(10**4 + s*E))

# Create a function which is the derivative of F
def Fderivative(x, E_0, s):
    return - s * F(x, E_0, s) * ((1/(E_0 + 10**x)) + 2/(10**4 + s*(10**x)))




# create own random generator call
# r = np.linspace(0, 1, 10)



class randgen(rv_continuous):
    def __init__(self, E_0 = 26, s = 0.8):
        super().__init__(a=0, b=10**4) # a and b are the lower and upper limits for E respectively
        self.E_0 = E_0
        self.s = s
    
    def _pdf(self, E):
        return (((((0.89*self.E_0) - 1.2) / ((self.E_0 + E)))**self.s) * (1 + (10**(-4)) * self.s*E)**(-2))
    
    # def _pdf(self, E):
    #     return (((((0.89*self.E_0) - 1.2) / ((self.E_0 + E)))**self.s) * (1 + (10**(-4)) * self.s*E)**(-2))
    

normalisation = 1/(T(np.inf, E_0(s), s) - T(10, E_0(s), s))
# normalisation = 1/(T(10**4, E_0(s), s) - T(10, E_0(s), s))

normalisationFderivative = 1/(F(4, E_0(s), s) - F(1, E_0(s), s))

x = np.linspace(1, 4, num_ele)

rg = randgen(E_0 = E_0(s), s = s)
plt.hist(rg.rvs(size = num_ele), bins = 50, density = True)
# plt.plot(E, normalisation*PDF(E, E_0(s), s))
# plt.plot(E, normalisation*T(E, E_0(s), s))
# plt.plot(E, T(E, E_0(s), s))
# plt.plot(E, normalisationFderivative*Fderivative(x, E_0(s), s))
# plt.xscale('log')
# plt.yscale('log')
plt.show()