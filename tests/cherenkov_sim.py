from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.integrate import quad

s = float(input("Value to use for shower age: "))
num_ele = int(input("Number of values to generate: "))
#initial_guess = float(input("Value to use for initial guess for numerical solver: "))
# E = float(input("Value to use for particle kinetic energy: "))

# Set value for E_0 according to s
if s >= 0.4:
    E_0 = 44 - 17*(s - 1.46)**2

elif s < 0.4:
    E_0 = 26

elif s is not float:
    print("Shower age needs to be a number.")


# Create a linspace of E values from 10 to 10^4 eVs
E = np.linspace(10, 10**4, num_ele)


# Determine T for each value of E
T = (((0.89*E_0) - 1.2) / ((E_0 + E)) * (1 + (10**(-4)) * s*E)**(-2))

# Plot E vs T
# plt.plot(E, T)
# plt.xlabel("T(E)", fontsize = 16)
# plt.ylabel("E", fontsize = 16)
# plt.rcParams.update({'font.size': 22})
# plt.show()

PDF = - (10**8)*s*((s + 2)*E + 2*E_0 + 10**4)*((((89*E_0/100) - 6/5) / (E + E_0))**s) / ((E + E_0)*((s*E + 10**4)**3))

plt.plot(E, PDF)
plt.xlabel("PDF(E)", fontsize = 16)
plt.ylabel("E", fontsize = 16)
plt.rcParams.update({'font.size': 22})
plt.show()

def f(E):
    return - (10**8)*s*((s + 2)*E + 2*E_0 + 10**4)*((((89*E_0/100) - 6/5) / (E + E_0))**s) / ((E + E_0)*((s*E + 10**4)**3))

# NUMERICAL SOLVER METHOD
# ===================================================================================
#print(root_scalar(PDF, args = (E, E_0, s), x0 = initial_guess, method = 'newton'))

#print(root_scalar(PDF, args = (E), x0 = initial_guess, method = 'newton'))



# SCIPY'S RV CONTINUOUS METHOD
# ================================================================================================
# create own random generator call
# typically don’t have the initialize anything

# r = np.linspace(0, 1, 10)

# class randgen(rv_continuous):
# 	def __init__(self, E_0 = 26, s = 0.8):
# 		#super.__init__(self, a=a, b=b)
# 		self.E_0 = E_0
# 		self.s = s	

#     def __pdf(self, E):
# 	    return -(10**8)*self.s*((self.s + 2)*E + 2*E_0 + 10**4)*((((89*self.E_0/100) - 6/5) / (E + self.E_0))**self.s) / ((E + self.E_0)*((self.s*E + 10**4)**3))


# rg = randgen(E_0 = E_0, s = s)
# plt.hist(rg.rvs(size = num_ele))
# plt.show()


# INTEGRATE NUMERICALLY THEN DETERMINE NORMALISATION FACTOR
# ============================================================================================================================================
# quad function is for when the explicit function form of the integrand function.
# limits can be infinite, but must be numbers
# used to find definite integral not a infinite integral.
# output of quad function is a tuple
# get actual result with 
# result = quad(<parameters>)
# result[0]
# result[1] gives estimate of the error
# integration_result = quad(f, -np.inf, np.inf)

#integration_result = quad(f, -15, np.inf)

err_vals = np.array([])

for i in range(-10000, 10000):
    try:
      integration_result = quad(f, i, np.inf)
    
    except Exception as e:
        #raise RuntimeError(f"Error integrating for lower limit of {i}: {e}")
        np.append(err_vals, i)
        print(f"Error integrating for lower limit of {i}")
        continue



#print(integration_result)