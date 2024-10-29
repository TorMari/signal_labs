import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

def function(x, n=6):
   return x**6 * np.exp(-x**2/6)

x = np.linspace(-np.pi, np.pi, 100)
plt.plot(x, function(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = x^24 * exp(-x^2/24)')
plt.show()



def fourier(li, lf, x, n):
   l = (lf-li)/2
   m=n
   a0=(2.0/l)*(quad(lambda x: function(x), 0, l)[0])
   an = []
   an.append(a0/2)

   for i in range(1, n+1):
      an_i = (2.0/l) * (quad(lambda x: x**6 * np.exp(-x**2/6) *np.cos(i*np.pi*x/l), 0, l)[0])
      an.append(an_i)

   fx = a0
   s = sum(an[i]*math.cos((i*x*np.pi)/l) for i in range(1, n+1))
   fx += s

   return an, fx

li = -np.pi
lf = np.pi

n = 10
x = np.pi/3
an, fx = fourier(li, lf, x, n)
print(an)
print('\n')
print(fx)

"""N=10
k_n=np.arange(0, N+1)

plt.subplot(2,1,1)
plt.stem(k_n, an)
plt.show()


x = np.arange(-np.pi, np.pi, 0.3)
y_approx = []
y_exact = []
y_approx_all = []
for i in x:
   an, fx = fourier(-np.pi, np.pi, i, 10)
   y_approx_all.append(fx)

for j in range(1, 10):
   for i in x:
      an_all, fx_all = fourier(-np.pi, np.pi, i, j)
      y_approx_all.append(fx_all)
   plt.plot(x, y_approx_all)
   y_approx_all = []

for i in x:
   y_exact.append(function(i))

error = []

for i in range(0, len(y_exact)):
   error.append((y_approx[i]-y_exact[i])/y_exact[i])

relative_error = np.abs(error)

plt.plot(x, y_exact, label = "Exact")
plt.plot(x, y_approx, label="Approx")
print(y_approx_all)

plt.legend()
plt.show()

plt.plot(x, relative_error)
plt.show()"""





def a_coef(x, N):
   coef = []
   for i in range(N):
      integral_function = lambda x=np.pi/3: function(x)*np.cos(i*x)
      integral, error = quad(integral_function, -np.pi, np.pi)
      #coef[i] = 1/math.pi*integral
      coef.append(integral/np.pi)
   print(coef)
   return coef


def b_coef(x, N):
   coef = []
   for i in range(N):
      integral_function = lambda x: function(x)*np.sin(i*x)
      integral, error = quad(integral_function, -np.pi, np.pi)
      #coef[i] = 1/math.pi*integral
      coef.append(integral/np.pi)
   print(coef)
   return coef



def trigonometric_series(a_coef, x, N):
   res = a_coef[0]
   for i in range(1, N-1):
      res += a_coef[i]*np.cos(i*x)
   return res


x = float(input("Введіть значення x: "))
print(function(x))

N = 10
a = a_coef(x, N)
print(trigonometric_series(a, x, N))