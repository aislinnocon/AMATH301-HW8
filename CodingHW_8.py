
import numpy as np
import scipy.integrate
import scipy.optimize

######### Problem 1 ###############
x0 = 1
x_true = lambda t: (1/2) * (np.cos(t) + np.sin(t) + np.exp(-t))

dt = 0.1
T = 10
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n-1):
    x[k + 1] = x[k] + dt * (np.cos(t[k]) - x[k])
A1 = x.reshape(1, 101)
# print("A1 = ", A1)

abs = np.abs(x - x_true(t))
A2 = abs.reshape(1, 101)
# print("A2 = ", A2)

dt = 0.1
T = 10
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n - 1):
    x[k + 1] = (x[k] + dt * np.cos(t[k + 1])) / (1 + dt)
A3 = x.reshape(1, 101)
# print("A3 = ",A3)

absD = np.abs(x - x_true(t))
A4 = absD.reshape(1, 101)
# print("A4 = ", A4)

f = lambda t, x: np.cos(t) - x
tspan = (0, T)
guess = np.array([x0])
sol = scipy.integrate.solve_ivp(f, tspan, guess, t_eval = t)
x = sol.y[0, :]
A5 = x.reshape(1, 101)
# print("A5 = ", A5)

abs = np.abs(x - x_true(t))
A6 = abs.reshape(1, 101)
# print("A6 = ", A6)

###### Problem 2 ############
a = 8
x0 = np.pi/4
x_true = lambda t: 2 * np.arctan(np.exp(a * t)/ (1 + np.sqrt(2)))

dt = 0.01
T = 2
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n-1):
    x[k + 1] = x[k] + dt * a * np.sin(x[k])
xK = x
A7 = x.reshape(1, 201)
# print("A7 = ", A7)

A8 = np.max(np.abs(x - x_true(t)))
# print("A8 = ", A8)

dt = 0.001
T = 2
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
for k in range(n-1):
    x[k + 1] = x[k] + dt * a * np.sin(x[k])
partCError = np.max(np.abs(x - x_true(t)))
A9 = A8 / partCError
# print("A9 = ", A9)

## Backward Euler
dt = 0.01
T = 2
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
z0 = 3
for k in range(n - 1):
    F = lambda z: z - x[k] - a * dt * np.sin(z)
    x[k + 1] = scipy.optimize.fsolve(F, 3)
A10 = x.reshape(1, 201)
# print("A10 = ", A10)

A11 = np.max(np.abs(x - x_true(t)))
print("A11 = ", A11)

dt = 0.001
T = 2
t = np.arange(0, T + dt, dt)
n = t.size
x = np.zeros(n)
x[0] = x0
z0 = 3
for k in range(n - 1):
    F = lambda z: z - x[k] - a * dt * np.sin(z)
    x[k + 1] = scipy.optimize.fsolve(F, 3)
A12 = A11 / (np.max(np.abs(x - x_true(t))))
print("A12 = ", A12)

a = 8
dt = 0.01
T = 2
t = np.arange(0, T + dt, dt)
f = lambda t, x: a * np.sin(x)
tspan = (0, T)
guess = np.array([x0])
sol = scipy.integrate.solve_ivp(f, tspan, guess, t_eval = t)
x = sol.y[0, :]
A13 = x.reshape(1, 201)
# print("A13 = ", A13)

A14 = np.max(np.abs(x - x_true(t)))
print("A14 = ", A14)

dt = 0.001
T = 2
t = np.arange(0, T + dt, dt)
f = lambda t, x: a * np.sin(x)
tspan = (0, T)
guess = np.array([x0])
sol = scipy.integrate.solve_ivp(f, tspan, guess, t_eval = t)
x = sol.y[0, :]
A15 = A14 / (np.max(np.abs(x - x_true(t))))
print("A15 = ", A15)
