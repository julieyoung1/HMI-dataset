##LEAST SQUARE METHOD
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

plt.rcParams['font.sans-serif'] = ['SimHei']

Xi = np.array([1.2, 1.3, 1.9, 1.1, 1.75, 1.78, 1.55, 1.45, 1.6, 2.2])
# Yi = np.array([0.802, 0.951, 1.529])
Yi = np.array([1.476, 1.524, 1.78, 1.441, 1.657, 1.674, 1.667, 1.564, 1.679, 1.939])

def func(p, x):
    k, b = p
    return k * x + b

def error(p, x, y):
    return func(p, x) - y

p0 = [1, 1]
Para = leastsq(error, p0, args=(Xi, Yi))
k, b = Para[0]
print("k=", round(k, 3), "b=", round(b, 3))
print("cost：" + str(Para[1]))
print("fit liner:")
print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))

plt.figure(figsize=(8, 6))
plt.scatter(Xi, Yi, color="green", label="data", linewidth=2)

x = np.array([1, 2, 3])
y = k * x + b
plt.plot(x, y, color="red", label="fit liner", linewidth=2)
plt.title('y={}+{}x'.format(b, k))
plt.legend(loc='lower right')
plt.show()