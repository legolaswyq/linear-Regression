import numpy as np
import matplotlib.pyplot as plt

f = lambda x,y: np.sin(x) * np.cos(y)
x = np.linspace(-2,2,50)
y = np.linspace(-2,2,50)

x,y = np.meshgrid(x,y)
F = f(x,y)
plt.contourf(x,y,F,20)
plt.show()