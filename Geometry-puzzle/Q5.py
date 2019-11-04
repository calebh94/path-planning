import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Shape:
    def __init__(self, points, anchor=0):
        self.points = points
        self.anchor = anchor
        self.polygon = plt.Polygon(self.points)

    def plot(self):
        plt.gca().add_patch(self.polygon)
        # plt.axis('scaled')
        # plt.show()

a = 2 #  scale
plt.gca()
plt.xlim(xmin=0,xmax=a*8/(2*np.sqrt(2)))
plt.ylim(ymin=0,ymax=a*4/(2*np.sqrt(2)))
points = [[0, a*2/(2*np.sqrt(2))], [a*2/(2*np.sqrt(2)), 0], [a*6/(2*np.sqrt(2)), 0], [a*8/(2*np.sqrt(2)), a*2/(2*np.sqrt(2))], [a*6/(2*np.sqrt(2)),a*4/(2*np.sqrt(2))], [a*2/(2*np.sqrt(2)), a*4/(2*np.sqrt(2))],[0, a*2/(2*np.sqrt(2))]]
bounds = plt.Polygon(points, closed=None, fill=None, edgecolor='r')
plt.gca().add_patch(bounds)


# shape1 = Shape([[0,0], [0, a], [a, a], [a, 0]])
# shape1.plot()
shape2 = Shape([[0,0], [0, a/np.sqrt(2)], [a/(2*np.sqrt(2)), a*3/(2*np.sqrt(2))], [a/np.sqrt(2), a/np.sqrt(2)], [a/np.sqrt(2), 0]])
shape2.plot()
plt.show()

