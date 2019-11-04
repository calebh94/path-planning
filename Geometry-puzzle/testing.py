import matplotlib.pyplot as plt




# class ShapePuzzle:
#

plt.axes()

# circle = plt.Circle((0, 0), radius=0.75, fc='y')
# plt.gca().add_patch(circle)

# plt.axis('scaled')
# plt.show()
# rectangle = plt.Rectangle((10, 10), 100, 100, fc='r')
# plt.gca().add_patch(rectangle)

points = [[2, 1], [8, 1], [8, 4]]
polygon = plt.Polygon(points)
plt.gca().add_patch(polygon)
plt.axis('scaled')
plt.show()


# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation
#
# fig = plt.figure()
# fig.set_dpi(100)
# fig.set_size_inches(7, 6.5)
#
# ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
# patch = plt.Circle((5, -5), 0.75, fc='y')
#
# def init():
#     patch.center = (5, 5)
#     ax.add_patch(patch)
#     return patch,
#
# def animate(i):
#     x, y = patch.center
#     x = 5 + 3 * np.sin(np.radians(i))
#     y = 5 + 3 * np.cos(np.radians(i))
#     patch.center = (x, y)
#     return patch,
#
# anim = animation.FuncAnimation(fig, animate,
#                                init_func=init,
#                                frames=360,
#                                interval=20,
#                                blit=True)
#
# plt.show()