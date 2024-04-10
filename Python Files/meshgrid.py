import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Generating a 2D grid of data
x = np.linspace(0, 4 * np.pi, 100)
y = np.linspace(0, 4 * np.pi, 100)
X, Y = np.meshgrid(x, y)
print('check data dimensions:\n', X.shape, Y.shape)
# values of np.sin(X) * np.cos(Y)
Z = np.sin(X) * np.cos(Y)
print('\ncheck Z: dimensions:')
print(Z.shape)
plt.figure(figsize=(8, 6))
# 'you can choose others like 'plasma', 'inferno', etc.
plt.pcolormesh(X, Y, Z, cmap='coolwarm')
plt.title("Pseudocolor Plot using pcolormesh")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()
plt.show()

# Define the range for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
# Create the meshgrid
X, Y = np.meshgrid(x, y)
# Define the function to plot: z = f(x, y) = x^2 + y^2
Z = X**2 + Y**2
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
edgecolor='none')
# Add labels and title with mathematical notation
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title(r'Surface Plot of $f(x, y) = x^2 + y^2$')
# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
# Show the plot
plt.show()



