import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n = 500
r_interact = 0.1  # Increased for visible swarming
v = 0.03
eta = 0.05          # Noise magnitude
dt = 1

# Initialize
pos = np.random.rand(n, 2)
theta = np.random.uniform(-np.pi, np.pi, n)

fig, ax = plt.subplots(figsize=(6, 6))
q = ax.quiver(pos[:, 0], pos[:, 1], np.cos(theta), np.sin(theta), color='b')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def update_model():
    global pos, theta
    
    # 1. Calculate all-to-all distances using broadcasting
    # dx.shape will be (n, n)
    dx = pos[:, 0, np.newaxis] - pos[:, 0]
    dy = pos[:, 1, np.newaxis] - pos[:, 1]
    
    # Apply periodic distance (closest path in a wrap-around box)
    dx = (dx + 0.5) % 1 - 0.5
    dy = (dy + 0.5) % 1 - 0.5
    
    dist = np.sqrt(dx**2 + dy**2)
    
    # 2. Find neighbors within interaction radius
    adj = dist < r_interact
    
    # 3. Average the angles of neighbors
    new_theta = theta.copy()
    for i in range(n):
        neighbors_theta = theta[adj[i]]
        avg_sin = np.mean(np.sin(neighbors_theta))
        avg_cos = np.mean(np.cos(neighbors_theta))
        new_theta[i] = np.arctan2(avg_sin, avg_cos)
    
    # 4. Add noise and update heading
    theta = new_theta + eta * np.random.uniform(-np.pi, np.pi, n)
    
    # 5. Move particles
    pos[:, 0] += v * np.cos(theta) * dt
    pos[:, 1] += v * np.sin(theta) * dt
    
    # 6. Periodic Boundary Conditions
    pos = pos % 1

def animate(frame):
    update_model()
    q.set_offsets(pos)
    q.set_UVC(np.cos(theta), np.sin(theta))
    return q,

#if __name__ == "__main__":
ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=True)
plt.show()