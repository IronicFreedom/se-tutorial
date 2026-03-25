import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import click

# Configuration Class
class Vicsek:
    def __init__(self, n, d, v, dt, eta):
        self.n = n        # Number of particles
        self.d = d        # Interaction radius
        self.v = v        # Constant velocity
        self.dt = dt      # Time step
        self.eta = eta    # Noise (randomness)

@click.command()
@click.option('-n', '--num', default=200, help='Number of elements.')
@click.option('-eta', '--noise', default=0.1, help='Noise level (0 to 1).')
def run_simulation(num, noise):
    """Runs the Vicsek Model simulation."""
    
    vicsek = Vicsek(n=num, d=0.05, v=0.03, dt=1.0, eta=noise)

    # Initialize positions (r) and angles (theta in radians)
    r = np.random.rand(vicsek.n, 2)
    theta = np.random.uniform(-np.pi, np.pi, vicsek.n)

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2) # Make room for buttons

    # Quiver plot setup
    q = ax.quiver(r[:, 0], r[:, 1], np.cos(theta), np.sin(theta), 
                  pivot='mid', color='teal', units='width', scale=30)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Vicsek Model (N={vicsek.n}, eta={vicsek.eta})")

    running = True

    def update_model():
        nonlocal r, theta
        if not running:
            return

        # 1. Calculate Distances (Vectorized)
        # Uses broadcasting to find distance between all pairs
        dx = r[:, np.newaxis, 0] - r[np.newaxis, :, 0]
        dy = r[:, np.newaxis, 1] - r[np.newaxis, :, 1]
        
        # Periodic Boundary Conditions for distance
        dx = (dx + 0.5) % 1 - 0.5
        dy = (dy + 0.5) % 1 - 0.5
        dist_sq = dx**2 + dy**2

        # 2. Find neighbors within radius d
        mask = dist_sq < vicsek.d**2

        # 3. Average the angles of neighbors
        sum_sin = np.dot(mask, np.sin(theta))
        sum_cos = np.dot(mask, np.cos(theta))
        avg_theta = np.arctan2(sum_sin, sum_cos)

        # 4. Add noise and update theta
        theta = avg_theta + vicsek.eta * np.random.uniform(-np.pi, np.pi, vicsek.n)

        # 5. Update positions
        r[:, 0] += vicsek.v * np.cos(theta) * vicsek.dt
        r[:, 1] += vicsek.v * np.sin(theta) * vicsek.dt

        # 6. Apply Periodic Boundary Conditions
        r %= 1

    def animate(frame):
        update_model()
        q.set_offsets(r)
        q.set_UVC(np.cos(theta), np.sin(theta))
        return q,

    # --- Button Logic ---
    def stop(event):
        nonlocal running
        running = False

    def cont(event):
        nonlocal running
        running = True

    ax_stop = plt.axes([0.3, 0.05, 0.15, 0.06])
    ax_cont = plt.axes([0.5, 0.05, 0.15, 0.06])
    btn_stop = Button(ax_stop, "Stop")
    btn_cont = Button(ax_cont, "Start")
    btn_stop.on_clicked(stop)
    btn_cont.on_clicked(cont)

    ani = FuncAnimation(fig, animate, interval=30, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()