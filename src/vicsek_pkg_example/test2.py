import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class VicsekModel:
    def __init__(self, n=500):
        self.n = n
        self.r = np.random.random((self.n, 2))
        self.theta = np.random.random(self.n) 

        self.L = 1.0     # Box size
        self.d = 0.2     # Increased radius slightly for better visual alignment
        self.v = 0.02    # Velocity
        self.dt = 1.0
        self.eta = 0.05   # Noise
        self.running = True

    @staticmethod
    def distances(r, i):
        """
        Distances from the ith entry
        Parameters
        -----------
        r   :   np.array
                numpy array with point coordinates for n entries
        i   :   int
                index for i-th vector entry
        Returns
        -----------
        np.array
            distances to i-th entry
        """
        return np.linalg.norm(r - r[i], axis=1)
    
    @staticmethod
    def neighbors_idx(distances, d): 
        """
        Return the indices of particles that are within a given distance.

        This static method identifies all neighbors of a particle based on a
        distance array and a specified cutoff distance `d`.

        Parameters
        ----------
        distances : ndarray of shape (n,)
        Array containing distances from a reference particle to all other particles.
        d : float
        Cutoff distance to determine neighbors.

        Returns
        -------
        ndarray of int
        Indices of particles whose distance is less than `d`.

        Notes
        -----
        - Typically used in the Vicsek model to find neighboring particles
        for computing local average orientations.
        - Does not include the reference particle itself unless its distance is < `d`.
        """ 
 

        return np.where(distances < d)[0]

    def update_model(self):
        """
        Update the positions and orientations of all particles in the Vicsek model.

        This method performs one time step of the Vicsek model, including:
        - Computing the average orientation of neighbors within a distance `d`.
        - Adding uniform noise to the orientation.
        - Updating the particle positions based on the updated orientations and velocity `v`.
        - Applying periodic boundary conditions to keep particles within the simulation box.

        Notes
        -----
        - Distances are computed using `VicsekModel.distances`.
        - Neighbors are determined using `VicsekModel.neighbors_idx`.
        - Orientations are normalized to the range [0, 1].
        - Positions are wrapped around using modulo `L` for periodic boundaries.

        Updates
        -------
        self.theta : ndarray of shape (n,)
        Updated orientations of all particles.
        self.r : ndarray of shape (n, 2)
        Updated positions of all particles.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        new_theta = self.theta.copy()
        
        for i in range(self.n):
            # Distance with Periodic Boundary Conditions can be complex, 
            # but keeping your original logic for simplicity:
            #distances = np.linalg.norm(self.r - self.r[i], axis=1)
            distances = VicsekModel.distances(self.r, i)
            #neighbors_idx = np.where(distances < self.d)[0]
            neighbors_idx = VicsekModel.neighbors_idx(distances, self.d)
            
            if len(neighbors_idx) > 0:
                angles = self.theta[neighbors_idx] * 2 * np.pi
                avg_sin = np.mean(np.sin(angles))
                avg_cos = np.mean(np.cos(angles))
                
                avg_theta = np.arctan2(avg_sin, avg_cos)
                noise = self.eta * (np.random.rand() * 2 - 1) # Uniform noise
                # Normalize back to [0, 1] range
                new_theta[i] = ((avg_theta / (2 * np.pi)) + noise) % 1.0

        self.theta = new_theta
        
        # Update positions
        dx = self.v * np.cos(2 * np.pi * self.theta)
        dy = self.v * np.sin(2 * np.pi * self.theta)
        
        self.r[:, 0] = (self.r[:, 0] + dx) % self.L
        self.r[:, 1] = (self.r[:, 1] + dy) % self.L

# --- Animation Setup ---
if __name__ == "__main__":
    model = VicsekModel(n=500)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    # Initial direction vectors for Quiver
    u = np.cos(2 * np.pi * model.theta)
    v = np.sin(2 * np.pi * model.theta)

    # Quiver plot: positions (r), directions (u, v), and color by theta
    q = ax.quiver(model.r[:, 0], model.r[:, 1], u, v, model.theta, 
                  cmap='hsv', clim=(0, 1), pivot='middle', width=0.005)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    def update(frame):
        if model.running:
            model.update_model()
            
            # Update Quiver Positions
            q.set_offsets(model.r)
            
            # Update Quiver Directions
            u_new = np.cos(2 * np.pi * model.theta)
            v_new = np.sin(2 * np.pi * model.theta)
            q.set_UVC(u_new, v_new, model.theta) # Third arg updates colors
            
        return q,

    # UI Buttons
    class Visualizer:
        def toggle(self, event):
            model.running = not model.running
            btn_stop.label.set_text('Start' if not model.running else 'Stop')

    vis = Visualizer()
    ax_stop = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn_stop = Button(ax_stop, 'Stop')
    btn_stop.on_clicked(vis.toggle)

    ani = FuncAnimation(fig, update, interval=30, blit=True, save_count=5)
    plt.show()