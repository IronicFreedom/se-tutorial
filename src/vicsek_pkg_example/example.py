import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import click

@click.command()
@click.option('-n', '--num', default=200, help='Number of elements.')

def test(num):
    """Simple function that returns the CLI --num input"""
    click.echo(f"Input was this: {num}")

plt.rcParams['animation.embed_limit'] = 300

n = 200
d = 0.01
v = 0.01
dt = 1
eta = 0.1

r = np.random.random((n, 2))
theta = np.random.random(n)

fig, ax = plt.subplots(figsize=(6, 6))

x = r[:, 0]
y = r[:, 1]
u = np.cos(2 * np.pi * theta)
vv = np.sin(2 * np.pi * theta)

q = ax.quiver(x, y, u, vv, angles='xy')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Vicsek Model")

counter = 0


def distance(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum())
# --- Control variables ---
running = True  # whether animation is running

def update_model():
    global r, theta, counter
    if running:
        for i in range(n):
            sum_sin = 0
            sum_cos = 0
            neighbours = 0

            for j in range(n):
                if i != j:
                    if distance(r[i], r[j]) < d:
                        theta_j = 2 * np.pi * theta[j]
                        sum_sin = sum_sin + np.sin(theta_j)
                        sum_cos = sum_cos + np.cos(theta_j)
                        neighbours = neighbours + 1

            if neighbours > 0:
                avg_theta = np.arctan2(sum_sin / neighbours, sum_cos / neighbours)
                theta[i] = (avg_theta / (2 * np.pi)) + eta * (np.random.rand() - 0.5)

            dx = v * dt * np.cos(2 * np.pi * theta[i])
            dy = v * dt * np.sin(2 * np.pi * theta[i])

            r[i, 0] = r[i, 0] + dx
            r[i, 1] = r[i, 1] + dy

            if r[i, 0] > 1:
                r[i, 0] = 0
            if r[i, 1] > 1:
                r[i, 1] = 0
            if r[i, 0] < 0:
                r[i, 0] = 1
            if r[i, 1] < 0:
                r[i, 1] = 1

            counter = counter + 1


def animate(frame):
    global q

    update_model()

    x = []
    y = []
    u = []
    vv = []

    for i in range(n):
        x.append(r[i, 0])
        y.append(r[i, 1])
        u.append(np.cos(2 * np.pi * theta[i]))
        vv.append(np.sin(2 * np.pi * theta[i]))

    q.set_offsets(np.c_[x, y])
    q.set_UVC(u, vv)

    print("frame", frame, "counter", counter)

    return q,

# --- Button callbacks ---
def stop(frame):
    global running
    running = False
    print("Animation stopped")

def cont(frame):
    global running
    running = True
    print("Animation continued")

# --- Create buttons ---
ax_stop = plt.axes([0.65, 0.90, 0.15, 0.075])
ax_continue = plt.axes([0.80, 0.90, 0.15, 0.075])
btn_stop = Button(ax_stop, "Stop")
btn_continue = Button(ax_continue, "Continue")
btn_stop.on_clicked(stop)
btn_continue.on_clicked(cont)

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()
test()
