'''
Created by Barnabás PIRI
Aeroelasticity - 2024. 02. 14.
'''
#%% Package imports

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.style.use('https://raw.githubusercontent.com/barnabaspiri/matplotlib_styles/main/PB_white_latex.mplstyle')


#%% Define the Van der Pol oscillator function

def van_der_pol(t, y, mu):
    """
    t: time
    y: vector of state variables [x1, x2]
    mu: van der Pol parameter
    """
    x1, x2 = y
    dxdt = x2
    dvdt = mu * (1 - x1**2) * x2 - x1
    return [dxdt, dvdt]

# Numerical conditions

y0 = [0.1, -0.1]  # Initial values for [x, v]
t_0 = 0          # Start time of the simulation
t_end = 20       # End time of the simulation
n_eval = 1000    # Number of evaluation points in the timespan
mu = 1       # van der Pol parameter

# Solve the Van der Pol oscillator

sol = solve_ivp(van_der_pol, (t_0, t_end), y0, args=(mu,), t_eval=np.linspace(t_0, t_end, n_eval))
t = sol.t
x = sol.y[0]
v = sol.y[1]

#%% Plot the solutions

plt.figure(1)
plt.plot(t, x, '-', linewidth=1.25)
plt.xlabel("$\\mathrm{Time}, ~ t ~ \\mathrm{(s)}$")
plt.ylabel("$\\mathrm{Position}, ~ x ~ \\mathrm{(m)}$")
plt.savefig('van_der_pol_position.pdf', dpi=300)

plt.figure(2)
plt.plot(t, v, '-', linewidth=1.25, color='#D99300')
plt.xlabel("$\\mathrm{Time}, ~ t ~ \\mathrm{(s)}$")
plt.ylabel("$\\mathrm{Velocity}, ~ v ~ \\mathrm{(m/s)}$")
plt.savefig('van_der_pol_velocity.pdf', dpi=300)

plt.figure(3)
plt.plot(x, v, '-', linewidth=1.25, color='#2E891E')
plt.xlabel("$\\mathrm{Position}, ~ x ~ \\mathrm{(m)}$")
plt.ylabel("$\\mathrm{Velocity}, ~ v ~ \\mathrm{(m/s)}$")
plt.savefig('van_der_pol_phase_plot.pdf', dpi=300)

#%% Creating an animation of the phase plot

from matplotlib.animation import FuncAnimation

def update_plot(i):
    global mu
    mu = 0.1 + 0.01 * i
    print(mu)
    sol = solve_ivp(van_der_pol, (t_0, t_end), y0, args=(mu,), t_eval=np.linspace(t_0, t_end, n_eval))
    line.set_data(sol.y[0], sol.y[1])
    mu_text.set_text('$\\mu = {:.2f}$'.format(mu))
    return line, mu_text
     
# Create the figure and axis

fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title('$\mathrm{Phase ~ plot ~ of ~ the ~ Van ~ der ~ Pol ~ oscillator ~ for ~ } \\mu = 0.1 ~ ... ~ 1.0$', fontsize=9)

# Initialize the object

line, = ax.plot([], [], '-', color='#2E891E')
mu_text = ax.text(2.1, 2.5, '', fontsize=9)

# Animate the plot
mu = 0.1
ani = FuncAnimation(fig, update_plot, frames=91, interval=70, blit=False)

plt.xlabel('$\\mathrm{Position}, ~ x ~ \\mathrm{(m)}$')
plt.ylabel('$\\mathrm{Velocity}, ~ v ~ \\mathrm{(m/s)}$')
plt.show()

# Save animation as GIF
ani.save('van_der_pol_animation.gif', writer='pillow', dpi=300)

plt.show()
