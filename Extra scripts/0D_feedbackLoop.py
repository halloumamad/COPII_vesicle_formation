import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import cm

plot_input = 0
plot_sar = 0
plot_sec = 0
plot_SarSec = 0
plot_many_cargo = 1
plot_cargo_gaussian = 0
plot_alpha_non_lin = 0
plot_cargo_dist = 0

run_sim = 1

fontsize = 16
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)

# %% parametersettings:
cargo_factor = 1#20

km1 = 0.5
km2 = 0.5
km3 = 0.5
km4 = 0.5

k1 = 20#20#8#15#35#54#3.25  #0.65/0.2#3  # inhibition of Sec on Sar
k2 = 1#1#1#5#15#0.75*10  #0.15/0.2#1 # upregulation of Sec
k4 = 1#1#3#10#15#38#1.125  #0.25/0.2#1  # positive feedback of Sar1 from Sar1.
print(k1,k2,k4)
dt = 0.025  # når den sættes til 0.5 ser det mærkelig ud.
maxTimesteps = 500

# preset parameters
gam1 = 1
gam2 = 4  # in reality we have changed this by multiplying by "factor".
AT = 1
ET = 1

def dA_dt(RacOneCell, A, E):# , lA=0, rA=0):
    return cargo_factor * (RacOneCell * (AT - A) / (AT - A + km1)  # Rac1 upregulates Sar1active
                 - A / gam1 - k1 * E * A / (A + km2)  # basal removal and inhibition by Sec23
                 + k4 * A * (AT - A) / (AT - A + km4) ) # positive feedback from Sar1 on Sar1
                 #+ k4 * lA * (AT - A) / (AT - A + km4)  # positive feedback from left neighbour
                 #+ k4 * rA * (AT - A) / (AT - A + km4))  # positive feedback from right neighbour... will this give a wave, or do we need refraction time.


def dE_dt(A, E):
    return 1 / cargo_factor * (k2 * A * (ET - E) / (ET - E + km3) - E / gam2)

# initialising data storage:
Rac = np.zeros(maxTimesteps)
A = np.zeros(maxTimesteps)
E = np.zeros(maxTimesteps)

time = np.arange(dt, maxTimesteps * dt+dt, dt)

# initial conditions
hRac = 100  #5
RacOneCell = 1 / (1 + (2 / time) ** hRac)  # time**hRac/(time**hRac+2**hRac)
#RacOneCell =


if plot_alpha_non_lin == 1:
    hAlp = 20
    def alpha_non_lin(Ai):
        return Ai / (1 + (0.6 / Ai) ** hAlp)
    Ai = np.linspace(0.025, 0.8, 1000)
    alpha = alpha_non_lin(Ai)
    plt.plot(Ai, alpha)
    plt.plot(Ai, Ai)
    plt. xlabel('Sar concentration')
    plt.ylabel('alpha')
    plt.show()

if plot_cargo_dist == 1:
    d_lim = 10
    hCar = 6
    cargo_height = 8
    def cargo_dist_func(d):
        return d_lim**hCar/(d**hCar+d_lim**hCar) * cargo_height
    d = np.linspace(0, 30, 500)
    cargo_dist = cargo_dist_func(d)
    print(cargo_dist[0])
    plt.plot(d, cargo_dist)
    plt.plot(d, np.ones_like(d)*0.1)
    plt.xlabel('Distance from center')
    plt.ylabel('Cargo')
    plt.show()


#Rac[:, int(np.floor(nUnits / 2))] = RacOneCell
if run_sim == 1:
    for i in range(maxTimesteps - 1):
        #left = np.roll(A, 1, axis=1)
        #right = np.roll(A, -1, axis=1)

        # for j in range(nUnits):
        A[i + 1] = (A[i] + dA_dt(RacOneCell[i], A[i], E[i]) * dt)
        E[i + 1] = E[i] + dE_dt(A[i], E[i]) * dt


print('ratio of second SS og A compared with highest A', A[-1]/max(A))
print('max(A)=', max(A))
index_max_A = np.argmax(A)
dist_end = 0.20 # in percent
index_bottom_A = np.where(np.logical_and(A>A[-1]-dist_end*A[-1], A<A[-1]+dist_end*A[-1]))
#print(index_bottom_A)
#print('Distance from top to peak bottom', index_bottom_A[0][0]-index_max_A)


if plot_input == 1:
    plt.figure()
    plt.plot(time, RacOneCell, label='input')
    plt.legend(fontsize=fontsize)
    plt.show()

if plot_sar == 1:
    plt.figure()
    plt.plot(time, A, label='Sar1')
    #plt.plot(index_max_A*dt, max(A), 'o', label='max A')
    #x_index_bottom = np.array(index_bottom_A)*dt
    #plt.plot(x_index_bottom[0], A[index_buottom_A])
    #plt.plot(index_bottom_A[0][0]*dt, A[index_bottom_A[0][0]], 'o', label='bottom')
    plt.legend(fontsize=fontsize)
    plt.show()

if plot_sec == 1:
    plt.figure()
    plt.plot(time, E, label='Sec')
    plt.legend(fontsize=fontsize)
    plt.show()

if plot_SarSec == 1:
    plt.figure()
    plt.plot(time, A, label='Sar1')
    plt.plot(time, E, label='Sec')
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.xlabel('time', fontsize=fontsize)
    plt.ylabel('Concentration', fontsize=fontsize)
    plt.tight_layout()
    plt.show()

if plot_many_cargo == 1:
    cargo_array = np.array([0.5, 1, 2, 3, 4, 5, 6]) #np.linspace(0.1, 1, 6) # np.array([2,3,4,5,6,7,8])
    #cargo_array2 = np.array([10])#np.linspace(7, 20, 4)
    #cargo_array = np.concatenate((cargo_array, cargo_array2))


    fig, axs = plt.subplots(2)
    #axs[0].set_title('Sar1', fontsize=fontsize)
    axs[0].set_ylabel('Sar1', fontsize=fontsize)
    #axs[1].set_title('Sec', fontsize=fontsize)
    axs[1].set_ylabel('Sec', fontsize=fontsize)

    for j in range(len(cargo_array)):
        cargo_factor = cargo_array[j]
        for i in range(maxTimesteps - 1):
            A[i + 1] = (A[i] + dA_dt(RacOneCell[i], A[i], E[i]) * dt)
            E[i + 1] = E[i] + dE_dt(A[i], E[i]) * dt
        axs[0].plot(time, A, label=f"$k^c$ = {cargo_factor:.1f}")
        axs[1].plot(time, E, label=f"$k^c$ = {cargo_factor:.1f}")

    #axs[0].plot(time, np.ones(len(time)) * 0.2, '--k')

    #axs[1].plot(time, np.ones(len(time)) * 0.05, '--k')
    axs[1].plot(time, np.ones(len(time)) * 0.10, '--k')
    #axs[1].plot(time, np.ones(len(time)) * 0.125, '--k')
    #axs[1].plot(time, np.ones(len(time)) * 0.15, '--k')
    plt.xlabel('Time (s)', fontsize=fontsize)
    #plt.ylabel('Concentration', fontsize=fontsize)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right', fontsize=fontsize-2)
    #fig.legend(loc ="upper right")
    plt.show()

if plot_cargo_gaussian == 1:
    cargo_sigma = 16
    x = np.arange(-28,28, 0.2)
    X, Y = np.meshgrid(x, x)
    cargo_gaus = np.exp( - (X**2 + Y**2) / (2*cargo_sigma**2) ) * 6

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, cargo_gaus, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

