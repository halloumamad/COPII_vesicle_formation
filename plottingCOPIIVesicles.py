import numpy as np
from mayavi import mlab
import pickle
import scipy.io
import glob
import os
import sys
from pathlib import Path
from tvtk.api import tvtk
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import cycler

print(sys.argv)
name = 1        # same number as the data was called.


save_p_q = 1       # Need to put this to 1, if the data was saved including p and q
save_beta = 1       # If beta was saved this need to be set to 1
save_cargo = 0      # If cargo was saved in its own variable the must be set to 1 (most often it wasn't)
save_V = 0
save_lam = 0

plot_q = 0          # If the q-vector should appear on the plots set this to 1 (only possible if q was saved)
choose_color_q = 0
centralUnits = [] #, 70, 121, 96, 131, 3, 6]#[18, 53, 76, 23, 39, 2, 38] #[76, 18, 53, 39, 140, 93, 110] # [2, 18, 53, 38, 5, 49, 68, 57]#
plot_colorbar = 0
fixed_cbar = 1     # Set this to 1 if the colorbar should be fixed for all frames.
plot_axes = 1
plot_V = 0

RGB = 0             # Set to 1 if plots should be colored by the RGB convention, Sar1 is red and Sec23 is green, blue is not used.
colorIs = 'highE'  # Choose 'alpha' or 'A' or 'E' or 'highA' or 'highE' or 'AandE' or 'all_limits' or ''within_radius', or 'radius_highE' or 'radia' or 'lambda', or 'twoLimits'
Alim_q = 0.03#0.2
Alim_Sar = 0.2#0.4
Elim = 0.05#0.4
r1_lambda = 5       # only important if 'within_radius' is chosen
r2_lambda = 10
centerUnit_q_zero = 1 # set this to 1 or 0. Set to 1 if the center unit always have lam1=1

grundFarve = None  # Set to None for gray, or set to (1,1,1) for white

plotSingle = 0
saveNr = 390 # when in time, when I want a single plot
animation = 1
saveAnim = 0

plotALineplots = 0
plot_A_E_spread_sum = 0
plot_A_E_spread_count = 0
find_vesicle_size = 0

filename = f'myData/data_{name}'
abspath = Path(filename).absolute()
print(abspath)

if ((save_p_q == 1) and (save_cargo == 1) and (save_beta != 1) and (save_V != 1) and (save_lam != 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, P, Q, C, nSaves, save_every  = pickle.load(f)
if ((save_p_q == 1) and (save_cargo != 1) and (save_beta != 1) and (save_V != 1) and (save_lam != 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, P, Q, nSaves, save_every  = pickle.load(f)
if ((save_p_q == 1) and (save_cargo != 1) and (save_beta != 1) and (save_V != 1) and (save_lam == 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, lam, P, Q, nSaves, save_every  = pickle.load(f)
if ((save_p_q == 1) and (save_cargo != 1) and (save_beta == 1) and (save_V != 1) and (save_lam != 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, beta, P, Q, nSaves, save_every  = pickle.load(f)
if ((save_p_q != 1) and (save_cargo != 1) and (save_beta != 1) and (save_V != 1) and (save_lam != 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, nSaves, save_every  = pickle.load(f)
if ((save_p_q == 1) and (save_cargo != 1) and (save_beta == 1) and (save_V == 1) and (save_lam != 1)):
    with open(str(abspath), 'rb') as f:
        X, A, E, alpha, beta, P, Q, V, nSaves, save_every  = pickle.load(f)

X = X[1:, :, :]
A = A[1:, :]
E = E[1:, :]

if plot_V == 1:
    plt.figure()
    plt.plot(np.arange(len(V)),V)
    plt.show()
    print('mean of potential of the last 20 iterations', np.mean(V[-21:-1]))
    print('std:', np.std(V[-21:-1]))
    sys.exit()


leaveOutUnits = 1
leaveOutAboveUnit = 910
if leaveOutUnits == 1:
    X = X[:, :leaveOutAboveUnit, :]
    A = A[:, :leaveOutAboveUnit]
    E = E[:, :leaveOutAboveUnit]
    #P = P[:, :leaveOutAboveUnit, :]
    #Q = Q[:, :leaveOutAboveUnit, :]
    alpha = alpha[:, :leaveOutAboveUnit]
    if save_lam == 1:
        lam = lam[:, :leaveOutAboveUnit, :]
    #beta = beta[:, :leaveOutAboveUnit]


Xshape = np.shape(X)
nUnits = Xshape[1]



scalefactor = 1.5
ballres = 10
ballmode='sphere'
scalefactor_q = 2#1.5

if plotSingle == 1:

    mlab.figure(figure=None, bgcolor=grundFarve, fgcolor=None, engine=None, size=(1000, 800)) # bgcolor=None for gray or bgcolor=(1,1,1) for white

    if RGB != 1:

        if colorIs == 'A':
            colors = A[saveNr, :]
            colormap = "autumn"
        if colorIs == 'E':
            colors = E[saveNr, :]
            colormap = "autumn"
        if colorIs == 'alpha':
            colors = alpha[saveNr, :]
            colormap = "autumn"
        if colorIs == 'highA':
            colors = np.zeros(nUnits)
            colors[A_color[saveNr, :] > Alim_Sar] = 1
        if colorIs == 'highE':
            colors = np.ones(nUnits)*0
            colors[E[saveNr, :] > Elim] = 1
            colormap = "autumn"
        if colorIs == 'AandE':
            colors = np.zeros(nUnits)
            # isotropyMask = ((A[saveNr, :] < Alim) & (E[saveNr, :] < Elim)) | (E[saveNr, :] > Elim)
            anisotropyMask = (A[saveNr, :] > Alim_Sar) & (E[saveNr, :] < Elim)
            colors[anisotropyMask] = 1
        if colorIs == 'all_limits':
            colors = np.ones(nUnits)*0
            Alim_q_mask = (A[saveNr, :] > Alim_q) & (A[saveNr, :] < Alim_Sar)
            Alim_Sar_mask = (A[saveNr, :] > Alim_Sar) & (E[saveNr, :] < Elim)
            Elim_mask = E[saveNr, :] > Elim
            colors[Alim_q_mask] = 0.7
            colors[Alim_Sar_mask] = 0.4#0.5
            colors[Elim_mask] = 1      #0.75
            if centerUnit_q_zero == 1:
                for index in centralUnits:
                    colors[index] = 0.5
            colormap = "autumn"#"GnBu"#"RdYlBu"#"summer"#"coolwarm"#"jet"
        if colorIs == 'within_radius':
            colors = np.zeros(nUnits)
            d = np.sqrt( np.sum(X[saveNr, :, :]**2, axis=1) )
            radius_mask = d < r1_lambda
            colors[radius_mask] = 1
            colormap = "autumn"
        if colorIs == 'radius_highE':
            colors = np.zeros(nUnits)
            d = np.sqrt(np.sum(X[0, :, :] ** 2, axis=1))
            radius_mask = d < r1_lambda
            colors[radius_mask] = 0.5
            colors[E[saveNr, :] > Elim] += 0.25
            colormap = "jet"
        if colorIs == 'radia':
            colors = np.ones(nUnits)*0
            d = np.sqrt(np.sum(X[saveNr, :, 0:2] ** 2, axis=1))
            #no_bend_mask = d > r2_lambda
            aniso_mask = (d < r2_lambda) & (d > r1_lambda)
            iso_mask = (d < r1_lambda) | (X[saveNr, :, 2] < -5)
            colors[aniso_mask] = 0.9
            colors[iso_mask] = 0.7
            colormap = "jet"
        if colorIs == 'lambda':
            colors = np.ones(nUnits) * 0
            aniso_mask = (lam[saveNr, :, 0] == 0.5)
            iso_mask = ((lam[saveNr, :, 0] == 1) & (X[saveNr, :, 2]<-5))
            colors[aniso_mask] = 0.9
            colors[iso_mask] = 0.7
            colormap = "jet"



        current_view = mlab.view()
        print(current_view)
        mlab.view(0, 180, distance=0.1)
        #mlab.view(270, 90, distance=0.1)
        p3d_plot = mlab.points3d(X[saveNr, :, 0], X[saveNr, :, 1], X[saveNr, :, 2], colors, colormap=colormap, scale_factor=scalefactor, scale_mode='none', resolution=ballres, mode=ballmode)

        if fixed_cbar == 1:
            lut_manager = mlab.colorbar()
            lut_manager.data_range = (0, 1)

        if plot_axes == 1:
            mlab.axes(x_axis_visibility=True)
        if plot_colorbar == 1:
            mlab.colorbar()
        if plot_q == 1:
            print('max', max(colors))
            print('min', min(colors))
            #sys.exit()
            q_mask = (colors > 0.5) #& (colors < 0.75)
            if choose_color_q == 1:
                #colors_q = (0.2, 0.5, 0.9)#np.ones((nUnits, 3))
                #colors_q[:, 0] = colors
                #colors_q[:, 1] = 0
                #colors_q[:, 2] = 0
                quiv_plot = mlab.quiver3d(X[saveNr, q_mask, 0], X[saveNr, q_mask, 1], X[saveNr, q_mask, 2], Q[saveNr, q_mask, 0], Q[saveNr, q_mask, 1], Q[saveNr, q_mask, 2], scalars = colors[q_mask]*100, scale_factor=scalefactor_q, mode='arrow')
            else:
                quiv_plot = mlab.quiver3d(X[saveNr, q_mask, 0], X[saveNr, q_mask, 1], X[saveNr, q_mask, 2]+1, Q[saveNr, q_mask, 0], Q[saveNr, q_mask, 1], Q[saveNr, q_mask, 2], colormap="binary", scale_factor=scalefactor_q, mode='arrow' )  # colormap="GnBu",

        #mlab.savefig('myData/single_plots/data345tid99.png')

    if RGB == 1:
        p3d_plot = mlab.points3d(X[saveNr, :, 0], X[saveNr, :, 1], X[saveNr, :, 2], scale_factor=scalefactor, scale_mode='none', mode=ballmode)
        colors = np.ones((nUnits, 3))  # LUT: RGBA, where the last number A is the opacity.
        colors[:, 0] = A_color[saveNr, :] * 255  # The amount of red is determined by the amount of Sar1,
        colors[:, 1] = E_color[saveNr, :] * 255  # The amount of green is determined by the amount of Sec23
        colors[:, 2] = 80  # I don't need three colors right now.
        # colors[:,3] =  255 # making it nontransparent
        sc = tvtk.UnsignedCharArray()
        sc.from_array(colors)

        p3d_plot.mlab_source.dataset.point_data.scalars = sc
        p3d_plot.mlab_source.dataset.modified()

    mlab.show()

if animation == 1:
    mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(1000, 800))
    #mlab.figure(figure=None, bgcolor=grundFarve, fgcolor=None, engine=None, size=(1000, 800))

    if RGB != 1:
        if colorIs == 'A':
            colors = A_color[0, :]
            colormap = "autumn"
        if colorIs == 'E':
            colors = E_color[0, :]
            colormap = "summer"
        if colorIs == 'alpha':
            colors = alpha[0, :]
            colormap = "autumn"
        if colorIs == 'highA':
            colors = np.zeros(nUnits)
            colors[A[0, :] > Alim_q] = 1
            colormap = "winter"
        if colorIs == 'highE':
            colors = np.zeros(nUnits)
            colors[E[0, :] > Elim] = 1
            colormap = "autumn"
        if colorIs == 'AandE':
            # isotropyMask = ((A[0, :] < Alim) & (E[0, :] < Elim)) | (E[0, :] > Elim)
            anisotropyMask = (A[0, :] > Alim_Sar) & (E[0, :] < Elim)
            colors = np.zeros(nUnits)
            colors[anisotropyMask] = 1
        if colorIs == 'all_limits':
            colors = np.zeros(nUnits)
            Alim_q_mask = (A[0, :] > Alim_q) & (A[0, :] < Alim_Sar)
            Alim_Sar_mask = (A[0, :] > Alim_Sar) & (E[0, :] < Elim)
            Elim_mask = E[0, :] > Elim
            colors[Alim_q_mask] = 0.25
            colors[Alim_Sar_mask] = 0.5
            colors[Elim_mask] = 0.75
            if centerUnit_q_zero == 1:
                for index in centralUnits:
                    colors[index] = 1
            colormap = "jet"
        if colorIs == 'twoLimits':
            colors = np.zeros(nUnits)
            Alim_Sar_mask = (A[0, :] > Alim_Sar) & (E[0, :] < Elim)
            Elim_mask = E[0, :] > Elim
            colors[Alim_Sar_mask] = 0.9
            colors[Elim_mask] = 0.7
            colormap = "jet"
        if colorIs == 'within_radius':
            colors = np.zeros(nUnits)
            d = np.sqrt( np.sum(X[0, :, :]**2, axis=1) )
            radius_mask = d < r1_lambda
            colors[radius_mask] = 1
            colormap = "autumn"
        if colorIs == 'lambda':
            colors = np.ones(nUnits) * 0
            aniso_mask = (lam[0, :, 0] == 0.5)
            iso_mask = ((lam[0, :, 0] == 1) & (X[0, :, 2]<-5))
            colors[aniso_mask] = 0.9
            colors[iso_mask] = 0.7
            colormap = "jet"

        #current_view = mlab.view()
        #print(current_view)
        mlab.view(0, 0)
        #new_view = mlab.view()
        #print(new_view)
        p3d_plot = mlab.points3d(X[0, :, 0], X[0, :, 1], X[0, :, 2], colors, colormap=colormap, scale_factor=1.5, scale_mode='none')
        mlab.axes()
        if plot_q == 1:
            q_mask = colors > 0
            if choose_color_q == 1:
                #colors_q = (0.2, 0.5, 0.9)#np.ones((nUnits, 3))
                #colors_q[:, 0] = colors
                #colors_q[:, 1] = 0
                #colors_q[:, 2] = 0
                quiv_plot = mlab.quiver3d(X[0, q_mask, 0], X[0, q_mask, 1], X[0, q_mask, 2], Q[0, q_mask, 0], Q[0, q_mask, 1], Q[0, q_mask, 2], scalars = colors[q_mask]*100, scale_factor=scalefactor_q, mode='arrow')
            else:
                quiv_plot = mlab.quiver3d(X[0, q_mask, 0], X[0, q_mask, 1], X[0, q_mask, 2]+1, Q[0, q_mask, 0], Q[0, q_mask, 1], Q[0, q_mask, 2], scale_factor=scalefactor_q, mode='arrow' )

        @mlab.animate
        def anim():
            for i in range(nSaves):
                if colorIs == 'A':
                    colors = A_color[i, :]
                if colorIs == 'E':
                    colors = E_color[i, :]
                if colorIs == 'alpha':
                    colors = alpha[i, :]
                if colorIs == 'highA':
                    colors = np.zeros(nUnits)
                    colors[A[i, :] > Alim_q] = 1
                if colorIs == 'highE':
                    colors = np.zeros(nUnits)
                    colors[E[i, :] > Elim] = 1
                if colorIs == 'AandE':
                    # isotropyMask = ((A[i, :] < Alim) & (E[i, :] < Elim)) | (E[i, :] > Elim)
                    anisotropyMask = (A[i, :] > Alim_Sar) & (E[i, :] < Elim)
                    colors = np.zeros(nUnits)
                    colors[anisotropyMask] = 1
                if colorIs == 'all_limits':
                    colors = np.zeros(nUnits)
                    Alim_q_mask = (A[i, :] > Alim_q) & (A[i, :] < Alim_Sar)
                    anisotropyMask = (A[i, :] > Alim_Sar) & (E[i, :] < Elim)
                    Elim_mask = E[i, :] > Elim
                    colors[Alim_q_mask] = 0.25
                    colors[anisotropyMask] = 0.5
                    colors[Elim_mask] = 0.75
                    if centerUnit_q_zero == 1:
                        for index in centralUnits:
                            colors[index] = 1
                if colorIs == 'twoLimits':
                    colors = np.zeros(nUnits)
                    Alim_Sar_mask = (A[i, :] > Alim_Sar) & (E[i, :] < Elim)
                    Elim_mask = E[i, :] > Elim
                    colors[Alim_Sar_mask] = 0.9
                    colors[Elim_mask] = 0.7
                if (colorIs == 'within_radius'):# and (i == 0):
                    colors = np.zeros(nUnits)
                    d = np.sqrt(np.sum(X[i, :, :] ** 2, axis=1))
                    radius_mask = d < r1_lambda
                    colors[radius_mask] = 0.5
                if colorIs == 'lambda':
                    colors = np.ones(nUnits) * 0
                    aniso_mask = (lam[i, :, 0] == 0.5)
                    iso_mask = ((lam[i, :, 0] == 1) & (X[i, :, 2]<-5))
                    colors[aniso_mask] = 0.9
                    colors[iso_mask] = 0.7
                    colormap = "jet"


                print(i)

                p3d_plot.mlab_source.set(x = X[i, :, 0], y = X[i, :, 1], z = X[i, :, 2], scalars=colors)
                if fixed_cbar == 1:
                    lut_manager = mlab.colorbar()
                    lut_manager.data_range = (0, 1)
                mlab.colorbar()
                #mlab.axes()
                if plot_q ==1:
                    d = np.sqrt(np.sum(X[0, :, :] ** 2, axis=1))
                    #radius_mask = d < r1_lambda
                    q_mask = d < 60 #colors > -1
                    if choose_color_q == 1:
                        #colors_q = (0.2, 0.5, 0.8)#np.ones((nUnits, 3))
                        #colors_q[:, 0] = colors
                        #colors_q[:, 1] = 0
                        #colors_q[:, 2] = 0
                        quiv_plot.mlab_source.reset(x=X[i, q_mask, 0], y=X[i, q_mask, 1], z=X[i, q_mask, 2]+1, u=Q[i, q_mask, 0], v=Q[i, q_mask, 1], w=Q[i, q_mask, 2], scalars=colors[q_mask]*100, scale_factor=scalefactor_q, mode='arrow')
                    else:
                        #print(X[42, q_mask, 0].shape, X[i, q_mask, 1].shape, (X[i, q_mask, 2] + 1).shape, Q[i, q_mask, 0].shape, Q[i, q_mask, 1].shape, Q[i, q_mask, 2].shape)
                        quiv_plot.mlab_source.reset(x = X[i, q_mask, 0], y = X[i, q_mask, 1], z = X[i, q_mask, 2]+1, u=Q[i, q_mask, 0], v=Q[i, q_mask, 1], w=Q[i, q_mask, 2], colormap="GnBu", scale_factor=scalefactor_q)
                yield

    if RGB == 1:
        p3d_plot = mlab.points3d(X[0, :, 0], X[0, :, 1], X[0, :, 2], scale_factor=1.5, scale_mode='none')

        @mlab.animate
        def anim():
            iend = nSaves
            for i in range(iend):
                #print(i)
                p3d_plot.mlab_source.set(x=X[i, :, 0], y=X[i, :, 1], z=X[i, :, 2])
                colors = np.ones((nUnits, 3))  # LUT: RGBA, where the last number A is the opacity.
                colors[:, 0] = A_color[i, :] * 255  # The amount of red is determined by the amount of Sar1,
                colors[:, 1] = E_color[i, :] * 255  # The amount of green is determined by the amount of Sec23
                colors[:, 2] = 0  # I don't need three colors right now.
                # colors[:,3] =  255 # making it nontransparent

                sc = tvtk.UnsignedCharArray()
                sc.from_array(colors)
                p3d_plot.mlab_source.dataset.point_data.scalars = sc
                p3d_plot.mlab_source.dataset.modified()
                yield
    anim()
    mlab.show()





if saveAnim == 1: # copyed from https://stackoverflow.com/questions/24958669/saving-a-mayavi-animation

    mlab.figure(figure=None, bgcolor=grundFarve, fgcolor=None, engine=None, size=(1000, 800))

    fps = 20
    prefix = 'animation'
    ext = '.png'
    padding = len(str(nSaves))

    if RGB != 1:

        try:
            os.mkdir(f'./myData/animations/data_{name}_RGB={RGB}_color={colorIs}_plotq={plot_q}')
        except OSError:
            pass
        out_path = f'./myData/animations/data_{name}_RGB={RGB}_color={colorIs}_plotq={plot_q}'
        out_path = os.path.abspath(out_path)

        # initializing plotting:
        if colorIs == 'A':
            colors = A_color[0, :]
            colormap = "autumn"
        if colorIs == 'alpha':
            colors = alpha[0, :]
            colormap = "autumn"
        if colorIs == 'highA':
            colors = np.zeros(nUnits)
            colors[A_color[0, :] > Alim_q] = 1
            colormap = "winter"
        if colorIs == 'highE':
            colors = np.zeros(nUnits)
            colors[E_color[0, :] > Elim] = 1
            colormap = "autumn"
        if colorIs == 'AandE':
            colors = np.zeros(nUnits)
            # isotropyMask = ((A[0, :] < Alim) & (E[0, :] < Elim)) | (E[0, :] > Elim)
            anisotropyMask = (A[0, :] > Alim) & (E[0, :] < Elim)
            colors[anisotropyMask] = 1
        if colorIs == 'all_limits':
            colors = np.zeros(nUnits)
            Alim_q_mask = (A[0, :] > Alim_q) & (A[0, :] < Alim_Sar)
            Alim_Sar_mask = (A[0, :] > Alim_Sar) & (E[0, :] < Elim)
            Elim_mask = E[0, :] > Elim
            colors[Alim_q_mask] = 0.25
            colors[Alim_Sar_mask] = 0.5
            colors[Elim_mask] = 0.75
            if centerUnit_q_zero == 1:
                for index in centralUnits:
                    colors[index] = 1
            colormap="jet"
        if colorIs == 'twoLimits':
            colors = np.zeros(nUnits)
            Alim_Sar_mask = (A[0, :] > Alim_Sar) & (E[0, :] < Elim)
            Elim_mask = E[0, :] > Elim
            colors[Alim_Sar_mask] = 0.9
            colors[Elim_mask] = 0.7
            colormap = "jet"
        if colorIs == 'lambda':
            colors = np.ones(nUnits) * 0
            aniso_mask = (lam[0, :, 0] == 0.5)
            iso_mask = ((lam[0, :, 0] == 1) & (X[0, :, 2]<-5))
            colors[aniso_mask] = 0.9
            colors[iso_mask] = 0.7
            colormap = "jet"

        #mlab.view(0, 0)
        mlab.view(0, 180)
        p3d_plot = mlab.points3d(X[0, :, 0], X[0, :, 1], X[0, :, 2], colors, colormap=colormap, scale_factor=1.5, scale_mode='none')
        if plot_q == 1:
            q_mask = colors > 0
            if choose_color_q == 1:
                quiv_plot = mlab.quiver3d(X[0, q_mask, 0], X[0, q_mask, 1], X[0, q_mask, 2], Q[0, q_mask, 0], Q[0, q_mask, 1], Q[0, q_mask, 2], scalars=colors[q_mask], scale_factor=scalefactor_q, mode='arrow')
            else:
                quiv_plot = mlab.quiver3d(X[0, q_mask, 0], X[0, q_mask, 1], X[0, q_mask, 2] + 1, Q[0, q_mask, 0], Q[0, q_mask, 1], Q[0, q_mask, 2], scale_factor=scalefactor_q, mode='arrow')

        for i in range(nSaves):
            # Creating new view
            if colorIs == 'A':
                colors = A_color[i, :]
            if colorIs == 'alpha':
                colors = alpha[i, :]
            if colorIs == 'highA':
                colors = np.zeros(nUnits)
                colors[A_color[i, :] > Alim_q] = 1
            if colorIs == 'highE':
                colors = np.zeros(nUnits)
                colors[E_color[i, :] > Elim] = 1
            if colorIs == 'AandE':
                colors = np.zeros(nUnits)
                # isotropyMask = ((A[i, :] < Alim) & (E[i, :] < Elim)) | (E[i, :] > Elim)
                anisotropyMask = (A[i, :] > Alim) & (E[i, :] < Elim)
                colors[anisotropyMask] = 1
            if colorIs == 'all_limits':
                colors = np.zeros(nUnits)
                Alim_q_mask = (A[i, :] > Alim_q) & (A[i, :] < Alim_Sar)
                anisotropyMask = (A[i, :] > Alim_Sar) & (E[i, :] < Elim)
                Elim_mask = E[i, :] > Elim
                colors[Alim_q_mask] = 0.25
                colors[anisotropyMask] = 0.5
                colors[Elim_mask] = 0.75
                if centerUnit_q_zero == 1:
                    for index in centralUnits:
                        colors[index] = 1
            if colorIs == 'twoLimits':
                colors = np.zeros(nUnits)
                Alim_Sar_mask = (A[i, :] > Alim_Sar) & (E[i, :] < Elim)
                Elim_mask = E[i, :] > Elim
                colors[Alim_Sar_mask] = 0.9
                colors[Elim_mask] = 0.7
            if colorIs == 'lambda':
                colors = np.ones(nUnits) * 0
                aniso_mask = (lam[i, :, 0] == 0.5)
                iso_mask = ((lam[i, :, 0] == 1) & (X[i, :, 2] < -5))
                colors[aniso_mask] = 0.9
                colors[iso_mask] = 0.7
                colormap = "jet"

            p3d_plot.mlab_source.set(x = X[i, :, 0], y = X[i, :, 1], z = X[i, :, 2], scalars=colors)
            if fixed_cbar == 1:
                lut_manager = mlab.colorbar()
                lut_manager.data_range = (0, 1)
            if plot_colorbar == 1:
                mlab.colorbar()
            if plot_q == 1:
                q_mask = colors > 0
                if choose_color_q == 1:
                    quiv_plot.mlab_source.reset(x=X[i, q_mask, 0], y=X[i, q_mask, 1], z=X[i, q_mask, 2] + 1, u=Q[i, q_mask, 0], v=Q[i, q_mask, 1], w=Q[i, q_mask, 2], scalars=colors[q_mask], scale_factor=scalefactor_q, mode='arrow')
                else:
                    quiv_plot.mlab_source.reset(x=X[i, q_mask, 0], y=X[i, q_mask, 1], z=X[i, q_mask, 2] + 1, u=Q[i, q_mask, 0], v=Q[i, q_mask, 1], w=Q[i, q_mask, 2], scale_factor=scalefactor_q)

            # saves each frame:
            zeros = '0' * (padding - len(str(i)))
            filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, i, ext))
            mlab.savefig(filename=filename)


    if RGB == 1:

        try:
            os.mkdir(f'./myData/animations/data_{name}_RGB={RGB}')
        except OSError:
            pass
        out_path = f'./myData/animations/data_{name}_RGB={RGB}'
        out_path = os.path.abspath(out_path)

        # initializing plotting:
        #mlab.view(0, 0)
        p3d_plot = mlab.points3d(X[0, :, 0], X[0, :, 1], X[0, :, 2], colormap="autumn", scale_factor=1.5,scale_mode='none')
        iend = nSaves
        for i in range(iend):
            # Creating new view

            p3d_plot.mlab_source.set(x=X[i, :, 0], y=X[i, :, 1], z=X[i, :, 2])
            colors = np.ones((nUnits, 3))  # LUT: RGBA, where the last number A is the opacity.
            colors[:, 0] = A_color[i, :] * 255  # The amount of red is determined by the amount of Sar1,
            colors[:, 1] = E_color[i, :] * 255  # The amount of green is determined by the amount of Sec23
            colors[:, 2] = 0  # I don't need three colors right now.
            # colors[:,3] =  255 # making it nontransparent

            sc = tvtk.UnsignedCharArray()
            sc.from_array(colors)
            p3d_plot.mlab_source.dataset.point_data.scalars = sc
            p3d_plot.mlab_source.dataset.modified()

            # saves each frame:
            zeros = '0' * (padding - len(str(i)))
            filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, i, ext))
            mlab.savefig(filename=filename)



if plotALineplots == 1:

    d = np.sqrt(np.sum((X[0, :, :]-X[0, 60, :])**2, axis=1)) #np.sqrt(np.sum((X[0, :, :])**2, axis=1)) # np.sqrt(np.sum((X[0, :, :]-X[0, 60, :])**2, axis=1))

    big_lim = 10
    small_lim = 5

    mask_outer = (d > 19.83) & (d < 20) # (d > 19.77) & (d < 20)
    mask_ring = (d > 12.75) & (d < 13) #(d > 12.87) & (d < 13)  #(d > small_lim) & (d < big_lim)
    mask_center = d<2.5 #2 #(d < 1) | ((d>2) & (d<2.5)) | ((d>3.3) & (d<3.5)) | ((d>4.5) & (d<4.6))  #d < small_lim

    calc_adap = 1

    if calc_adap == 1:
        A_c = A[:, mask_center]
        maxA_c = np.amax(A_c, axis=0)
        minA_c = np.zeros_like(maxA_c)
        for i in range(sum(mask_center)):
            i_maxA_c = int(np.where(A_c[:, i] == maxA_c[i])[0])
            minA_c[i] = np.amin(A_c[i_maxA_c:, i])
        adap_perc_c = minA_c/maxA_c

        A_r = A[:, mask_ring]
        maxA_r = np.amax(A_r, axis=0)
        minA_r = np.zeros_like(maxA_r)
        for i in range(sum(mask_ring)):
            i_maxA_r = int(np.where(A_r[:, i] == maxA_r[i])[0])
            minA_r[i] = np.amin(A_r[i_maxA_r:, i])
        adap_perc_r = minA_r / maxA_r

        print(adap_perc_c)
        print(adap_perc_r)
        print('mean of ring units', np.mean(adap_perc_r))

    plot_placement = 1

    if plot_placement == 1:
        colors = np.ones(nUnits)
        colors[mask_outer] = 0.9
        colors[mask_ring] = 0.5
        colors[mask_center] = 0.2

        p3d_plot = mlab.points3d(X[0, :, 0], X[0, :, 1], X[0, :, 2], colors, colormap="spectral", scale_factor=1.5, scale_mode='none')
        mlab.view(0, 180)

        if fixed_cbar == 1:
            lut_manager = mlab.colorbar()
            lut_manager.data_range = (0, 1)
        mlab.show()

    # not plotting the extremes:
    mask_A_center = mask_center
    mask_A_ring = mask_ring
    mask_A_outer = mask_outer
    mask_E_center = mask_center
    mask_E_ring = mask_ring
    mask_E_outer = mask_outer

    notPlotExtreme = 0

    if notPlotExtreme == 1:
        for i in range(len(mask_center)): # mask_center/ring/outer are all equally long
            if any( np.abs(A[:, i]) > 2 ):
                mask_A_center[i] = mask_A_ring[i] = mask_A_outer[i] = False

            if any(np.abs(E[:, i]) > 10):
                mask_E_center[i] = mask_E_ring[i] = mask_E_outer[i] = False



    startTime = 0
    xtime = np.arange(startTime, len(A[:, 0]) )*0.025
    #x_lines = np.arange(startTime,nUnits)
    fontsize = 18
    fontsize2 = 14
    mpl.rc('xtick', labelsize=14)
    mpl.rc('ytick', labelsize=14)

    plot_together = 0
    if plot_together == 1:


        n = 5
        color = plt.cm.Blues(np.linspace(0.5, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

        figB, axB = plt.subplots(1, 2, figsize=(10, 7))
        plt.xlabel("Time steps", fontsize=fontsize)
        axB[0].set_ylabel("Sar1 concentration", fontsize=fontsize)
        axB[0].plot(xtime, A[startTime:, mask_A_center], label='Central units')
        axB[0].plot(xtime, A[startTime:, mask_A_ring], label='Ring units')
        #axB[0].plot(xtime, A[startTime:, mask_A_outer], label='Outer units')

        plt.show()

    plot_6 = 0
    if plot_6 == 1:
        n = 5
        color_inner = plt.cm.Blues(np.linspace(0.2, 1, n))
        color_ring = plt.cm.Greens(np.linspace(0.2, 1, n))
        color_outer= plt.cm.OrRd(np.linspace(0.2, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_inner)

        figA, axA = plt.subplots(3, 1, figsize=(8,10))
        # add a big axes, hide frame
        figA.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sar1 concentration", fontsize=fontsize)
        axA[0].plot(xtime, A[startTime:, mask_A_center])
        axA[0].set_title('Central units', fontsize=fontsize2)

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_ring)
        axA[1].plot(xtime, A[startTime:, mask_A_ring])
        axA[1].set_title('Ring units', fontsize=fontsize2)

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_outer)
        axA[2].plot(xtime, A[startTime:, mask_A_outer])
        axA[2].set_title('Outer units', fontsize=fontsize2)
        #plt.xticks(fontsize=fontsize)
        #plt.yticks(fontsize=fontsize)
        figA.tight_layout(h_pad=2)
        #plt.savefig('LineplotA_data931_5lines.pdf')

        figE, axE = plt.subplots(3, 1, figsize=(8, 10))
        # add a big axes, hide frame
        figE.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sec concentration", fontsize=fontsize)
        axE[0].plot(xtime, E[startTime:, mask_center])
        axE[0].set_title('Central units', fontsize=fontsize2)
        axE[1].plot(xtime, E[startTime:, mask_ring])
        axE[1].set_title('Ring units', fontsize=fontsize2)
        axE[2].plot(xtime, E[startTime:, mask_outer])
        axE[2].set_title('Outer units', fontsize=fontsize2)
        figE.tight_layout(h_pad=2)
        #plt.savefig('LineplotE_data931_5lines.pdf')
        plt.show()

    plot6_sep = 1
    if plot6_sep == 1:
        fontsize = 20
        n = 5
        color_inner = plt.cm.Blues(np.linspace(0.4, 1, n))
        color_ring = plt.cm.Greens(np.linspace(0.4, 1, n))
        color_outer = plt.cm.OrRd(np.linspace(0.4, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_inner)

        figA, axA = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sar1 concentration", fontsize=fontsize)
        axA.plot(xtime, A[startTime:, mask_A_center])
        axA.set_title('Central units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figA.tight_layout(h_pad=2)
        #plt.savefig('LineplotA_inner_data931_5lines.pdf')

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_ring)
        figB, axB = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sar1 concentration", fontsize=fontsize)
        axB.plot(xtime, A[startTime:, mask_A_ring])
        axB.set_title('Ring units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figB.tight_layout(h_pad=2)
        #plt.savefig('LineplotA_ring_data931_5lines.pdf')

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_outer)
        figC, axC = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sar1 concentration", fontsize=fontsize)
        axC.plot(xtime, A[startTime:, mask_A_outer])
        axC.set_title('Outer units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figC.tight_layout(h_pad=2)
        #plt.savefig('LineplotA_outer_data931_5lines.pdf')

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_inner)
        figD, axD = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sec concentration", fontsize=fontsize)
        axD.plot(xtime, E[startTime:, mask_center])
        axD.set_title('Central units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figD.tight_layout(h_pad=2)
        #plt.savefig('LineplotE_inner_data931_5lines.pdf')

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_ring)
        figE, axE = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sec concentration", fontsize=fontsize)
        axE.plot(xtime, E[startTime:, mask_ring])
        axE.set_title('Ring units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figE.tight_layout(h_pad=2)
        #plt.savefig('LineplotE_ring_data931_5lines.pdf')

        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_outer)
        figF, axF = plt.subplots(1, 1, figsize=(8, 4))
        plt.xlabel("Time (s)", fontsize=fontsize)
        plt.ylabel("Sec concentration", fontsize=fontsize)
        axF.plot(xtime, E[startTime:, mask_outer])
        axF.set_title('Outer units', fontsize=fontsize2)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        figF.tight_layout(h_pad=2)
        #plt.savefig('LineplotE_outer_data931_5lines.pdf')
        plt.show()

if plot_A_E_spread_sum == 1:
    E_sum = np.sum(E, axis=1)
    E_change = E_sum[1:]-E_sum[:-1]
    E_acc = E_change[1:] - E_change[:-1]

    x = np.arange(nSaves)
    x_change = np.arange(1,nSaves)#*0.025*500
    x_acc = np.arange(2,nSaves)

    plt.figure()
    plt.plot(x, E_sum/max(E_sum), label='E')
    plt.plot(x_change, E_change/max(E_change), label='Change in E')
    plt.plot(x_acc, E_acc/max(E_acc), label='Acceleration of E')
    plt.legend()

    A_sum = np.sum(A, axis=1)
    A_change = A_sum[1:]-A_sum[:-1]
    A_acc = A_change[1:] - A_change[:-1]

    plt.figure()
    plt.plot(x, A_sum/max(A_sum), label='A')
    plt.plot(x_change, A_change/max(A_change), label='Change in A')
    plt.plot(x_acc, A_acc/max(A_acc), label='Acceleration of A')
    plt.legend()

    plt.figure()
    plt.plot(x_change, E_change/max(E_change), label='Change in E')
    plt.plot(x_change, A_change / max(A_change), label='Change in A')
    plt.legend()

    plt.figure()
    plt.plot(x_acc, E_acc / max(E_acc), label='Accelleration in E')
    plt.plot(x_acc, A_acc / max(A_acc), label='Accelleration in A')
    plt.legend()
    plt.show()

if plot_A_E_spread_count == 1:
    fontsize = 16
    x = np.arange(nSaves)
    x_change = np.arange(1, nSaves)  # *0.025*500
    x_acc = np.arange(2, nSaves)
    x_diff_smooth = np.arange(7, nSaves)

    numIso = np.zeros(len(E[:,0]))
    numLifted = np.zeros(len(E[:, 0]))
    numAniso = np.zeros(len(E[:, 0]))
    for u in range(len(E[:,0])):
        mask_iso = E[u,:] > Elim
        numIso[u] = sum(mask_iso)
        mask_lifted = X[u,:,2] < -4
        numLifted[u] = sum(mask_lifted)
        mask_aniso = A[u,:] > Alim_Sar
        numAniso[u] = sum(mask_aniso)
    numIso_diff = numIso[1:] - numIso[:-1]
    numIso_acc = numIso_diff[1:] - numIso_diff[:-1]
    numAniso_diff = numAniso[1:] - numAniso[:-1]
    numAniso_acc = numAniso_diff[1:] - numAniso_diff[:-1]
    numLifted_diff = numLifted[1:] - numLifted[:-1]
    numLifted_acc = numLifted_diff[1:] - numLifted_diff[:-1]

    #smoothing out changes:
    numIso_diff_smooth = (numIso_diff[:-6] + numIso_diff[1:-5] + numIso_diff[2:-4] + numIso_diff[3:-3] + numIso_diff[4:-2] + numIso_diff[5:-1] + numIso_diff[6:]) / 7
    numAniso_diff_smooth = (numAniso_diff[:-6] + numAniso_diff[1:-5] + numAniso_diff[2:-4] + numAniso_diff[3:-3] + numAniso_diff[4:-2] + numAniso_diff[5:-1] + numAniso_diff[6:]) / 7
    numLifted_diff_smooth = (numLifted_diff[:-6] + numLifted_diff[1:-5] + numLifted_diff[2:-4] + numLifted_diff[3:-3] + numLifted_diff[4:-2] + numLifted_diff[5:-1] + numLifted_diff[6:]) / 7

    plt.figure()
    plt.plot(x, numLifted, label='# particles farther away than 4')
    plt.plot(x, numIso, label='# isotropic particles')
    plt.plot(x, numAniso, label='# Sar1 > 0.2')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.figure()
    plt.plot(x_change, numLifted_diff, label='Change in lifted particles')
    plt.plot(x_change, numIso_diff, label='Change in isotropic particles')
    plt.plot(x_change, numAniso_diff, label='Change in particles with Sar1>0.2')
    plt.legend()

    plt.figure()
    plt.plot(x_diff_smooth, numLifted_diff_smooth, label='Smooth change in lifted particles')
    plt.plot(x_diff_smooth, numIso_diff_smooth, label='Smooth change in isotropic particles')
    plt.plot(x_diff_smooth, numAniso_diff_smooth, label='Smooth change in particles with Sar1>0.2')
    plt.legend()

    #plt.figure()
    #plt.plot(x_acc, numIso_acc, label='Acceleration of isotropic units')
    #plt.plot(x_acc, numLifted_acc, label='Acceleration of lifted units')
    #plt.legend()
    plt.show()

if find_vesicle_size == 1:
    mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(1000, 800))
    indexes = np.arange(nUnits)
    comb_i = itertools.combinations(indexes, 2)
    productSave = 3
    checkSave = 3
    indexSave = np.array([0, 0])
    belowZ = -5
    for pair in comb_i:
        product = np.dot(P[saveNr, pair[0], :2], P[saveNr, pair[1], :2])
        antiParCheck = np.abs(product + 1)
        if (antiParCheck < checkSave) & (X[saveNr, pair[0], 2]<belowZ) & (X[saveNr, pair[1], 2]<belowZ): # and (X[saveNr, pair[0], 2] < -5) and (X[saveNr, pair[1], 2] < -5):
            checkSave = antiParCheck
            productSave = product
            indexSave = pair
    print('units with the indexes', indexSave, 'have the dot product', productSave)

    vesicle_size = np.sqrt(np.sum( (X[saveNr, indexSave[0], :] - X[saveNr, indexSave[1], :]) ** 2))
    print('The distance between them is', vesicle_size)

    colors = np.zeros(nUnits)
    colors[indexSave[0]] = 1
    colors[indexSave[1]] = 1
    colormap = "autumn"

    current_view = mlab.view()
    print(current_view)
    mlab.view(0, 180, distance=0.1)
    # mlab.view(270, 90, distance=0.1)
    p3d_plot = mlab.points3d(X[saveNr, :, 0], X[saveNr, :, 1], X[saveNr, :, 2], colors, colormap=colormap, scale_factor=scalefactor, scale_mode='none', resolution=ballres, mode=ballmode)
    mlab.axes(x_axis_visibility=True)
    mlab.show()

find_vesicle_size2 = 0

if find_vesicle_size2 == 1:
    list_1 = ["a", "b", "c", "d"]
    list_2 = [1, 4, 9]

    unique_combinations = []

    # Getting all permutations of list_1 with length of list_2
    permut = itertools.permutations(list_1, len(list_2))

    print('all permutations:', permut)

    # zip() is called to pair each permutation and shorter list element into combination
    for comb in permut:
        zipped = zip(comb, list_2)
        unique_combinations.append(list(zipped))

        # printing unique_combination list
    print('unique combinations:', unique_combinations)

