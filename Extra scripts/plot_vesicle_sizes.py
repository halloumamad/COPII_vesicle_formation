import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./External_Functions') ## har har External Functions-mappen ligget en mappe ude.
from ExternalFunctions import Chi2Regression, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax
from iminuit import Minuit
from scipy import stats
fontsize = 32

plot_own = 0
plot_const = 0
plot_linear = 0
plot_bothInOne = 1

plot_raw_data = 0

# Varying spread of cargo:
cargo_sigma = np.array([[6, 6, 6, 6, 6], [7, 7, 7, 7, 7], [8, 8, 8, 8, 8], [9, 9, 9, 9, 9], [10, 10, 10, 10, 10]])
size_sigma = np.array([[5.53, 4.87, 4.59, 4.77, 4.45], [6.46, 6.38, 5.90, 6.51, 5.54], [7.45, 7.12, 6.80, 7.28, 6.27], [8.05, 8.25, 7.63, 7.64, 8.00], [8.46, 8.77, 7.91, 7.72, 9.16]])
mean_size_s = np.mean(size_sigma, axis=1)
#print('smallest vesicle varying spread of cargo:', min(mean_size_s))
#print('largest vesicle varying spread of cargo:', max(mean_size_s))
std_size_s = np.std(size_sigma, axis=1)
cargo_sigma_x = np.array([6, 7, 8, 9, 10])

print(len(cargo_sigma_x), len(mean_size_s), len(std_size_s))

if plot_own == 1:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.errorbar(cargo_sigma_x, mean_size_s, yerr=std_size_s, fmt='ob', capsize=3, capthick=1, ecolor='b') #label='bending',
    ax1.plot(cargo_sigma, size_sigma, '.k')
    ax1.set_xlabel('Radius of spread of cargo', fontsize=fontsize)
    ax1.set_ylabel('Diameter of vesicle', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    #plt.show()


# data from sigma = 10:

cargo_height = [[3, 3, 3, 3, 3], [3.5, 3.5, 3.5, 3.5, 3.5], [4, 4, 4, 4, 4], [4.5, 4.5, 4.5, 4.5, 4.5]] #
size_height = [[5.48, 6.32, 5.20, 6.34, 7.63], [7.71, 6.52, 8.25, 7.70, 7.96], [8.46, 8.77, 7.91, 7.72, 9.16], [8.73, 9.53, 9.06,  9.14, 8.92]]


print(len(cargo_height))
mean_size_h = []
std_size_h = []
for i in range(len(cargo_height)):
    print(size_height[i])
    mean_size_h.append(np.mean(size_height[i]))
    std_size_h.append(np.std(size_height[i]))
#print('mean_size_h:',mean_size_h)
#mean_size_h = np.mean(size_height, axis=1)
#std_size_h = np.std(size_height, axis=1)
cargo_height_x = np.array([3, 3.5, 4, 4.5])

if plot_own == 1:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.errorbar(cargo_height_x, mean_size_h, yerr=std_size_h, fmt='ob', capsize=3, capthick=1, ecolor='b') #label='bending',
    ax2.plot(cargo_height[0], size_height[0], '.k')
    ax2.plot(cargo_height[1], size_height[1], '.k')
    ax2.plot(cargo_height[2], size_height[2], '.k')
    ax2.plot(cargo_height[3], size_height[3], '.k')
    ax2.set_xlabel('Concentration of cargo', fontsize=fontsize)
    ax2.set_ylabel('Diameter of vesicle', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    #plt.show()

####
# General fitting (Not tried out yet):
plt.rc('xtick',labelsize=fontsize)
plt.rc('ytick',labelsize=fontsize)


# Configuartion:

def f(x, b): # Definer en funktion
    return np.ones_like(x)*b

def calc_and_plot(x, y, sig_y, x_label, y_label, startGuess_b, figureSaveName):
    labelData = 'Data'
    chi2_f = Chi2Regression(f, x, y, sig_y)
    minuit_f = Minuit(chi2_f, pedantic=False, b=startGuess_b) # start guesses for parameters
    minuit_f.migrad()                  # Performs the actual fit
    Ndof = len(y) - len(minuit_f.args)
    Prob = stats.chi2.sf(minuit_f.fval, Ndof)

    print(f'p(Chi2={minuit_f.fval:.1f},Ndof={Ndof:d}) = {Prob:.3}')

    # y values for plotting start guess:
    #ySG = f(x, b_SG)

    # y values for plotting fit:
    yModel = f(x, *minuit_f.args) # Er funktionsnavnet ændret?

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plotting data:
    ax.errorbar(x, y, yerr=sig_y, fmt='.k',  ecolor='k', elinewidth=1, capsize=4, capthick=2) # label=labelData,
    # Plotting start guess:
    #ax.plot(x, ySG, '-', label='Start guess')
    # Plotting fit:
    ax.plot(x, yModel, '-', label=f'Fit of constant line,  p(Chi2={minuit_f.fval:.1f},Ndof={Ndof:d}) = {Prob:.3}')

    #ax.set(ylim=(0.87, 1.24))
    ax.set_xlabel(x_label,fontsize=fontsize)
    ax.set_ylabel(y_label,fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='lower right')
    for i in range(len(x)):
        ax.plot(x[i], y[i], '.k')
    fig.savefig(figureSaveName, bbox_inches='tight')
    #plt.show()
    return fig, ax

print('mean_size_s', mean_size_s)
print('std_size_s', std_size_s)
print('mean_size_h', mean_size_h)
print('std_size_h', std_size_h)


if plot_const == 1:
    fig, ax = calc_and_plot(cargo_sigma_x, mean_size_s, std_size_s, 'Radius of spread of cargo', 'Diameter of vesicle', 7, "cargoSpread_const.pdf")
    if plot_raw_data == 1:
        ax.plot(cargo_sigma, size_sigma, '.r')
    plt.show()

    fig, ax = calc_and_plot(cargo_height_x, mean_size_h, std_size_h, 'Concentration of cargo', 'Diameter of vesicle', 8, 'cargoConcentration_const_sigma10.pdf')
    if plot_raw_data == 1:
        ax.plot(cargo_height[0], size_height[0], '.r')
        ax.plot(cargo_height[1], size_height[1], '.r')
        ax.plot(cargo_height[2], size_height[2], '.r')
        ax.plot(cargo_height[3], size_height[3], '.r')
    plt.show()

def f(x, a, b): # Definer en funktion
    return a*x+b

def calc_and_plot_2(x, y, sig_y, x_label, y_label, startGuess_a, startGuess_b, figureSaveName):
    labelData = 'Data'
    chi2_f = Chi2Regression(f, x, y, sig_y)
    minuit_f = Minuit(chi2_f, pedantic=False, a=startGuess_a, b=startGuess_b) # start guesses for parameters
    minuit_f.migrad();                  # Performs the actual fit
    Ndof = len(y) - len(minuit_f.args)
    Prob = stats.chi2.sf(minuit_f.fval, Ndof)

    print(f'p(Chi2={minuit_f.fval:.1f},Ndof={Ndof:d}) = {Prob:.3}')

    # y values for plotting start guess:
    #ySG = f(x, b_SG)

    # y values for plotting fit:
    yModel = f(x, *minuit_f.args) # Er funktionsnavnet ændret?

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plotting data:
    ax.errorbar(x, y, yerr=sig_y, fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1) # label=labelData,
    # Plotting start guess:
    #ax.plot(x, ySG, '-', label='Start guess')
    # Plotting fit:
    ax.plot(x, yModel, '-', label=f'Fit of constant line,  p(Chi2={minuit_f.fval:.1f},Ndof={Ndof:d}) = {Prob:.3}')

    #ax.set(ylim=(0.87, 1.24))
    ax.set_xlabel(x_label,fontsize=fontsize)
    ax.set_ylabel(y_label,fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='lower right') # loc='lower left'
    fig.savefig(figureSaveName, bbox_inches='tight')
    plt.show()

if plot_linear == 1:
    calc_and_plot_2(cargo_sigma_x, mean_size_s, std_size_s, 'Radius of spread of cargo', 'Diameter of vesicle', 2, 5, "cargoSpread_linear.pdf")
    calc_and_plot_2(cargo_height_x, mean_size_h, std_size_h, 'Concentration of cargo', 'Diameter of vesicle', 1, 7, 'cargoConcentration_sigma10_linear.pdf')


def f_const(x, b): # Definer en funktion
    return np.ones_like(x)*b

def f_linear(x, a, b): # Definer en funktion
    return a*x+b

def calc_and_plot_bothDatasets(x, y, sig_y, x_label, y_label, startGuess_b_const, startGuess_a_lin, startGuess_b_lin, figureSaveName):
    # calculate constant model:
    chi2_f_const = Chi2Regression(f_const, x, y, sig_y)
    minuit_f_const = Minuit(chi2_f_const, pedantic=False, b=startGuess_b_const)  # start guesses for parameters
    minuit_f_const.migrad();  # Performs the actual fit
    Ndof_const = len(y) - len(minuit_f_const.args)
    Prob_const = stats.chi2.sf(minuit_f_const.fval, Ndof_const)

    # y values for plotting fit:
    yModel_const = f_const(x, *minuit_f_const.args)

    # calculate linear model:
    chi2_f_lin = Chi2Regression(f_linear, x, y, sig_y)
    minuit_f_lin = Minuit(chi2_f_lin, pedantic=False, a=startGuess_a_lin, b=startGuess_b_lin)  # start guesses for parameters
    minuit_f_lin.migrad()  # Performs the actual fit
    Ndof_lin = len(y) - len(minuit_f_lin.args)
    Prob_lin = stats.chi2.sf(minuit_f_lin.fval, Ndof_lin)

    # y values for plotting fit:
    yModel_lin = f_linear(x, *minuit_f_lin.args)

    # Plot both models:
    fig, ax = plt.subplots(figsize=(13, 9))
    # Plotting data:
    ax.errorbar(x, y, yerr=sig_y, fmt='ok',  ecolor='k', elinewidth=2, capsize=5, capthick=2)  # label=labelData,
    # Plotting start guess:
    # ax.plot(x, ySG, '-', label='Start guess')
    # Plotting fit:
    ax.plot(x, yModel_const, '-', linewidth=3, label=f'Fit of constant line,  p(Chi2={minuit_f_const.fval:.1f},Ndof={Ndof_const:d}) = {Prob_const:.2}')
    ax.plot(x, yModel_lin, '-', linewidth=3, label=f'Fit of linear line,  p(Chi2={minuit_f_lin.fval:.1f},Ndof={Ndof_lin:d}) = {Prob_lin:.2}')

    # ax.set(ylim=(0.87, 1.24))
    #ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.legend(fontsize=23, loc='upper left')  # loc='lower left'
    ax.set_ylim([4, 10])
    plt.xlabel(x_label, labelpad=20, fontsize=fontsize)
    fig.savefig(figureSaveName, bbox_inches='tight')
    plt.show()

if plot_bothInOne == 1:
    calc_and_plot_bothDatasets(cargo_sigma_x, mean_size_s, std_size_s, 'Radius of spread of cargo', 'Diameter of vesicle', 7, 2, 5, "cargoSpread_bothFits.pdf")
    #calc_and_plot_bothDatasets(cargo_height_x, mean_size_h, std_size_h, 'Concentration of cargo', 'Diameter of vesicle', 8, 1, 7, 'cargoConcentration_sigma10_bothFits.pdf')