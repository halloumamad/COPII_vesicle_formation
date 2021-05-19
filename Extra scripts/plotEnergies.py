import matplotlib.pyplot as plt
import numpy as np
import sys




bended = np.array([2724.5, 2723.1, 2724.3, 2723.0, 2724, 2718.3, 2724.9, 2727.4, 2721.8, 2723.2, 2722.0, 2725.2])*-1
bended_std = np.array([1.4, 1.8, 1.4, 1.5, 2, 1.5, 1.6, 1.2, 1.7, 1.9, 1.4, 1.3])
bended_mean = np.average(bended, weights=bended_std)
bended_mean_array = np.ones(len(bended))*bended_mean
bend_mean_std = np.sqrt(1/sum(1/bended_std**2)) # barlow lign. 4.7
#print(bended_mean, np.mean(bended))


bended_x = np.arange(1, len(bended)+1, 1)
print(bended_x)

radial = np.array([2723, 2722.6])*-1
radial_std = np.array([1, 1.5])
radial_x = np.arange(len(bended)+1, len(bended)+len(radial)+1, 1)
radial_mean = np.average(radial, weights=radial_std)
radial_mean_array = np.ones(len(radial))*radial_mean
radial_mean_std = np.sqrt(1/sum(1/radial_std**2))
#print(radial_x)

circular = np.array([2723.5, 2724.7, 2721.6, 2727.8, 2726.5, 2727.3])*-1
circular_std = np.array([1.4, 1.4, 1.3, 1.3, 1.6, 1.3])
circular_x = np.arange(len(bended)+len(radial)+1, len(bended)+len(radial)+len(circular)+1, 1)
circular_mean = np.average(circular, weights=circular_std)
circular_mean_array = np.ones(len(circular))*circular_mean
circ_mean_std = np.sqrt(1/sum(1/circular_std**2))
print(circular_x)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.errorbar(bended_x, bended, yerr=bended_std, fmt='ob', capsize=3, capthick=1, label='bending', ecolor='b')
#plt.plot(bended_x, bended_mean_array, 'b')
ax1.errorbar(radial_x, radial, yerr=radial_std, fmt='or', capsize=3, capthick=1, label='radial spirals', ecolor='r')
#plt.plot(radial_x, radial_mean_array, 'r')
ax1.errorbar(circular_x, circular, yerr=circular_std, fmt='og', capsize=3, capthick=1, label='circular spirals', ecolor='g')
#plt.plot(tangential_x, tangential_mean_array, 'g')
ax1.legend()
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_ylabel('Potential, V', fontsize=16)
#fig1.savefig('./energiesPlot.pdf', bbox_inches ="tight")
#plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.errorbar(0, bended_mean, yerr=bend_mean_std, fmt='ob', capsize=4, capthick=1, label='bending', ecolor='b')
ax2.errorbar(0, radial_mean, yerr=radial_mean_std, fmt='or', capsize=4, capthick=1, label='radial spirals', ecolor='r')
ax2.errorbar(0, circular_mean, yerr=circ_mean_std, fmt='og', capsize=4, capthick=1, label='circular spirals', ecolor='g')
ax2.legend()
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_ylabel('Potential, V', fontsize=16)
#fig2.savefig('energiesMean.pdf', bbox_inches ="tight")


print('bended_mean', bended_mean, 'radial_mean', radial_mean, 'circular_mean', circular_mean)

rel_dif = (circular_mean - bended_mean) / circular_mean * 100
rel_dif2 = (circular_mean - bended_mean) / bended_mean * 100
print('relative difference in procent: ', rel_dif, 'or', rel_dif2)

plt.show()