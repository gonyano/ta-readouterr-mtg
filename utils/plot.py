import numpy as np
import pickle
import matplotlib.pyplot as plt

exp_shots_list = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

y_noisy, yerr_noisy = [], []; y_mitigated_inv, yerr_mitigated_inv = [], []; y_mitigated_m3, yerr_mitigated_m3 = [], []
for exp_shots in exp_shots_list:
    path = f'../outputs/s{exp_shots}.pickle'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    time_inv_list, time_m3_list, exp_noisy_list, exp_mitigated_inv_list, exp_mitigated_m3_list = data
    for y, yerr, exp_list in zip([y_noisy, y_mitigated_inv, y_mitigated_m3],[yerr_noisy, yerr_mitigated_inv, yerr_mitigated_m3],[exp_noisy_list, exp_mitigated_inv_list, exp_mitigated_m3_list]):
        y.append(np.mean(exp_list))
        yerr.append(np.std(exp_list))    

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
cmap = plt.get_cmap("tab20")
ax.errorbar(exp_shots_list, y_noisy, yerr=yerr_noisy, fmt='.', color=cmap(2), capsize=5, ms=10, label="Raw")
ax.errorbar(exp_shots_list, y_mitigated_m3, yerr=yerr_mitigated_m3, fmt='.', color=cmap(0), capsize=5, ms=10, label="M3")
ax.errorbar(exp_shots_list, y_mitigated_inv, yerr=yerr_mitigated_inv, fmt='.', color=cmap(1), capsize=5, ms=10, label="Ignis tensored")
ax.axhline(0.446, color='b', ls='--'); ax.text(110, 0.45, "Ideal value")
leg = plt.legend(loc='upper right', frameon=True)
leg.get_frame().set_edgecolor('k')
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
ax.grid()
ax.set_xlabel('Samples')
ax.set_ylabel('Expectation value')
ax.set_ylim(0.25,0.5)
ax.set_xscale('log')
plt.savefig('./result.png')
