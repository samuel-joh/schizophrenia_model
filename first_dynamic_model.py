import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from neurolib.models.hopf import HopfModel
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace
from os.path import join as pjoin
import scipy.io as sio

### load structural connectivity matrix and normalize it ###
data_dir = pjoin(os.getcwd(), 'data_for_modeling')
mat_name = pjoin(data_dir, 'atlas-4S156Parcels_desc-mean_sc.mat')
sc_mat = sio.loadmat(mat_name)
sc_mat = np.array(sc_mat["connectivity"])
print(sc_mat)
sc_mat_normalized = sc_mat/np.max(sc_mat)  # sc_mat * (0.2/np.max(sc_mat))
print(np.max(sc_mat_normalized))
######

### load empirical data ###
os.chdir(pjoin(os.getcwd(), "data_for_modeling", "extracted_timeseries_schizophrenia"))

subfolders = [os.path.join(f.path, "func") for f in os.scandir(os.getcwd()) if f.is_dir() and "sub-" in f.name]

data = []  # holds all control and patient signal arrays
for folder in subfolders:
    for file in os.scandir(folder):
        with open(file, "r") as f:
            data.append(pd.read_csv(file, sep='\t'))

vision_columns = [col for col in data[0].columns if col.startswith('LH_Vis') or col.startswith('RH_Vis')]

# Show the columns
print("Columns starting with 'Vis':")
print(vision_columns)
######

### get indices of vision columns and restrict sc_mat to vision areas ###
vision_columns_indices = []
for col_name in vision_columns:
    vision_columns_indices.append(data[0].columns.to_list().index(col_name))
vision_sc_mat = sc_mat_normalized[np.ix_(vision_columns_indices, vision_columns_indices)]
######

## divide data into control and patient groups, with only visual areas ###
data_control = data[:22]
data_schizo = data[-23:]

data_control_vis = [array[vision_columns] for array in data_control]
data_schizo_vis = [array[vision_columns] for array in data_schizo]

n_slices, n_regions = np.shape(data_control_vis[0])  # number of slices and regions
single_control = np.mean(data_control_vis, axis=0)  # data_control_vis[0]
single_schizo = np.mean(data_schizo_vis, axis=0)  # data_schizo_vis[0]
######

# TODO: print sc_mat and compare to data sc, normalize sc_mat to 0.2 and match omega to paper ; Automate find least error between both
### Hopf model 2x2 ###
fig_model, ax_model = plt.subplots(1, 2)
label_model = "Vision (model)"
# n_regions must be < 20, otherwise model explodes
cmat = vision_sc_mat  # normalized sc_mat with only the visual areas
dmat = np.full((n_regions, n_regions), 0)  # no delays as starting point

hopfModel = HopfModel(Cmat=cmat, Dmat=dmat)

# params for modelling healthy people
hopfModel.params['duration'] = len(single_control) * 2 * 1000  # 304 - 8 = 296 seconds
hopfModel.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
hopfModel.params['sigma_ou'] = 0.4  # set the noise here [0, 2]
hopfModel.params['K_gl'] = 2.5  # global coupling [0, 5]
hopfModel.params['w'] = 1 * np.pi  # intrinsic angular frequency of the oscillation (omega) [0, 3pi]
hopfModel.params['a'] = 0  # bifurcation parameter (we will disregard it for now) [-1, 1]

hopfModel.run()
t, x = hopfModel.getOutputs()["t"], hopfModel.getOutputs()["x"]  # get time slices and real values of Hopf oscillator
for idx, region in enumerate(x):
    ax_model[0].plot(t / 1000, x[idx], color="red", label=label_model)
    ax_model[0].set_xlabel("Time [s]")
    ax_model[0].set_ylabel("Activity")
    ax_model[0].grid(True)
    label_model = "_nolegend_"
ax_model[0].set_title(f'Time Series for Vision regions \n{vision_columns[:2]}, ..., {vision_columns[-2:]}')
ax_model[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

correlation_measure = ConnectivityMeasure(kind='correlation')
functional_connectivity = correlation_measure.fit_transform([x.T])  # functional connectivity measure
tcf = ax_model[1].imshow(np.squeeze(functional_connectivity), cmap=plt.cm.Blues)
cbar = fig_model.colorbar(tcf, ax=ax_model, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity")
fig_model.suptitle("FC and BOLD from Hopf model", fontsize=16)
######

### plot empirical BOLD signal and FC ###
fig_emp, ax_emp = plt.subplots(1, 2)
label = "Vision (control)"
labeled = False
for idx, col in enumerate(vision_columns):
    ax_emp[0].plot(np.arange(0, 296, 2), single_control.T[idx], color="blue", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
    label = "_nolegend_"

ax_emp[0].set_xlabel('Time [s]')
ax_emp[0].set_ylabel('Activity')
ax_emp[0].set_title(f'Time Series for Vision regions \n{vision_columns[:2]}, ..., {vision_columns[-2:]}')
ax_emp[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

correlation_measure = ConnectivityMeasure(kind='correlation')
functional_connectivity = correlation_measure.fit_transform([single_control])
tcf = ax_emp[1].imshow(np.squeeze(functional_connectivity), cmap=plt.cm.Blues)
fig_emp.colorbar(tcf, ax=ax_emp, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity")
fig_emp.suptitle("FC and BOLD from empirical data", fontsize=16)
plt.show()
######



# # Beispiel: Fitness-Funktion, die Korrelationsmatrix vergleicht
# correlation_measure = ConnectivityMeasure(kind='correlation')
# emp_fc = correlation_measure.fit_transform([single_control])
# emp_fc = emp_fc.reshape(emp_fc.shape[1], emp_fc.shape[2])
#
# def fitness_function(traj):
#     model = evolution.getModelFromTraj(traj)
#     model.run()
#     t, x = hopfModel.getOutputs()["t"], hopfModel.getOutputs()["x"]  # get time slices and real values of Hopf oscillator
#
#     # Simulierte FC
#     sim_fc = np.corrcoef(x)
#
#     # Flatten (nur obere Dreiecksmatrix ohne Diagonale vergleichen)
#     idx = np.triu_indices_from(emp_fc, k=1)
#     sim_fc_flat = sim_fc[idx]
#     emp_fc_flat = emp_fc[idx]
#
#     # Ähnlichkeit als Korrelationskoeffizient der FCs
#     fitness_tuple  = ()
#     fitness_tuple  += (np.corrcoef(sim_fc_flat, emp_fc_flat)[0, 1],)
#     return fitness_tuple , {}
#
#
# hopfModel.params['duration'] = 6 * 2 * 1000  # 304 - 8 = 296 seconds
# hopfModel.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
# pars = ParameterSpace(
#     ["w", "a", "K_gl"],
#     [[0.1, np.pi], [-0.3, 0.3], [0.0, 3.0]]
# )
#
# evolution = Evolution(
#     model=hopfModel,
#     parameterSpace=pars,
#     evalFunction=fitness_function,
#     POP_INIT_SIZE=1600,
#     POP_SIZE=160,
#     NGEN=100
# )
#
# evolution.run()
#
# # the current population is always accesible via
# pop = evolution.pop
# # we can also use the functions registered to deap
# # to select the best of the population:
# best_10 = evolution.toolbox.selBest(pop, k=10)
# # Remember, we performed a minimization so a fitness
# # of 0 is optimal
# print("Best individual", best_10[0], "fitness", best_10[0].fitness)
# evolution.info(plot=True)
# run model with found params
# hopfModel = HopfModel()
# hopfModel.set_params(best_10[0].params)
# hopfModel.run()





# ### Hopf model 2x2 ###
# fig_model, ax_model = plt.subplots(2, 2)
# fc_imgs = []
# label_model = "Vision (model)"
# for i in range(1):
#     # n_regions must be < 20, otherwise model explodes
#     cmat = vision_sc_mat  # normalized sc_mat with only the visual areas
#     dmat = np.zeros((n_regions, n_regions))  # no delays as starting point
#
#     model = HopfModel(Cmat=cmat, Dmat=dmat)
#
#     # params for modelling healthy people
#     model.params['duration'] = len(single_control) * 2 * 1000  # 304 - 8 = 296 seconds
#     model.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
#     model.params['sigma_ou'] = 0.02  # set the noise here
#     model.params['K_gl'] = 1  # global coupling
#     model.params['w'] = 0.2 * np.pi  # intrinsic angular frequency of the oscillation (omega)
#     model.params['a'] = 0  # bifurcation parameter (we will disregard it for now)
#
#     model.run()
#     t, x = model.getOutputs()["t"], model.getOutputs()["x"]  # get time slices and real values of Hopf oscillator
#     correlation_measure = ConnectivityMeasure(kind='correlation')
#     functional_connectivity = correlation_measure.fit_transform([x.T])  # functional connectivity measure
#     for idx, region in enumerate(x):
#         ax_model[i % 2, 0].plot(t / 1000, x[idx], color="red", label=label_model)
#         ax_model[i % 2, 0].set_xlabel("t [s]")
#         ax_model[i % 2, 0].set_ylabel("Activity")
#         ax_model[i % 2, 0].grid(True)
#
#         tcf = ax_model[i % 2, 1].imshow(np.squeeze(functional_connectivity), cmap=plt.cm.Blues)
#         fc_imgs.append(tcf)
#         ax_model[i % 2, 1].set_xticks([])
#         ax_model[i % 2, 1].set_yticks([])
#         label_model = "_nolegend_"
#     ax_model[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
#     # plt.tight_layout()
# fig_model.suptitle("FC and BOLD from model", fontsize=16)
# cbar = fig_model.colorbar(fc_imgs[0], ax=ax_model, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity")
# #######

### plot columns of all participants ###
# plt.figure(figsize=(15, 10))
# label = 'Vision (control)'
# labeled = False
# for col in vision_columns:
#     for tms in data_control:
#         plt.plot(tms.index, tms[col], color="blue", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
#         label = "_nolegend_"
#     label = "Vision (schizo)"
#     for tms in data_schizo:
#         if labeled:
#             label = "_nolegend_"
#         plt.plot(tms.index, tms[col], color="red", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
#         labeled = True
#
# print(f"Data from {len(data_control)} healthy people and {len(data_schizo)} schizophrenia patients")
# plt.xlabel('Time (timepoints)')
# plt.ylabel('BOLD signal')
# plt.title(f'Time Series for Vision regions \n{vision_columns[:2]}, ..., {vision_columns[-2:]}')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
# plt.tight_layout()
# os.chdir(os.path.join(os.getcwd(), "..", ".."))
# plt.savefig("Striatum.png")
# plt.show()


# t, x = model.getOutputs()["t"], model.getOutputs()["x"]  # get time slices and real values of Hopf oscillator
# plt.figure(figsize=(10, 10))
# for idx, region in enumerate(x):
#     plt.plot(t, x[idx], alpha=0.5, label=idx)
# plt.xlabel("t [ms]")
# plt.ylabel("Activity")
# plt.grid(True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
# plt.tight_layout()
# plt.show()

# This is one implementaton of the Hopf model based on Neurolib
# functional connectivity measure
# correlation_measure = ConnectivityMeasure(kind='correlation')
# figure = plt.figure(figsize=(4, 4))
# functional_connectivity = correlation_measure.fit_transform([model.x.T])
# tcf = plt.imshow(np.squeeze(functional_connectivity), cmap=plt.cm.Blues)
# figure.colorbar(tcf)
# plt.title("FC aus Simulationsdaten")
# plt.show()
