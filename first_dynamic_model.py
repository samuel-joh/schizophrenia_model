import pandas as pd
import matplotlib.pyplot as plt
import os
from neurolib.models.hopf import HopfModel
import numpy as np
from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace
from nilearn.connectome import ConnectivityMeasure
from os.path import join as pjoin
import scipy.io as sio
import neurolib.utils.functions as func
import seaborn


def runSingleHopf(sc_matrix, bolds_emp, vision_columns):
    ### Hopf model 2x2 ###
    fig_model, ax_model = plt.subplots(1, 2)
    label_model = "Whole brain model"

    sm = sc_matrix
    dm = np.full((n_regions, n_regions), 0)  # no delays as starting point

    hopf = HopfModel(Cmat=sm, Dmat=dm)

    # params for modelling healthy people
    hopf.params['duration'] = len(bolds_emp) * 2 * 1000  # 304 - 8 = 296 seconds
    hopf.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
    hopf.params['sigma_ou'] = 0.14  # set the noise here [0, 2]
    hopf.params['K_gl'] = 0.15  # global coupling [0, 5]
    hopf.params['w'] = 2.35  # intrinsic angular frequency of the oscillation (omega) [0, 3pi]
    hopf.params['a'] = 0.20  # bifurcation parameter (we will disregard it for now) [-1, 1]

    hopf.run()

    t, x = hopf.getOutputs()["t"], hopf.getOutputs()["x"]  # get time slices and real values of Hopf oscillator
    for index, region in enumerate(x):
        ax_model[0].plot(t / 1000, x[index], color="red", label=label_model)
        ax_model[0].set_xlabel("Time [s]")
        ax_model[0].set_ylabel("Activity")
        ax_model[0].grid(True)
        label_model = "_nolegend_"
    ax_model[0].set_title(f'Time Series for Vision regions \n{vision_columns[:2]}, ..., {vision_columns[-2:]}')
    ax_model[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

    seaborn.heatmap(func.fc(x.T), ax=ax_model[1], label="Connectivity")  # plot functional connectivity
    fig_model.suptitle("FC and BOLD from Hopf model", fontsize=16)
    plt.show()
    ######


def plotVisionEmpirical(bolds_emp, vision_columns):
    ### plot empirical BOLD signal and FC ###
    fig_emp, ax_emp = plt.subplots(1, 2)
    label = "Vision (control)"
    for idx, col in enumerate(vision_columns):
        ax_emp[0].plot(np.arange(0, 296, 2), bolds_emp.T[idx], color="blue", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
        label = "_nolegend_"

    ax_emp[0].set_xlabel('Time [s]')
    ax_emp[0].set_ylabel('Activity')
    ax_emp[0].set_title(f'Time Series for Vision regions \n{vision_columns[:2]}, ..., {vision_columns[-2:]}')
    ax_emp[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

    functional_connectivity = func.fc(bolds_emp)  # functional connectivity measure
    seaborn.heatmap(functional_connectivity, ax=ax_emp[1], label="Connectivity")
    fig_emp.suptitle("FC and BOLD from empirical data", fontsize=16)
    plt.show()
    ######

# ref: https://neurolib-dev.github.io/examples/example-2-evolutionary-optimization-minimal/
def fitness_function(traj):
    model = evolution.getModelFromTraj(traj)
    model.run()

    x = model.getOutputs()["x"]  # get values from Hopf oscillator
    sim_fc = func.fc(x)  # compute model FC
    print(f"AAAAAAAAAAAAA: {sim_fc.shape}\n\n\n\n\n\n\n\n")

    # get pearson correlation of the lower triagonals of empirical and model FC
    fitness_tuple = ()
    fitness_tuple += (func.matrix_correlation(emp_fc, sim_fc))
    return fitness_tuple, {}


if __name__ == "__main__":
    ### load structural connectivity matrix and normalize it ###
    data_dir = pjoin(os.getcwd(), 'data_for_modeling')
    mat_name = pjoin(data_dir, 'atlas-4S156Parcels_desc-mean_sc.mat')
    sc_mat = sio.loadmat(mat_name)
    sc_mat = np.array(sc_mat["connectivity"])
    sc_mat_normalized = sc_mat / np.max(sc_mat)  # sc_mat * (0.2/np.max(sc_mat))
    ######

    ### load empirical data ###
    subfolders = [os.path.join(f.path, "func") for f in os.scandir(pjoin(os.getcwd(), "data_for_modeling", "extracted_timeseries"))
                  if f.is_dir() and "sub-" in f.name]

    data = []  # holds all control and patient signal arrays
    for folder in subfolders:
        for file in os.scandir(folder):
            with open(file, "r") as f:
                data.append(pd.read_csv(file, sep='\t'))

    data_control = data[:22]
    data_schizo = data[-23:]

    # vision_columns = [col for col in data[0].columns if col.startswith('LH_Vis') or col.startswith('RH_Vis')]
    # print("Columns starting with 'Vis':")
    # print(vision_columns)

    ## get indices of vision columns and restrict sc_mat to vision areas
    # vision_columns_indices = []
    # for col_name in vision_columns:
    #     vision_columns_indices.append(data[0].columns.to_list().index(col_name))
    # vision_sc_mat = sc_mat_normalized[np.ix_(vision_columns_indices, vision_columns_indices)]
    # data_control_vis = [array[vision_columns] for array in data_control]
    # data_schizo_vis = [array[vision_columns] for array in data_schizo]
    ######

    n_slices, n_regions = np.shape(data_control[0])  # number of slices and regions
    control_mean = np.mean(data_control, axis=0)  # data_control_vis[0]
    schizo_mean = np.mean(data_schizo, axis=0)  # data_schizo_vis[0]

    # compute empirical FC to use for evolution algorithm model fitting
    correlation_measure = ConnectivityMeasure(kind='correlation')
    emp_fc = func.fc(control_mean.T)

    # init HopfModel with empirical SC matrix, duration and sampling rate
    cmat = sc_mat_normalized  # normalized sc_mat with only the visual areas
    dmat = np.full((n_regions, n_regions), 0)  # no delays as starting point
    hopfModel = HopfModel(Cmat=cmat, Dmat=dmat)
    hopfModel.params['duration'] = 6 * 2 * 1000  # 304 - 8 = 296 seconds
    hopfModel.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds

    # ref: https://neurolib-dev.github.io/examples/example-2-evolutionary-optimization-minimal/
    # noinspection PyTypeChecker
    pars = ParameterSpace(
        ["sigma_ou", "w", "a", "K_gl"],
        [[0.0, 0.5], [0.1, np.pi], [-0.3, 0.3], [0.0, 3.0]]
    )

    evolution = Evolution(
        model=hopfModel,
        parameterSpace=pars,
        evalFunction=fitness_function,
        POP_INIT_SIZE=2000,
        POP_SIZE=200,
        NGEN=125
    )

    evolution.run()

    EVOLUTION_DILL = "saved_evolution.dill"
    evolution.saveEvolution(EVOLUTION_DILL)
    # the current population is always accesible via
    pop = evolution.pop
    # we can also use the functions registered to deap
    # to select the best of the population:
    best_10 = evolution.toolbox.selBest(pop, k=10)
    # Remember, we performed a minimization so a fitness
    # of 0 is optimal
    print("Best individual", best_10[0], "fitness", best_10[0].fitness)
    evolution.info(plot=True)

    # run model with found params
    # hopfModel = HopfModel()
    # hopfModel.set_params(best_10[0].params)
    # hopfModel.run()
