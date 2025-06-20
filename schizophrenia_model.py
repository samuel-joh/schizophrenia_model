import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from neurolib.models.hopf import HopfModel
from neurolib.models.pheno_hopf import PhenoHopfModel, timeIntegration
import numpy as np
from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace
from nilearn.connectome import ConnectivityMeasure
from os.path import join as pjoin
import scipy.io as sio
import neurolib.utils.functions as func
import seaborn
from datetime import datetime
from multiprocessing import Pool, cpu_count
import copy


def sim_single_hopf(args):
    biFurc_a, repeat, fc_empirical, hopf_model, isPheno = args
    model = copy.deepcopy(hopf_model)
    model.params['a'] = biFurc_a
    model.params["seed"] = np.random.randint(0, 2 ** 31)
    if isPheno:
        print(f"Running a={biFurc_a[0]}, repeat {repeat}")
    else:
        print(f"Running a={biFurc_a}, repeat {repeat}")
    model.run()
    x = model.getOutputs()["x"]
    fc_model = func.fc(x)
    correlation = func.matrix_correlation(fc_model, fc_empirical)
    if isPheno:
        return biFurc_a[0], correlation
    return biFurc_a, correlation


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
    ######

    ### setup params and values for model fitting ###
    n_slices, n_regions = np.shape(data_control[0])  # number of slices and regions
    control_mean = np.mean(data_control, axis=0)  # data_control_vis[0]
    schizo_mean = np.mean(data_schizo, axis=0)  # data_schizo_vis[0]

    # compute empirical FC to use for evolution algorithm model fitting
    emp_fc = func.fc(control_mean.T)

    # init HopfModel with empirical SC matrix, duration and sampling rate
    cmat = sc_mat_normalized  # normalized sc_mat with only the visual areas
    dmat = np.full((n_regions, n_regions), 0)  # no delays as starting point
    hopfModel = HopfModel(Cmat=cmat, Dmat=dmat)
    hopfModel.params['duration'] = 50 * 2 * 1000  # 304 - 8 = 296 seconds
    hopfModel.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
    hopfModel.params['sigma_ou'] = 0.02  # noise
    ######

    ### fit model using evolutionary algorithm ###
    global evolution
    # fit_evolutionary(hopfModel)
    ######

    ### fit model to control participants using multiprocessing and simple parameter sweep over bifurcation parameter a ###
    healthy_correlations = []

    # n_runs = 2
    # a_vals = np.linspace(0.3, 1, 40)
    # print(a_vals)
    # starting_person = 0
    # for idx, person in enumerate(data_control[starting_person:]):
    #     print(f"Person {starting_person + idx}")
    #     person_correlations = []
    #     fc_emp = func.fc(np.array(person.T))
    #     tasks = []
    #     for a in a_vals:
    #         for i in range(n_runs):
    #             hopfModel.params['seed'] = np.random.randint(0, 2 ** 31)
    #             tasks.append((a, i, fc_emp, hopfModel, False))
    #     with Pool(processes=cpu_count() - 2) as pool:
    #         results = pool.map(sim_single_hopf, tasks)
    #         # print(f"{a}: {func.matrix_correlation(fc_model, fc_emp)}")
    #         person_correlations += [results]
    #     healthy_correlations += [person_correlations]
    #     # print(correlations)
    #     np.save(f"healthy_correlations/person{starting_person + idx}_correlations.npy", np.array(person_correlations), allow_pickle=True)
    ######

    ### fit model to schizophrenia patients using multiprocessing and simple parameter sweep over bifurcation parameter a ###
    vision_columns = [col for col in data[0].columns if col.startswith('LH_Vis') or col.startswith('RH_Vis')]

    # get indices of vision columns and create mask for vision areas
    vision_columns_indices = []
    for col_name in vision_columns:
        vision_columns_indices.append(data[0].columns.to_list().index(col_name))
    vision_columns_mask = np.array([i in vision_columns_indices for i in range(n_regions)])

    # mean a over all optimal a's from healthy people
    healthy_a = 0.17549191005073358

    n_runs = 2
    a_vals = np.linspace(-0.1, 0.75, 50)
    # print(a_vals)
    sz_correlations = []
    starting_person = 0

    # for idx, person in enumerate(data_schizo[starting_person:]):
    #     print(f"Person {starting_person + idx}")
    #     person_correlations = []
    #     fc_emp = func.fc(np.array(person.T))
    #     tasks = []
    #     for a in a_vals:
    #         # create array with all healthy a value, then change only the vision regions a's
    #         aVis = np.full(n_regions, healthy_a)
    #         aVis[vision_columns_indices] = a
    #         for i in range(n_runs):
    #             hopfModel = PhenoHopfModel(Cmat=cmat, Dmat=dmat)
    #             hopfModel.params['duration'] = 30 * 2 * 1000  # 304 - 8 = 296 seconds
    #             hopfModel.params['sampling_dt'] = 2 * 1000  # output signal every 2 seconds
    #             hopfModel.params['sigma_ou'] = 0.02  # noise
    #             hopfModel.params["seed"] = np.random.randint(0, 2 ** 31)
    #             tasks.append((aVis, i, fc_emp, hopfModel, True))
    #     with Pool(processes=cpu_count() - 2) as pool:
    #         results = pool.map(sim_single_hopf, tasks)
    #         # print(f"{a}: {func.matrix_correlation(fc_model, fc_emp)}")
    #         person_correlations += [results]
    #     correlations += [person_correlations]
    #     print(correlations)
    #     np.save(f"schizophrenia_correlations/person{starting_person + idx}_correlations.npy", np.array(person_correlations), allow_pickle=True)
    ######

    ### analyse found params and correlation ###
    sz_max_corr = []
    healthy_max_corr = []
    for i in range(22):
        sz_correlations += [np.load(f"schizophrenia_correlations/person{i}_correlations.npy", allow_pickle=True)[0]]
        healthy_correlations += [np.hstack((np.load(f"healthy_correlations/person{i}_correlations2.npy", allow_pickle=True),
                                            np.load(f"healthy_correlations/person{i}_correlations.npy", allow_pickle=True)))[0]]
    sz_correlations += [np.load(f"schizophrenia_correlations/person22_correlations.npy", allow_pickle=True)[0]]

    for person_correlations in sz_correlations:
        sz_max_corr += [max(person_correlations, key=lambda x: x[1])]
    for person_correlations in healthy_correlations:
        healthy_max_corr += [max(person_correlations, key=lambda x: x[1])]

    sz_optimal_as = []
    sz_optimal_corrs = []
    healthy_optimal_as = []
    healthy_optimal_corrs = []
    for a, corr in sz_max_corr:
        sz_optimal_as.append(a)
        sz_optimal_corrs.append(corr)
        plt.scatter(sz_optimal_as, sz_optimal_corrs)
        plt.title(f"Optimal a for SZ FC correlation model vs. emp")
        plt.xlabel("a")
        plt.ylabel("correlation")
    plt.grid()
    plt.show()

    for a, corr in healthy_max_corr:
        healthy_optimal_as.append(a)
        healthy_optimal_corrs.append(corr)
        plt.scatter(healthy_optimal_as, healthy_optimal_corrs)
        plt.title(f"Optimal a for healthy FC correlation model vs. emp")
        plt.xlabel("a")
        plt.ylabel("correlation")
    plt.grid()
    plt.show()

    mean_sz = np.mean(sz_optimal_as)
    median_sz = np.median(sz_optimal_as)
    mean_healthy = np.mean(healthy_optimal_as)
    median_healthy = np.median(healthy_optimal_as)
    plt.scatter(sz_optimal_as, sz_optimal_corrs)
    plt.scatter(healthy_optimal_as, healthy_optimal_corrs)
    plt.axvline(mean_sz, color="blue", linestyle='--', label=f'SZ mean a ({mean_sz:.4f})')
    plt.axvline(median_sz, color="blue", linestyle='-', label=f'SZ median a ({median_sz:.4f})')
    plt.axvline(mean_healthy, color="orange", linestyle='--', label=f'Healthy mean a({mean_healthy:.4f})')
    plt.axvline(median_healthy, color="orange", linestyle='-', label=f'Healthy median a({median_healthy:.4f})')
    plt.title(f"Optimal a for healthy and SZ FC correlation model vs. emp")
    plt.xlabel("a")
    plt.ylabel("correlation")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid()
    plt.show()

    print(f"Optimal SZ a's\n"
          f"range (min - max): {np.min(sz_optimal_as)} - {np.max(sz_optimal_as)}\n"
          f"mean: {np.mean(sz_optimal_as)}\n"  # = 0.45331890331890334
          f"median: {np.median(sz_optimal_as)}\n"
          f"standard deviation: {np.std(sz_optimal_as)}\n"
          f"variance: {np.var(sz_optimal_as)}\n")

    print(f"Optimal healthy a's\n"
          f"range (min - max): {np.min(healthy_optimal_as)} - {np.max(healthy_optimal_as)}\n"
          f"mean: {np.mean(healthy_optimal_as)}\n"  # = 0.45331890331890334
          f"median: {np.median(healthy_optimal_as)}\n"
          f"standard deviation: {np.std(healthy_optimal_as)}\n"
          f"variance: {np.var(healthy_optimal_as)}\n")

    statistic, pvalue = stats.ttest_ind(sz_optimal_as, healthy_optimal_as)
    print(pvalue)
    ######



## everything below was used in older versions or can just be used for visualization ##


### run Hopf model 2 times and plot resulting FC and activity ###
def runSingleHopf(sc_matrix, columns):
    fig_model, ax_model = plt.subplots(1, 2)
    label_model = "Whole brain model"

    sm = sc_matrix
    dm = np.full((n_regions, n_regions), 0)  # no delays as starting point

    hopf = HopfModel(Cmat=sm, Dmat=dm)

    # params for modelling healthy people
    hopf.params['duration'] = 148 * 2 * 1000  # 304 - 8 = 296 seconds
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
    ax_model[0].set_title(f'Time Series for Vision regions \n{columns[:2]}, ..., {columns[-2:]}')
    ax_model[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

    seaborn.heatmap(func.fc(x.T), ax=ax_model[1], label="Connectivity")  # plot functional connectivity
    fig_model.suptitle("FC and BOLD from Hopf model", fontsize=16)
    plt.show()
######


### plot empirical BOLD signal and FC ###
def plotVisionEmpirical(bolds_emp, columns):
    fig_emp, ax_emp = plt.subplots(1, 2)
    label = "Vision (control)"
    for idx, col in enumerate(columns):
        ax_emp[0].plot(np.arange(0, 296, 2), bolds_emp.T[idx], color="blue", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
        label = "_nolegend_"

    ax_emp[0].set_xlabel('Time [s]')
    ax_emp[0].set_ylabel('Activity')
    ax_emp[0].set_title(f'Time Series for Vision regions \n{columns[:2]}, ..., {columns[-2:]}')
    ax_emp[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

    functional_connectivity = func.fc(bolds_emp)  # functional connectivity measure
    seaborn.heatmap(functional_connectivity, ax=ax_emp[1], label="Connectivity")
    fig_emp.suptitle("FC and BOLD from empirical data", fontsize=16)
    plt.show()
######


# ref: https://neurolib-dev.github.io/examples/example-2-evolutionary-optimization-minimal/
### function to evaluate model fitness by returning the FC correlation ###
def fitness_function(traj):
    global evolution
    model = evolution.getModelFromTraj(traj)
    model.run()

    x = model.getOutputs()["x"]  # get values from Hopf oscillator
    sim_fc = func.fc(x)  # compute model FC

    # get pearson correlation of the lower triagonals of empirical and model FC
    fitness_tuple = ()
    fitness_tuple += (func.matrix_correlation(emp_fc, sim_fc),)
    return fitness_tuple, {}
######


# ref: https://neurolib-dev.github.io/examples/example-2-evolutionary-optimization-minimal/
### find optimal model parameters using neurolibs evolutionary algorithm ###
def fit_evolutionary(model):
    # noinspection PyTypeChecker
    pars = ParameterSpace(
        ["a", "K_gl"],
        [[-0.4, 0.4], [0.2, 2.0]]
    )
    global evolution
    evolution = Evolution(
        model=model,
        parameterSpace=pars,
        evalFunction=fitness_function,
        POP_INIT_SIZE=1000,
        POP_SIZE=100,
        NGEN=90
    )

    evolution.run()
    EVOLUTION_DILL = f"saved_evolution_{datetime.now().strftime('%d_%m_%Y-%I_%M')}.dill"
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
######
