from sys import argv
# import sys
# import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as stats
# from scipy.interpolate import interp1d
# import os
# import h5py
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# import math
# import re
import gc
from ChEFT_GP_imports import joint_plot, offset_xlabel, m_p, m_n, hbarc, E_to_p, Q_approx, \
deg_to_qcm, deg_to_qcm2, softmax_mom, GPHyperparameters, FileNaming, PosteriorBounds, \
OrderInfo, versatile_train_test_split, InputSpaceBunch, \
ObservableBunch, Interpolation, TrainTestSplit, ScaleSchemeBunch, LengthScale, \
GSUMDiagnostics
# import urllib
# import tables

# import warnings
# warnings.filterwarnings("error")

# See: https://ianstormtaylor.com/design-tip-never-use-black/
# softblack = '#262626'
softblack = 'k'  # Looks better when printed on tex file
gray = '0.7'

mpl.rcParams['figure.dpi'] = 180
mpl.rcParams['font.size'] = 9
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.edgecolor'] = softblack
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.labelcolor'] = softblack
mpl.rcParams['axes.linewidth']

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['xtick.color'] = softblack
mpl.rcParams['ytick.color'] = softblack
mpl.rcParams['xtick.minor.size'] = 2.4
mpl.rcParams['ytick.minor.size'] = 2.4

mpl.rcParams['legend.title_fontsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderaxespad'] = 0.8
mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=softblack, lw=0.8)
mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.05, dpi=300, format='pdf')

edgewidth = 0.6

# # We get the NN data from a separate place in our github respository.
# nn_online_pot = 'pwa93'
# nn_online_url = 'https://github.com/buqeye/buqeyebox/blob/master/nn_scattering/NN-online-Observables.h5?raw=true'
# nno_response = urllib.request.urlopen(nn_online_url) 
# nn_online_file = tables.open_file("nn_online_example.h5", driver="H5FD_CORE",
#                           driver_core_image=nno_response.read(),
#                           driver_core_backing_store=0)
# SGT_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/SGT').read()
# DSG_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/DSG').read()[:, :-1]
# AY_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/PB').read()[:, :-1]
# A_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/A').read()[:, :-1]
# D_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/D').read()[:, :-1]
# AXX_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/AXX').read()[:, :-1]
# AYY_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/AYY').read()[:, :-1]

# for each choice of scale and scheme, sets the total possible orders and nomenclature
EKM0p8fm = ScaleSchemeBunch('scattering_observables_EKM_R-0p8fm.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EKM", "0p8fm")
EKM0p9fm = ScaleSchemeBunch('scattering_observables_EKM_R-0p9fm.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EKM", "0p9fm")
EKM1p0fm = ScaleSchemeBunch('scattering_observables_EKM_R-1p0fm.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EKM", "1p0fm")
EKM1p1fm = ScaleSchemeBunch('scattering_observables_EKM_R-1p1fm.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EKM", "1p1fm")
EKM1p2fm = ScaleSchemeBunch('scattering_observables_EKM_R-1p2fm.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EKM", "1p2fm")

RKE400MeV = ScaleSchemeBunch('scattering_observables_RKE_L-400MeV.h5', \
            np.array([0, 2, 3, 4, 5, 6]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples']], \
            "RKE", "400MeV")
RKE450MeV = ScaleSchemeBunch('scattering_observables_RKE_L-450MeV.h5', \
            np.array([0, 2, 3, 4, 5, 6]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples']], \
            "RKE", "450MeV")
RKE500MeV = ScaleSchemeBunch('scattering_observables_RKE_L-500MeV.h5', \
            np.array([0, 2, 3, 4, 5, 6]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples']], \
            "RKE", "500MeV")
RKE550MeV = ScaleSchemeBunch('scattering_observables_RKE_L-550MeV.h5', \
            np.array([0, 2, 3, 4, 5, 6]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples']], \
            "RKE", "550MeV")

EMN450MeV = ScaleSchemeBunch('scattering_observables_EM-450MeV.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EMN", "450MeV")
EMN500MeV = ScaleSchemeBunch('scattering_observables_EM-500MeV.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EMN", "500MeV")
EMN550MeV = ScaleSchemeBunch('scattering_observables_EM-550MeV.h5', \
            np.array([0, 2, 3, 4, 5]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']], \
            "EMN", "550MeV")

GT0p9fm = ScaleSchemeBunch('scattering_observables_Gezerlis-0p9fm.h5', \
            np.array([0, 2, 3]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens']], \
            "GT", "0p9fm")
GT1p0fm = ScaleSchemeBunch('scattering_observables_Gezerlis-1p0fm.h5', \
            np.array([0, 2, 3]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens']], \
            "GT", "1p0fm")
GT1p1fm = ScaleSchemeBunch('scattering_observables_Gezerlis-1p1fm.h5', \
            np.array([0, 2, 3]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens']], \
            "GT", "1p1fm")
GT1p2fm = ScaleSchemeBunch('scattering_observables_Gezerlis-1p2fm.h5', \
            np.array([0, 2, 3]), \
            [plt.get_cmap(name) for name in ['Oranges', 'Greens']], \
            "GT", "1p2fm")

# scale_scheme_bunch_array = [EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm, \
#             RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV, EMN450MeV, EMN500MeV, \
#             EMN550MeV, GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm]

# # based on user input, allows only certain scale/scheme combinations through
# if argv[1] == "EKM0p8fm" or argv[1] == "EKM0p9fm" or argv[1] == "EKM1p0fm" \
#     or argv[1] == "EKM1p1fm" or argv[1] == "EKM1p2fm" or argv[1] == "EMN450MeV" \
#     or argv[1] == "EMN500MeV" or argv[1] == "EMN550MeV" or argv[1] == "RKE400MeV" \
#     or argv[1] == "RKE450MeV" or argv[1] == "RKE500MeV" or argv[1] == "RKE550MeV" \
#     or argv[1] == "GT0p9fm" or argv[1] == "GT1p0fm" or argv[1] == "GT1p1fm" \
#     or argv[1] == "GT1p2fm":
#     # print(argv[1])
#     scale_scheme_bunch_array = [b for b in scale_scheme_bunch_array \
#             if (b.potential_string + b.cutoff_string) == argv[1]]
#     # for b in scale_scheme_bunch_array: print(b.potential_string + b.cutoff_string)
# elif argv[1] == "all_potentials":
#     pass
# else:
#     raise NameError("Valid scheme/scale combinations: EKM0p8fm, EKM0p9fm, "+ 
#             "EKM1p0fm, EKM1p1fm, EKM1p2fm, EMN450MeV, EMN500MeV, "+
#             "EMN550MeV, RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV, "+
#             "GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm, all_potentials")
    
# # gets energies for evaluating observables from user
# energies_input = argv[3]
# E_input_array = [int(x) for x in list(energies_input.split(","))]
        
# # sets the method for parametrizing the dimensionless parameter Q
# Q_param_method_array = ["poly", "max", "sum"]

# # based on user input, allows only certain parametrizations through
# if argv[5] == "poly" or argv[5] == "max" or argv[5] == "sum":
#     # print(argv[5])
#     Q_param_method_array = [b for b in Q_param_method_array \
#             if b == argv[5]]
#     # print(Q_param_method_array)
# elif argv[5] == "all_Q":
#     pass
# else:
#     raise NameError("Valid Q parametrizations: poly, max, sum, all_Q")

# creates the training and testing masks for observables plotted against angle
Fullspaceanglessplit = TrainTestSplit("allangles", 6, 3, \
                                    xmin_train_factor = 0, xmax_train_factor = 1)
Forwardanglessplit = TrainTestSplit("forwardangles", 6, 3, \
                                    xmin_train_factor = 0, xmax_train_factor = 5/6)
Backwardanglessplit = TrainTestSplit("backwardangles", 6, 3, \
                                    xmin_train_factor = 1/6, xmax_train_factor = 1)
traintestsplit_vsangle_array = [Fullspaceanglessplit, Forwardanglessplit, Backwardanglessplit]

# creates the training and testing masks for observables plotted against energy
Nolowenergysplit = TrainTestSplit("nolowenergy", 3, 4, \
                            offset_train_min_factor = 100/350, xmin_train_factor = 100/350, \
                            offset_test_min_factor = 100/350, xmin_test_factor = 100/350, \
                            offset_train_max_factor = -50/350, offset_test_max_factor = -50/350)
Yeslowenergysplit = TrainTestSplit("yeslowenergy", 4, 4, \
                            offset_train_min_factor = 0, xmin_train_factor = 0.01, \
                            offset_test_min_factor = 0, xmin_test_factor = 0, \
                            offset_train_max_factor = -50/350, offset_test_max_factor = -50/350)
traintestsplit_vsenergy_array = [Nolowenergysplit, Yeslowenergysplit]

# # based on user input, allows only certain train/test splits through
# if argv[6] == "allangles" or argv[6] == "forwardangles" \
#     or argv[6] == "backwardangles":
#     # print(argv[6])
#     train_test_split_array = [b for b in traintestsplit_vsangle_array \
#             if b.name == argv[6]]
#     # print(train_test_split_array)
# elif argv[6] == "nolowenergy" or argv[6] == "yeslowenergy":
#     # print(argv[6])
#     train_test_split_array = [b for b in traintestsplit_vsenergy_array \
#             if b.name == argv[6]]
#     # print(train_test_split_array)
# elif argv[6] == "all_vsangle_splits":
#     train_test_split_array = traintestsplit_vsangle_array
# elif argv[6] == "all_vsenergy_splits":
#     train_test_split_array = traintestsplit_vsenergy_array
# else:
#     raise NameError("Valid train/test splits: allangles, forwardangles, "+
#             "backwardangles, nolowenergy, yeslowenergy, all_vsangle_splits"+
#             "all_vsenergy_splits")
    
# # bounds for 2D posterior PDF figures
# PosteriorBounds_deg = PosteriorBounds(1e-6, 100, 100, 300, 900, 300)
# PosteriorBounds_cos = PosteriorBounds(1e-6, 2, 100, 300, 900, 300)
    
# # gets orders for evaluation from user
# orders_input = argv[7]

# def README():
#     print("Arguments for GPAnalysis():\n")
#     print("scale_scheme_bunch_array (ScaleSchemeBunch list): EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm, RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV, EMN450MeV, EMN500MeV, EMN550MeV, GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm")
#     print("Default: [EKM0p9fm]\n")
#     print("observable_input (str list): \"SGT\", \"DSG\", \"AY\", \"A\", \"D\", \"AXX\", \"AYY\"")
#     print("Default: [\"DSG\"]\n")
#     print("E_input_array (int list): any integer x such that 1 <= x <= 350")
#     print("Must be [0] for SGT")
#     print("Default: [150]\n")
#     print("Q_param_method_array (str list): \"poly\", \"max\", \"cos\"")
#     print("Default: [\"poly\"]\n")
#     print("input_space_input (str list): \"Elab\", \"prel\" for SGT")
#     print("input_space_input (str list): \"deg\", \"cos\", \"qcm\", \"qcm2\" for all other observables")
#     print("Default: [\"cos\"]\n")
#     print("train_test_split_array (TrainTestSplit list): Nolowenergysplit, Yeslowenergysplit for SGT")
#     print("train_test_split_array (TrainTestSplit list): Fullspaceanglessplit, Forwardanglessplit, Backwardanglessplit for all other observables")
#     print("Default: Fullspaceanglessplit\n")
#     print("orders_input (int list, or str): 2, 3, 4, 5, 6")
#     print("May be \"all\" to evaluate all orders for all potentials")
#     print("Default: \"all\"\n")
#     print("length_scale_input (LengthScale)")
#     print("Default: LengthScale(0.25, 0.25, 4, whether_fit = True)\n")
#     print("fixed_sd (float): any positive float")
#     print("Default: None\n")

def GPAnalysis(scale_scheme_bunch_array = [EKM0p9fm], 
               observable_input = ["DSG"], 
               E_input_array = [150],
               Q_param_method_array = ["poly"], 
               input_space_input = ["cos"], 
               train_test_split_array = [Fullspaceanglessplit], 
               orders_input = "all", 
               length_scale_input = LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit = True),
               fixed_sd = None, 
               print_all = False):
    """
    scale_scheme_bunch_array (ScaleSchemeBunch list): potential/cutoff 
        combinations for evaluation.
    Built-in options: 
        EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm 
        ( https://doi.org/10.1140/epja/i2015-15053-8 );
        RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV 
        ( https://doi.org/10.1140/epja/i2018-12516-4 );
        EMN450MeV, EMN500MeV, EMN550MeV 
        ( https://doi.org/10.1103/PhysRevC.96.024004 );
        GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm 
        ( https://doi.org/10.1103/PhysRevC.90.054323 )
    Default: [EKM0p9fm]

    observable_input (str list): observables for evaluation. Note that SGT 
        should not be in the same list as other observables.
    Built-in options: "SGT", "DSG", "AY", "A", "D", "AXX", "AYY"
    Default: ["DSG"]

    E_input_array (int list): energies for evaluation. Note that SGT must be 
        treated differently since it is not evaluated at one energy at a time.
    May be any integer x such that 1 <= x <= 350
    Must be [0] for SGT
    Default: [150]

    Q_param_method_array (str list): methods of parametrizing the dimensionless 
        expansion parameter Q for evaluation.
    Built-in options: "poly", "max", "cos"
    Default: ["poly"]

    input_space_input (str list): input spaces for evaluation. Note that SGT 
        must be treated differently since it is not evaluated at one energy at 
        a time.
    Built-in options: "Elab", "prel" for SGT
    Built-in options: "deg", "cos", "qcm", "qcm2" for all other observables
    Default: ["cos"]

    train_test_split_array (TrainTestSplit list): splits of training and 
        testing points for evaluation. Note that SGT must be treated 
        differently since it is not evaluated at one energy at a time.
    Built-in options: Nolowenergysplit, Yeslowenergysplit for SGT
    Built-in options: Fullspaceanglessplit, Forwardanglessplit, 
        Backwardanglessplit for all other observables
    Default: Fullspaceanglessplit

    orders_input (int list, or str if "all"): orders for evaluation. May be 
        any list containing 2, 3, 4, 5, 6 in any order. May be "all" to 
        evaluate all orders for all potentials.
    Built-in options: [0, 2, 3, 4, 5] for EKM; [0, 2, 3, 4, 5, 6] for RKE; 
        [0, 2, 3, 4, 5] for EMN; [0, 2, 3] for GT+.
    Default: "all"

    length_scale_input (LengthScale): initial guess for the correlation 
        length in the kernel (as a factor of the total size of the input space) 
        plus boundaries of the fit procedure as factors of the initial guess 
        for the correlation length. Fitting may be bypassed when whether_fit = 
        False.
    Default: LengthScale(0.25, 0.25, 4, whether_fit = True)

    fixed_sd (float): fixed standard deviation for the Gaussian process fit. 
        May be any positive float. If None, then there is no fixed standard 
        deviation and it is calculated by the fitting procedure.
    Default: None
    """
    try:
    # runs through the potentials
        for o, scalescheme in enumerate(scale_scheme_bunch_array):
            # gets observable data from a file in the same folder
            SGT = scalescheme.get_data('SGT')
            DSG = scalescheme.get_data('DSG')
            AY = scalescheme.get_data('PB')
            A = scalescheme.get_data('A')
            D = scalescheme.get_data('D')
            AXX = scalescheme.get_data('AXX')
            AYY = scalescheme.get_data('AYY')
            t_lab = scalescheme.get_data('t_lab')
            degrees = scalescheme.get_data('degrees')
            
            # creates the bunch for each observable to be plotted against angle
            SGTBunch = ObservableBunch("SGT", SGT, E_input_array, '\sigma_{\mathrm{tot}}', \
                                        "dimensionful")
            DSGBunch = ObservableBunch("DSG", DSG, E_input_array, '\sigma', \
                                        "dimensionful")
            AYBunch = ObservableBunch("AY", AY, E_input_array, 'A_{y}', \
                                        "dimensionless")
            ABunch = ObservableBunch("A", A, E_input_array, 'A', \
                                        "dimensionless")
            DBunch = ObservableBunch("D", D, E_input_array, 'D', \
                                        "dimensionless")
            AXXBunch = ObservableBunch("AXX", AXX, E_input_array, 'A_{xx}', \
                                        "dimensionless")
            AYYBunch = ObservableBunch("AYY", AYY, E_input_array, 'A_{yy}', \
                                        "dimensionless")
        
            observable_array = [SGTBunch, DSGBunch, AYBunch, ABunch, DBunch, AXXBunch, AYYBunch]
            
            observable_array = [b for b in observable_array if b.name in observable_input]
            
            # # filters out observables based upon user input
            # if argv[2] == "SGT" or argv[2] == "DSG" or argv[2] == "AY" or argv[2] == "A" \
            #     or argv[2] == "D" or argv[2] == "AXX" or argv[2] == "AYY":
            #     # print(argv[2])
            #     observable_array = [b for b in observable_array if b.name == argv[2]]
            #     # print(vsangle_observable_array)
            # elif argv[2] == "our_obs":
            #     observable_array = [SGTBunch, DSGBunch, ABunch]
            # elif argv[2] == "all_obs":
            #     pass
            # else:
            #     raise NameError("Valid observables: SGT, DSG, AY, A, D, AXX, AYY, our_obs, "+
            #             "all_obs")
                
            # # turns the string argument for orders into an array for orders
            # if orders_input == "all":
            #     orders_input_array = scalescheme.orders_full
            # else:
            #     orders_input_array = [int(x) for x in list(orders_input.split(","))]
            #     # print(orders_input_array)
                
            # turns the string argument for orders into an array for orders
            if orders_input == "all":
                orders_input_array = scalescheme.orders_full
            else:
                orders_input_array = (orders_input.copy()).sort()
                
            # turns the array for orders into an array for colors
            colors_index_array = orders_input_array.copy()
            for i, o in enumerate(colors_index_array):
                colors_index_array[i] = o - 2
            # print(colors_index_array)
            
            # adds a 0 as the first entry in the array for orders if one is not already there
            if orders_input_array[0] != 0:
                orders_input_array = [0] + orders_input_array
            # print(orders_input_array)
            
            # creates a mask for orders and colors
            mask_orders = np.zeros(len(scalescheme.cmaps), dtype = bool)
            for i, o in enumerate(colors_index_array):
                mask_orders[o] = True
            # print(mask_orders)
            
            # This ensures we only analyze the non-trivial information at
            # O(Q^2), O(Q^3), O(Q^4), and O(Q^5)
            excluded = [0]
            mask_full = ~ np.isin(scalescheme.orders_full, excluded)
            Lambdab = 600
            
            # runs through the observables
            for m, observable in enumerate(observable_array):
                # runs through the energies at which to evaluate the observables
                for j, E_lab in enumerate(observable.energies):
                    # creates the bunches for the input spaces
                    DegBunch = InputSpaceBunch("deg", \
                                        degrees, \
                                        E_to_p(E_lab, "np"), \
                                        r'$\theta$ (deg)', \
                                        [r'$', observable.title, r'(\theta, E_{\mathrm{lab}}= ', E_lab, \
                                          '\,\mathrm{MeV})$'])
                    
                    # note that the input space here is -cos(theta), not cos(theta) (as it once was)
                    CosBunch = InputSpaceBunch("cos", \
                                        -np.cos(np.radians(degrees)), \
                                        np.array( [softmax_mom(E_to_p(E_lab, "np"), q) \
                                                  for q in deg_to_qcm(E_to_p(E_lab, "np"), degrees)] ), \
                                        r'$-\mathrm{cos}(\theta)$', \
                                        [r'$', observable.title, r'(-\mathrm{cos}(\theta), E_{\mathrm{lab}}= ', \
                                          E_lab, '\,\mathrm{MeV})$'])
            
                    QcmBunch = InputSpaceBunch("qcm", \
                                        deg_to_qcm(E_to_p(E_lab, "np"), degrees), \
                                        np.array( [softmax_mom(E_to_p(E_lab, "np"), q) \
                                                  for q in deg_to_qcm(E_to_p(E_lab, "np"), degrees)] ), \
                                        r'$q_{\mathrm{cm}}$ (MeV)', \
                                        [r'$', observable.title, r'(q_{\mathrm{cm}}, E_{\mathrm{lab}}= ', \
                                          E_lab, '\,\mathrm{MeV})$'])
            
                    Qcm2Bunch = InputSpaceBunch("qcm2", \
                                        deg_to_qcm2(E_to_p(E_lab, "np"), degrees), \
                                        np.array( [softmax_mom(E_to_p(E_lab, "np"), q) \
                                                  for q in deg_to_qcm(E_to_p(E_lab, "np"), degrees)] ), \
                                        r'$q_{\mathrm{cm}}^{2}$ (MeV$^{2}$)', \
                                        [r'$', observable.title, r'(q_{\mathrm{cm}}^{2}, E_{\mathrm{lab}}= ', \
                                          E_lab, '\,\mathrm{MeV})$'])
                        
                    ElabBunch = InputSpaceBunch("Elab", \
                                        t_lab, \
                                        E_to_p(t_lab, "np"), \
                                        r'$E_{\mathrm{lab}}$ (MeV)', \
                                        [r'$', observable.title, r'(E_{\mathrm{lab}})$'])
                
                    PrelBunch = InputSpaceBunch("prel", \
                                        E_to_p(t_lab, "np"), \
                                        E_to_p(t_lab, "np"), \
                                        r'$p_{\mathrm{rel}}$ (MeV)', \
                                        [r'$', observable.title, r'(p_{\mathrm{rel}})$'])
                
                    vsquantity_array = [DegBunch, CosBunch, QcmBunch, Qcm2Bunch, ElabBunch, PrelBunch]
                    vsquantity_array = [b for b in vsquantity_array if b.name in input_space_input]
                    
                    # # filters out input spaces based upon user input
                    # if argv[4] == "deg" or argv[4] == "cos" or argv[4] == "qcm" \
                    #     or argv[4] == "qcm2" or argv[4] == "Elab" or argv[4] == "prel":
                    #     # print(argv[4])
                    #     vsquantity_array = [b for b in vsquantity_array if b.name == argv[4]]
                    #     # print(vsangle_quantity_array)
                    # elif argv[4] == "all_vsangle_inputs":
                    #     vsquantity_array = [DegBunch, CosBunch, QcmBunch, Qcm2Bunch]
                    # elif argv[4] == "all_vsenergy_inputs":
                    #     vsquantity_array = [ElabBunch, PrelBunch]
                    # else:
                    #     raise NameError("Valid input spaces: deg, cos, qcm, qcm2, "+
                    #             "Elab, prel, all_inputs")
                    
                    # creates each input space bunch's title
                    for bunch in vsquantity_array: bunch.make_title()
            
                    # runs through the parametrization methods
                    for k, Q_param_method in enumerate(Q_param_method_array):
                        # runs through the input spaces
                        for i, vs_quantity in enumerate(vsquantity_array):
                            # runs through the training and testing masks
                            for l, traintestsplit in enumerate(train_test_split_array):
                                # conforms the training and testing masks to each input space
                                traintestsplit.make_masks(vs_quantity.input_space, observable.data)
                                # print("data = " + str(observable.data))
                                
                                # chooses a starting guess for the GP length scale optimization procedure
                                LengthScaleGuess = length_scale_input
                                LengthScaleGuess.make_guess(vs_quantity.input_space)
                                
                                # creates the GP with all its hyperparameters
                                ratio_dsg = Q_approx(vs_quantity.mom, Q_param_method, Lambda_b = Lambdab)
                                # print("ratio = " + str(ratio_dsg))
                                center = 0
                                df = 1
                                disp = 0
                                std_scale = 1
                                GPHyper_DSG = GPHyperparameters(LengthScaleGuess, center, ratio_dsg, \
                                                df = df, disp = disp, scale = std_scale, seed = 4, 
                                                sd = fixed_sd)
            
                                # information for naming the savefiles
                                FileName_DSG = FileNaming(scalescheme.potential_string, \
                                                scalescheme.cutoff_string, Q_param_method)
            
                                # information on the orders for each potential
                                Orders_DSG = OrderInfo(scalescheme.orders_full, mask_full, \
                                                scalescheme.colors, scalescheme.light_colors, \
                                                orders_restricted = orders_input_array, \
                                                mask_restricted = mask_orders)
            
                                # creates the object used to generate and plot statistical diagnostics
                                MyPlot = GSUMDiagnostics(observable, Lambdab, vs_quantity, \
                                        traintestsplit, GPHyper_DSG, Orders_DSG, FileName_DSG, \
                                        E_lab = E_lab, E_lab_x = t_lab, constrained = False)
            
                                # plots figures
                                MyPlot.PlotCoefficients()
                                MyPlot.PlotMD()
                                MyPlot.PlotPC()
                                MyPlot.PlotCredibleIntervals()
                                # if vs_quantity.name == "deg":
                                #     MyPlot.PlotPosteriorPDF(PosteriorBounds_deg)
                                # elif vs_quantity.name == "cos":
                                #     MyPlot.PlotPosteriorPDF(PosteriorBounds_cos)
                                MyPlot.Plotzilla()
    except:
        pass
    
    # prints all instances of the classes relevant for the arguments of 
    # GPAnalysis()
    if print_all:
        scalescheme_current_list = []
        observable_current_list = []
        inputspace_current_list = []
        traintest_current_list = []
        lengthscale_current_list = []
        
        for obj in gc.get_objects():
            if isinstance(obj, ScaleSchemeBunch):
                scalescheme_current_list.append(obj.name)
            elif isinstance(obj, ObservableBunch):
                observable_current_list.append(obj.name)
            elif isinstance(obj, InputSpaceBunch):
                inputspace_current_list.append(obj.name)
            elif isinstance(obj, TrainTestSplit):
                traintest_current_list.append(obj.name)
            elif isinstance(obj, LengthScale):
                lengthscale_current_list.append(obj.name)
        
        print("\n\n************************************")
        print("Available potentials: " + str(scalescheme_current_list))
        print("Available observables: " + str(observable_current_list))
        print("Available Q parametrizations: ['poly', 'max', 'sum']")
        print("Available input spaces: " + str(inputspace_current_list))
        print("Available train/test splits: " + str(traintest_current_list))
        print("Available length scales: " + str(lengthscale_current_list))
        print("************************************")