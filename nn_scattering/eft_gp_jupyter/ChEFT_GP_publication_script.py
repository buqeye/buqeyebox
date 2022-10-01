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
p_approx, deg_to_qcm, deg_to_qcm2, softmax_mom, GPHyperparameters, FileNaming, PosteriorBounds, \
OrderInfo, versatile_train_test_split, InputSpaceBunch, \
ObservableBunch, Interpolation, TrainTestSplit, ScaleSchemeBunch, LengthScale, \
GSUMDiagnostics
import copy as cp
import urllib
import tables

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

# We get the NN data from a separate place in our github respository.
nn_online_pot = 'pwa93'
nn_online_url = 'https://github.com/buqeye/buqeyebox/blob/master/nn_scattering/NN-online-Observables.h5?raw=true'
nno_response = urllib.request.urlopen(nn_online_url) 
nn_online_file = tables.open_file("nn_online_example.h5", driver="H5FD_CORE",
                          driver_core_image=nno_response.read(),
                          driver_core_backing_store=0)
SGT_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/SGT').read()
DSG_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/DSG').read()[:, :-1]
AY_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/PB').read()[:, :-1]
A_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/A').read()[:, :-1]
D_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/D').read()[:, :-1]
AXX_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/AXX').read()[:, :-1]
AYY_nn_online = nn_online_file.get_node('/' + nn_online_pot + '/AYY').read()[:, :-1]

# creates a dictionary that links the NN online data for each observable to the 
# eventual predictions for that observable by a given potential scheme and scale
online_data_dict = {"SGT" : SGT_nn_online, 
                    "DSG" : DSG_nn_online, 
                    "AY" : AY_nn_online, 
                    "A" : A_nn_online, 
                    "D" : D_nn_online, 
                    "AXX" : AXX_nn_online, 
                    "AYY" : AYY_nn_online}

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

# scale_scheme_bunch_array = [EKM0p8fm, EKM0p9fm, EKM1p0fm, EKM1p1fm, EKM1p2fm, 
#             RKE400MeV, RKE450MeV, RKE500MeV, RKE550MeV, 
#             EMN450MeV, EMN500MeV, EMN550MeV, 
            # GT0p9fm, GT1p0fm, GT1p1fm, GT1p2fm]

# creates the training and testing masks for observables plotted against angle
Fullspaceanglessplit = TrainTestSplit("allangles", 6, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 1)
Fullspaceanglessplit1 = TrainTestSplit("allangles1", 5, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 1)
Fullspaceanglessplit2 = TrainTestSplit("allangles2", 4, 4, 
                                    xmin_train_factor = 0, xmax_train_factor = 1)
Forwardanglessplit = TrainTestSplit("forwardangles", 6, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 5/6)
Backwardanglessplit = TrainTestSplit("backwardangles", 6, 3, 
                                    xmin_train_factor = 1/6, xmax_train_factor = 1)
Backwardanglessplit1 = TrainTestSplit("backwardangles1", 5, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 1, 
                                    xmin_test_factor = 1/5, xmax_test_factor = 1)
Forwardanglessplit2 = TrainTestSplit("forwardangles2", 6, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 5/6, 
                                    xmin_test_factor = 0, xmax_test_factor = 5/6)
Backwardanglessplit2 = TrainTestSplit("backwardangles2", 6, 3, 
                                    xmin_train_factor = 1/6, xmax_train_factor = 1, 
                                    xmin_test_factor = 1/6, xmax_test_factor = 1)
# Split1704 = TrainTestSplit("1704", 1, )
traintestsplit_vsangle_array = [Fullspaceanglessplit, Forwardanglessplit, Backwardanglessplit, 
                                Forwardanglessplit2, Backwardanglessplit2, Fullspaceanglessplit1, 
                                Fullspaceanglessplit2]

# creates the training and testing masks for observables plotted against energy
Nolowenergysplit = TrainTestSplit("nolowenergy", 3, 4, 
                            offset_train_min_factor = 100/350, xmin_train_factor = 100/350, 
                            offset_test_min_factor = 100/350, xmin_test_factor = 100/350, 
                            offset_train_max_factor = -50/350, offset_test_max_factor = -50/350)
Yeslowenergysplit = TrainTestSplit("yeslowenergy", 4, 4, 
                            offset_train_min_factor = 0, xmin_train_factor = 0.01, 
                            offset_test_min_factor = 0, xmin_test_factor = 0, 
                            offset_train_max_factor = -50/350, offset_test_max_factor = -50/350)
Allenergysplit = TrainTestSplit("allenergy", 4, 4, 
                            offset_train_min_factor = 0, xmin_train_factor = 0, 
                            offset_test_min_factor = 0, xmin_test_factor = 0, 
                            offset_train_max_factor = -50/350, offset_test_max_factor = -50/350)
Allenergysplit1 = TrainTestSplit("allenergy1", 5, 3, 
                                    xmin_train_factor = 0, xmax_train_factor = 1)
Allenergysplit2 = TrainTestSplit("allenergy2", 4, 4, 
                                    xmin_train_factor = 0, xmax_train_factor = 1)
traintestsplit_vsenergy_array = [Nolowenergysplit, Yeslowenergysplit, Allenergysplit, 
                                 Allenergysplit1, Allenergysplit2]

def GPAnalysis(scale_scheme_bunch_array = [EKM0p9fm], 
               observable_input = ["DSG"], 
               E_input_array = [150],
               deg_input_array = [0],
               Q_param_method_array = ["poly"], 
               p_param_method_array = ["Qofpq"], 
               input_space_input = ["cos"], 
               train_test_split_array = [Fullspaceanglessplit], 
               orders_input = "all", 
               length_scale_input = LengthScale("1/16-1_fitted", 0.25, 0.25, 4, whether_fit = True),
               fixed_sd = None, 
               Lambdab = 600,
               print_all_classes = False, 
               savefile_type = 'pdf', 
               plot_coeffs_bool = True, 
               plot_md_bool = True, 
               plot_pc_bool = True, 
               plot_ci_bool = True, 
               plot_pdf_bool = True, 
               plot_trunc_bool = True, 
               plot_plotzilla_bool = True, 
               save_coeffs_bool = True, 
               save_md_bool = True, 
               save_pc_bool = True, 
               save_ci_bool = True, 
               save_pdf_bool = True, 
               save_trunc_bool= True, 
               save_plotzilla_bool = True, 
               filename_addendum = ""):
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

    E_input_array (int list): lab energies in MeV for evaluation. Note that SGT 
        must be treated differently since it is not evaluated at one energy at 
        a time.
    May be any integer x such that 1 <= x <= 350
    Must be [0] for SGT
    If no evaluation at fixed lab energy is desired, set equal to [0]
    Default: [150]
    
    deg_input_array (int list): angles in degrees for evaluation.
    May be any integer x such that 1 <= x <= 179
    Must be [0] for SGT
    If no evaluation at fixed angle measure is desired, set equal to [0]
    Default: []

    Q_param_method_array (str list): methods of parametrizing the dimensionless 
        expansion parameter Q for evaluation.
    Built-in options: "poly", "max", "cos"
    Default: ["poly"]
    
    p_param_method_array (str list): methods of parametrizing the characteristic 
        momentum p in Q(p) for evaluation.
    Built-in options: "Qofprel", "Qofqcm", "Qofpq"
    Default: ["Qofpq"]

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
    
    Lambdab (float): breakdown scale for the theory (in MeV).
    Default: 600
    
    savefile_type (str): string for specifying the type of file to be saved.
    Default: 'png'
    
    plot_..._bool (bool): boolean for whether to plot different figures
    Default: True
    
    save_..._bool (bool): boolean for whether to save different figures
    Default: True
    
    filename_addendum (str): string for distinguishing otherwise similarly named
        files.
    Default: ''
    """
    mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.05, dpi=300, 
           format=savefile_type)
    
    # try:
    # runs through the potentials
    for o, scalescheme in enumerate(scale_scheme_bunch_array):
        # gets observable data from a local file
        # default location is the same as this program's
        try:
            SGT = scalescheme.get_data('SGT')
            DSG = scalescheme.get_data('DSG')
            AY = scalescheme.get_data('PB')
            A = scalescheme.get_data('A')
            D = scalescheme.get_data('D')
            AXX = scalescheme.get_data('AXX')
            AYY = scalescheme.get_data('AYY')
            t_lab = scalescheme.get_data('t_lab')
            degrees = scalescheme.get_data('degrees')
        except:
            raise Exception("Data could not be found in the location specified.")
        
        # creates the bunch for each observable to be plotted against angle
        SGTBunch = ObservableBunch("SGT", SGT, E_input_array, deg_input_array, 
                                 '\sigma_{\mathrm{tot}}', "dimensionful")
        DSGBunch = ObservableBunch("DSG", DSG, E_input_array, deg_input_array, 
                                 '\sigma', "dimensionful")
        # AYBunch = ObservableBunch("AY", AY, E_input_array, deg_input_array, 
        #                           'A_{y}', "dimensionless")
        # ABunch = ObservableBunch("A", A, E_input_array, deg_input_array, 
        #                           'A', "dimensionless")
        AYBunch = ObservableBunch("AY", AY, E_input_array, deg_input_array, 
                                  'A_{y}', "dimensionless", 
                                  constraint = [[0, -1], [0, 0], "angle"])
        ABunch = ObservableBunch("A", A, E_input_array, deg_input_array, 
                                  'A', "dimensionless", 
                                  constraint = [[0], [0], "angle"])
        DBunch = ObservableBunch("D", D, E_input_array, deg_input_array, 
                                 'D', "dimensionless")
        AXXBunch = ObservableBunch("AXX", AXX, E_input_array, deg_input_array,
                                 'A_{xx}', "dimensionless")
        AYYBunch = ObservableBunch("AYY", AYY, E_input_array, deg_input_array,
                                 'A_{yy}', "dimensionless")
    
        observable_array = [SGTBunch, DSGBunch, AYBunch, ABunch, DBunch, AXXBunch, AYYBunch]
        
        observable_array = [b for b in observable_array if b.name in observable_input]
            
        # turns the string argument for orders into an array for orders
        if orders_input == "all":
            orders_input_array = scalescheme.orders_full
        else:
            orders_input_array = np.array(orders_input.copy())
            orders_input_array.sort()
            
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
        
        # runs through the observables
        for m, observable in enumerate(observable_array):
            # runs through the parametrizations of p in Q(p)
            for p, p_param_method in enumerate(p_param_method_array):
                # runs through the energies at which to evaluate the observables
                for j, E_lab in enumerate(observable.energies):
                    # creates the bunches for the vs-angle input spaces
                    DegBunch = InputSpaceBunch("deg", 
                                        lambda x : x, 
                                        p_approx(p_param_method, E_to_p(E_lab, "np"), degrees), 
                                        r'$\theta$ (deg)', 
                                        [r'$', observable.title, r'(\theta, E_{\mathrm{lab}}= ', E_lab, 
                                          '\,\mathrm{MeV})$'])
                    CosBunch = InputSpaceBunch("cos", 
                                        lambda x : -np.cos(np.radians(x)), 
                                        p_approx(p_param_method, E_to_p(E_lab, "np"), degrees), 
                                        r'$-\mathrm{cos}(\theta)$', 
                                        [r'$', observable.title, r'(-\mathrm{cos}(\theta), E_{\mathrm{lab}}= ', 
                                          E_lab, '\,\mathrm{MeV})$'])
                    QcmBunch = InputSpaceBunch("qcm", 
                                        lambda x : deg_to_qcm(E_to_p(E_lab, "np"), x), 
                                        p_approx(p_param_method, E_to_p(E_lab, "np"), degrees), 
                                        r'$q_{\mathrm{cm}}$ (MeV)', 
                                        [r'$', observable.title, r'(q_{\mathrm{cm}}, E_{\mathrm{lab}}= ', 
                                          E_lab, '\,\mathrm{MeV})$'])
                    Qcm2Bunch = InputSpaceBunch("qcm2", 
                                        lambda x : deg_to_qcm2(E_to_p(E_lab, "np"), x), 
                                        p_approx(p_param_method, E_to_p(E_lab, "np"), degrees), 
                                        r'$q_{\mathrm{cm}}^{2}$ (MeV$^{2}$)', 
                                        [r'$', observable.title, r'(q_{\mathrm{cm}}^{2}, E_{\mathrm{lab}}= ', 
                                          E_lab, '\,\mathrm{MeV})$'])
                
                    vsquantity_array = [DegBunch, CosBunch, QcmBunch, Qcm2Bunch]
                    vsquantity_array = [b for b in vsquantity_array if b.name in input_space_input]
                    
                    # creates each input space bunch's title
                    for bunch in vsquantity_array: bunch.make_title()
            
                    # runs through the parametrization methods
                    for k, Q_param_method in enumerate(Q_param_method_array):
                        # runs through the input spaces
                        for i, vs_quantity in enumerate(vsquantity_array):
                            # creates the posterior bounds for the Lambda-ell
                            # posterior probability distribution function scaled using 
                            # the current value of Lambdab and an estimate of 
                            # 1/4 of the total input space size for the correlation 
                            # length
                            MyPosteriorBounds = PosteriorBounds((max(vs_quantity.input_space(degrees)) - min(vs_quantity.input_space(degrees))) / 9, 
                                                                (max(vs_quantity.input_space(degrees)) - min(vs_quantity.input_space(degrees))) / 2, 
                                                                100, 
                                                                Lambdab * 0.5, 
                                                                Lambdab * 1.5, 
                                                                100)
                            # runs through the training and testing masks
                            for l, traintestsplit in enumerate(train_test_split_array):
                                # conforms the training and testing masks to each input space
                                traintestsplit.make_masks(vs_quantity.input_space(degrees), observable.data)
                                # print("data = " + str(observable.data))
                                
                                # chooses a starting guess for the GP length scale optimization procedure
                                LengthScaleGuess = length_scale_input
                                LengthScaleGuess.make_guess(vs_quantity.input_space(degrees))
                                
                                # creates the GP with all its hyperparameters
                                ratio_dsg = Q_approx(vs_quantity.mom, Q_param_method, Lambda_b = Lambdab)
                                center = 0
                                df = 1
                                disp = 0
                                std_scale = 0.25
                                GPHyper = GPHyperparameters(LengthScaleGuess, center, ratio_dsg, 
                                                df = df, disp = disp, scale = std_scale, seed = None, 
                                                sd = fixed_sd)
            
                                # information for naming the savefiles
                                FileName = FileNaming(scalescheme.potential_string, 
                                                scalescheme.cutoff_string, Q_param_method, 
                                                p_param_method, 
                                                filename_addendum = filename_addendum)
            
                                # information on the orders for each potential
                                Orders = OrderInfo(scalescheme.orders_full, mask_full, \
                                                scalescheme.colors, scalescheme.light_colors, \
                                                orders_restricted = orders_input_array, \
                                                mask_restricted = mask_orders)
                               
                                # creates the object used to generate and plot statistical diagnostics
                                MyPlot = GSUMDiagnostics(observable, Lambdab, vs_quantity, 
                                        traintestsplit, GPHyper, Orders, 
                                        FileName, 
                                        fixed_quantity = ["energy", E_lab, t_lab, "MeV"], 
                                        x_quantity = ["angle", degrees, "degrees"], 
                                        posteriorgrid = MyPosteriorBounds)
                                
                                # plots figures
                                if plot_coeffs_bool:
                                    MyPlot.PlotCoefficients(whether_save = save_coeffs_bool)
                                if plot_md_bool:
                                    MyPlot.PlotMD(whether_save = save_md_bool)
                                if plot_pc_bool:
                                    MyPlot.PlotPC(whether_save = save_pc_bool)
                                if plot_ci_bool:
                                    MyPlot.PlotCredibleIntervals(whether_save = save_ci_bool)
                                if plot_pdf_bool:
                                    MyPlot.PlotPosteriorPDF(whether_save = save_pdf_bool)
                                if plot_trunc_bool:
                                    MyPlot.PlotTruncationErrors(online_data_dict[observable.name], 
                                                                whether_save = save_trunc_bool, 
                                                                residual_plot = False)
                                if plot_plotzilla_bool:
                                    MyPlot.Plotzilla(whether_save = save_plotzilla_bool)
            
                for j, angle_lab in enumerate(observable.angles):
                    # creates the bunches for the vs-energy input spaces
                    ElabBunch = InputSpaceBunch("Elab", 
                                        lambda x : x, 
                                        p_approx("Qofprel", E_to_p(t_lab, "np"), degrees), 
                                        r'$E_{\mathrm{lab}}$ (MeV)', 
                                        [r'$', observable.title, r'(E_{\mathrm{lab}})$'])
                
                    PrelBunch = InputSpaceBunch("prel", 
                                        lambda x : E_to_p(x, "np"), 
                                        p_approx("Qofprel", E_to_p(t_lab, "np"), degrees), 
                                        r'$p_{\mathrm{rel}}$ (MeV)', 
                                        [r'$', observable.title, r'(p_{\mathrm{rel}})$'])
                
                    vsquantity_array = [ElabBunch, PrelBunch]
                    vsquantity_array = [b for b in vsquantity_array if b.name in input_space_input]
                    
                    # creates each input space bunch's title
                    for bunch in vsquantity_array: bunch.make_title()
                    # runs through the parametrization methods
                    for k, Q_param_method in enumerate(Q_param_method_array):
                        # runs through the input spaces
                        for i, vs_quantity in enumerate(vsquantity_array):
                            # creates the posterior bounds for the Lambda-ell
                            # posterior probability distribution function scaled using 
                            # the current value of Lambdab and an estimate of 
                            # 1/4 of the total input space size for the correlation 
                            # length
                            MyPosteriorBounds = PosteriorBounds((max(vs_quantity.input_space(t_lab)) - min(vs_quantity.input_space(t_lab))) / 9, 
                                                                (max(vs_quantity.input_space(t_lab)) - min(vs_quantity.input_space(t_lab))) / 2, 
                                                                100, 
                                                                Lambdab * 0.5, 
                                                                Lambdab * 1.5, 
                                                                100)
                            # runs through the training and testing masks
                            for l, traintestsplit in enumerate(train_test_split_array):
                                # conforms the training and testing masks to each input space
                                try:
                                    traintestsplit.make_masks(vs_quantity.input_space(t_lab), observable.data.swapaxes(1, 2))
                                except:
                                    traintestsplit.make_masks(vs_quantity.input_space(t_lab), observable.data)
                                
                                # chooses a starting guess for the GP length scale optimization procedure
                                LengthScaleGuess = length_scale_input
                                LengthScaleGuess.make_guess(vs_quantity.input_space(t_lab))
                                
                                # creates the GP with all its hyperparameters
                                ratio_dsg = Q_approx(vs_quantity.mom, Q_param_method, Lambda_b = Lambdab)
                                center = 0
                                df = 1
                                disp = 0
                                std_scale = 0.25
                                GPHyper = GPHyperparameters(LengthScaleGuess, center, ratio_dsg, 
                                                df = df, disp = disp, scale = std_scale, seed = None, 
                                                sd = fixed_sd)
            
                                # information for naming the savefiles
                                FileName = FileNaming(scalescheme.potential_string, 
                                                scalescheme.cutoff_string, Q_param_method, 
                                                p_param_method, 
                                                filename_addendum = filename_addendum)
            
                                # information on the orders for each potential
                                Orders = OrderInfo(scalescheme.orders_full, mask_full, 
                                                scalescheme.colors, scalescheme.light_colors, 
                                                orders_restricted = orders_input_array, 
                                                mask_restricted = mask_orders)
                                
                                
                                # creates the object used to generate and plot statistical diagnostics
                                MyPlot = GSUMDiagnostics(observable, Lambdab, vs_quantity, 
                                        traintestsplit, GPHyper, Orders, 
                                        FileName, 
                                        fixed_quantity = ["angle", angle_lab, degrees, "degrees"], 
                                        x_quantity = ["energy", t_lab, "MeV"], 
                                        posteriorgrid = MyPosteriorBounds)
                                
                                # plots figures
                                if plot_coeffs_bool:
                                    MyPlot.PlotCoefficients(whether_save = save_coeffs_bool)
                                if plot_md_bool:
                                    MyPlot.PlotMD(whether_save = save_md_bool)
                                if plot_pc_bool:
                                    MyPlot.PlotPC(whether_save = save_pc_bool)
                                if plot_ci_bool:
                                    MyPlot.PlotCredibleIntervals(whether_save = save_ci_bool)
                                if plot_pdf_bool:
                                    MyPlot.PlotPosteriorPDF(whether_save = save_pdf_bool)
                                if plot_trunc_bool:
                                    MyPlot.PlotTruncationErrors(online_data_dict[observable.name], 
                                                                whether_save = save_trunc_bool, 
                                                                residual_plot = False)
                                if plot_plotzilla_bool:
                                    MyPlot.Plotzilla(whether_save = save_plotzilla_bool)
    # except:
    #     print("Error encountered in running loop.")
    
    # prints all instances of the classes relevant for the arguments of 
    # GPAnalysis()
    if print_all_classes:
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