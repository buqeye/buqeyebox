import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# from matplotlib.text import Text
import numpy as np
# import scipy.stats as stats
from scipy.interpolate import interp1d
# import os
import h5py
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import math
# import re
# import urllib
# import tables
# import pandas
# import seaborn
import itertools
import ray
import time
import decimal
# See: https://ianstormtaylor.com/design-tip-never-use-black/
# softblack = '#262626'
softblack = 'k'  # Looks better when printed on tex file
gray = '0.7'

mpl.rcParams['figure.dpi'] = 180
mpl.rcParams['font.size'] = 9
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['axes.labelsize'] = 14 # 9
mpl.rcParams['axes.edgecolor'] = softblack
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.labelcolor'] = softblack
mpl.rcParams['axes.linewidth']

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 11 # 9
mpl.rcParams['ytick.labelsize'] = 11 # 9
mpl.rcParams['xtick.color'] = softblack
mpl.rcParams['ytick.color'] = softblack
mpl.rcParams['xtick.minor.size'] = 2.4
mpl.rcParams['ytick.minor.size'] = 2.4

mpl.rcParams['legend.title_fontsize'] = 9
mpl.rcParams['legend.fontsize'] = 14 # 9
mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderaxespad'] = 0.8
mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

mpl.rcParams['lines.markersize'] = 10

text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=softblack, lw=0.8)
mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.05, dpi=300, format='pdf')

cmaps = [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']]
colors = [cmap(0.55 - 0.1 * (i==0)) for i, cmap in enumerate(cmaps)]
light_colors = [cmap(0.25) for cmap in cmaps]
edgewidth = 0.6

# Constants: proton/neutron masses and hbar
m_p = 938.27208  # MeV/c^2
m_n = 939.56541  # MeV/c^2
hbarc = 197.33  # Mev-fm


def correlation_coefficient(x, y, pdf):
    # normalizes the pdf
    pdf /= np.trapz(np.trapz(pdf, x = y, axis = 0), x = x, axis = 0)
    # print("pdf = " + str(pdf))
    
    # finds the maximum value
    pdf_max = np.amax(pdf)
    # print("pdf_max = " + str(pdf_max))
    
    # figures out the x and y coordinates of the max
    # print(np.argwhere(pdf == pdf_max))
    x_max = x[np.argwhere(pdf == pdf_max)[0, 1]]
    # print("x_max = " + str(x_max))
    y_max = y[np.argwhere(pdf == pdf_max)[0, 0]]
    # print("y_max = " + str(y_max))
    
    # finds variance in x and y
    sigma_x_sq = np.trapz(
                np.trapz(np.tile((x - x_max)**2, (len(y), 1)) * pdf, 
                         x = x, axis = 1), 
                x = y, axis = 0)
    # print("sigma_x_sq = " + str(sigma_x_sq))
#     sigma_x_sq = np.trapz(pdf @ (x - x_max)**2, x = y, axis = 0)
#     print("sigma_x_sq = " + str(sigma_x_sq))
    sigma_y_sq = np.trapz(
                np.trapz(np.tile(np.reshape((y - y_max)**2, (len(y), 1)), (1, len(x))) * pdf, 
                         x = y, axis = 0), 
                x = x, axis = 0)
#     print("sigma_y_sq = " + str(sigma_y_sq))
#     sigma_y_sq = np.trapz((y - y_max)**2 @ pdf, x = x, axis = 0)
    # print("sigma_y_sq = " + str(sigma_y_sq))
    
    # finds sigmaxy
    sigma_xy_sq = np.trapz(
                np.trapz(np.tile(np.reshape(y - y_max, (len(y), 1)), (1, len(x))) * \
                             np.tile(x - x_max, (len(y), 1)) * pdf, 
                         x = x, axis = 1), 
                x = y, axis = 0)
    # print("sigma_xy_sq = " + str(sigma_xy_sq))
#     sigma_xy_sq = (y - y_max) @ pdf @ (x - x_max)
#     print("sigma_xy_sq = " + str(sigma_xy_sq))
    
    # finds the correlation coefficient
    corr_coeff = sigma_xy_sq / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq))
    # print(corr_coeff)
    
    return corr_coeff

def mean_and_stddev(x, pdf):
    # normalizes the pdf
    pdf /= np.trapz(pdf, x = x, axis = 0)
    # print("pdf = " + str(pdf))
    
    # finds the maximum value
    pdf_max = np.amax(pdf)
    # print("pdf_max = " + str(pdf_max))
    
    # figures out the x coordinate of the max
    # print(np.argwhere(pdf == pdf_max))
    x_max = x[np.argwhere(pdf == pdf_max)][0]
    # print("x_max = " + str(x_max))
    
    # finds the mean
    mean = np.trapz(x * pdf, x = x, axis = 0)
    # print("mean = " + str(mean))
    
    # finds the standard deviation
    sigma_x = np.sqrt(np.trapz((x - x_max)**2 * pdf, x = x, axis = 0))
    # print("sigma_x = " + str(sigma_x))
    
    return mean, sigma_x

def joint_plot(ratio = 1, height = 3):
    """Taken from Seaborn JointGrid"""
    fig = plt.figure(figsize=(height, height))
    gsp = plt.GridSpec(ratio+1, ratio+1)

    ax_joint = fig.add_subplot(gsp[1:, :-1])
    ax_marg_x = fig.add_subplot(gsp[0, :-1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gsp[1:, -1], sharey=ax_joint)

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Make the grid look nice
    from seaborn import utils
    # utils.despine(fig)
    utils.despine(ax=ax_marg_x, left=True)
    utils.despine(ax=ax_marg_y, bottom=True)
    fig.tight_layout(h_pad=0, w_pad=0)
    
    ax_marg_y.tick_params(axis='y', which='major', direction='out')
    ax_marg_x.tick_params(axis='x', which='major', direction='out')
    ax_marg_y.tick_params(axis='y', which='minor', direction='out')
    ax_marg_x.tick_params(axis='x', which='minor', direction='out')
    ax_marg_y.margins(x=0.1, y=0.)

    fig.subplots_adjust(hspace=0, wspace=0)
    
    return fig, ax_joint, ax_marg_x, ax_marg_y

def corner_plot(n_plots = 3, height = 9):
    """Taken from Seaborn JointGrid"""
    fig = plt.figure(figsize=(height, height))
    gsp = plt.GridSpec(n_plots, n_plots, wspace = 0.05, hspace = 0.05)

#     ax_joint_array_unshaped = [[fig.add_subplot(gsp[n_plots * i + j - 1]) 
#                       for j in range(1, i + 1)]
#                       for i in range(1, n_plots)]
#     print(ax_joint_array_unshaped)
#     print(fig.axes)
#     ax_joint_array_iter = list(itertools.chain.from_iterable(ax_joint_array_unshaped))
#     print(ax_joint_array_iter)
#     ax_joint_array = np.reshape(ax_joint_array_iter, 
#                 int(n_plots * (n_plots - 1) / 2))
#     print(ax_joint_array)
#     print("ax_joint_array has shape " + str(np.shape(ax_joint_array)))
    
    for i in range(1, n_plots):
        for j in range(1, i + 1):
            if ((n_plots * i + j - 1) % n_plots != 0) and ((n_plots * i + j - 1) < (n_plots * (n_plots - 1))):
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1], 
                                xticklabels = [], 
                                yticklabels = [])
            elif (n_plots * i + j - 1) % n_plots != 0:
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1], 
                                yticklabels = [])
            elif (n_plots * i + j - 1) < (n_plots * (n_plots - 1)):
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1], 
                                xticklabels = [])
            else:
                # print(n_plots * i + j - 1)
                # print("I have both sets of labels.")
                fig.add_subplot(gsp[n_plots * i + j - 1])
    # print(fig.axes)
    ax_joint_array = np.reshape(fig.axes, 
                int(n_plots * (n_plots - 1) / 2))
    # print(ax_joint_array)
    # print("ax_joint_array has shape " + str(np.shape(ax_joint_array)))
    # print(ax_joint_array[-1].get_yticklabels())
    # print(ax_joint_array[-2].get_yticklabels())
    # print(ax_joint_array[-3].get_yticklabels())
    # print("\n")
    # print(ax_joint_array[-1].get_xticklabels())
    # print(ax_joint_array[-2].get_xticklabels())
    # print(ax_joint_array[-3].get_xticklabels())
    ax_marg_array = np.reshape(
                    [fig.add_subplot(gsp[i * (n_plots + 1)],
                            yticklabels = [], yticks = [], 
                            xticklabels = []) 
                     for i in range(0, n_plots)], 
                (n_plots))
    
    
    # print((i * (n_plots + 1) == n_plots**2 - 1))
    # print(fig.axes)
    # print("ax_marg_array has shape " + str(np.shape(ax_marg_array)))
    
    ax_title = fig.add_subplot(gsp[n_plots - 1])
    ax_title.axis('off')
    
    return fig, ax_joint_array, ax_marg_array, ax_title

def offset_xlabel(ax):
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax

def sig_figs(number, n_figs):
    """
    Parameters
    ----------
    number : float
        A number.
    n_figs : int
        Number of significant figures.

    Returns
    -------
    None.

    """
    # formats the number as a string
    number_string = np.format_float_positional(
                    np.float64(
                    np.format_float_scientific(
                    number, precision = n_figs - 1)))
    # print("number_string = " + number_string)
    
    # eliminates any unncessary zeros and decimal points
    # while((np.float64(number_string) > 10**(n_figs - 1)) and ((number_string[-1] == '0') or (number_string[-1] == '.'))):
    #     number_string = number_string[:-1]
    if((np.float64(number_string) > 10**(n_figs - 1)) and (number_string[-1] == '.')):
        number_string = number_string[:-1]
        # print("We chopped off the decimal point.")
        return np.int(number_string)
    else:
        return np.float64(number_string)
    
def round_to_same_digits(number, comparand):
    # print("We called the function correctly.")
    # print(str(comparand))
    # print(str(decimal.Decimal(str(comparand)).as_tuple().exponent))
    if decimal.Decimal(str(comparand)).as_tuple().exponent == 0:
        return int(number)
    else:
        return np.around(number, decimals = decimal.Decimal(str(comparand)).as_tuple().exponent)

def E_to_p(E_lab, interaction = "np", **kwargs):
    """Returns p in MeV.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """

    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    p_rel = np.sqrt(
        E_lab * m2**2 * (E_lab + 2 * m1) /
        ((m1 + m2)**2 + 2 * m2 * E_lab)
        )
    return p_rel

def Qsum_to_Qsmoothmax(m_pi):
    return m_pi / 320 + 1

def Q_approx(p, Q_parametrization, Lambda_b, m_pi = 138, interaction = 'np', 
             single_expansion = False):
    """
    Returns the dimensionless expansion parameter Q.
    
    Parameters
    ----------
    p (float or array) : momentum (in MeV)
    Q_parametrization (str) : can be "smoothmax", "max", or "sum"
    Lambda_b (float) : value for the cutoff (in MeV)
    interaction (str) : can be "np", "nn", or "pp"
    """
    if single_expansion:
        m_pi = 0
    # else:
    #     m_pi = 200  # Set to 0 to just return p/Lambda_b
    
#     p = E_to_p(E, interaction)
    
    if Q_parametrization == "smoothmax":
        # Interpolate to smooth the transition from m_pi to p with a ratio 
        # of polynomials
        n = 8
        q = (m_pi**n + p**n) / (m_pi**(n-1) + p**(n-1)) / Lambda_b
        return q
    
    elif Q_parametrization == "max":
        # Transition from m_pi to p with a maximum function
        try:
            q = [max(P, m_pi) / Lambda_b for P in p]
        except:
            q = max(p, m_pi) / Lambda_b
        return q
    
    elif Q_parametrization == "sum":
        # Transition from m_pi to p with a simple sum
        q = (p + m_pi) / (Qsum_to_Qsmoothmax(m_pi) * Lambda_b)
        # q = (p + m_pi) / Lambda_b
        return q
    
def p_approx(p_name, prel, degrees):
    """
    Returns the dimensionless expansion parameter Q.
    
    Parameters
    ----------
    p_name (str): name for the parametrization of the momentum
    prel (float): relative momentum for the interaction (in MeV)
    degrees (float array): degrees
    """
    
    if p_name == "Qofprel":
        try:
            return np.array(prel * np.ones(len(degrees)))
        except:
            return np.array(prel)
    
    elif p_name == "Qofqcm":
        return np.array(deg_to_qcm(prel, degrees))
    
    elif p_name == "Qofpq":
        return np.array( [softmax_mom(prel, q) 
            for q in deg_to_qcm(prel, degrees)] )

def deg_fn(deg_input, **kwargs):
    return deg_input

def neg_cos(deg_input, **kwargs):
    return -1 * np.cos(np.radians(deg_input))

def deg_to_qcm(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return p_input * np.sqrt( 2 * (1 - np.cos(np.radians(deg_input))) )

def deg_to_qcm2(p_input, deg_input, **kwargs):
    """
    Returns the center-of-momentum momentum transfer q squared, in MeV^2.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return (p_input * np.sqrt( 2 * (1 - np.cos(np.radians(deg_input))) ))**(2)

def Elab_fn(E_lab, **kwargs):
    return E_lab

def sin_thing(deg, **kwargs):
    return np.array([np.sin(np.radians(d)) if d <= 90 else 2 - np.sin(np.radians(d)) for d in deg])

def softmax_mom(p, q, n = 5):
    """
    Two-place softmax function.
    
    Parameters
    ----------
    p       = float
            one interpolant.
    q       = float
            another interpolant.
    n       = float
            scaling parameter.
    """
    return 1 / n * math.log(1.01**(n * p) + 1.01**(n * q), 1.01)

def Lb_logprior(Lambda_b):
    """Melendez et al., Eq. (31)"""
    # return np.where((300 <= Lambda_b) & (Lambda_b <= 1500), np.log(1. / Lambda_b), -np.inf)
    return np.where((300 <= Lambda_b) & (Lambda_b <= 1500), 0, -np.inf)

def mpieff_logprior(m_pi):
    """Melendez et al., Eq. (31)"""
    # return np.where((50 <= m_pi) & (m_pi <= 300), np.log(1. / m_pi), -np.inf)
    return np.where((10 <= m_pi) & (m_pi <= 500), 0, -np.inf)

def compute_posterior_intervals(model, data, ratios, ref, orders, max_idx, logprior, Lb):
    # print("We're about to fit.")
    # print("data has shape " + str(np.shape(data[:max_idx+1].T)))
    # print("ratio has shape " + str(np.shape(ratios[0])))
    # print("ref has shape " + str(np.shape(ref)))
    # print("orders has shape " + str(np.shape(orders[:max_idx+1])))
    model.fit(data[:max_idx+1].T, ratio=ratios[0], ref=ref, orders=orders[:max_idx+1])
    # raise ValueError("something")
    log_like = np.array([model.log_likelihood(ratio=ratio) for ratio in ratios])
    log_like += logprior
    posterior = np.exp(log_like - np.max(log_like))
    posterior /= np.trapz(posterior, x=Lb)  # Normalize

    bounds = np.zeros((2,2))
    for i, p in enumerate([0.68, 0.95]):
        # bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
        bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb)

    median = gm.median_pdf(pdf=posterior, x=Lb)
    return posterior, bounds, median

def draw_summary_statistics(bounds68, bounds95, median, height=0, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(bounds68, [height, height], c='gray', lw=6, solid_capstyle='round')
    ax.plot(bounds95, [height, height], c='gray', lw=2, solid_capstyle='round')
    ax.plot([median], [height], c='white', marker='o', zorder=10, markersize=3)

class GPHyperparameters:
    def __init__(self, ls_class, center, ratio, nugget = 1e-10, seed = None, df = np.inf, 
                 disp = 0, scale = 1, sd = None):
        """
        Class for the hyperparameters of a Gaussian process.
        """
        self.ls = ls_class.ls_guess
        self.ls_lower = ls_class.ls_bound_lower
        self.ls_upper = ls_class.ls_bound_upper
        self.whether_fit = ls_class.whether_fit
        self.center = center
        self.ratio = ratio
        self.nugget = nugget
        self.seed = seed
        self.df = df
        self.disp = disp
        self.scale = scale
        self.sd = sd
        
class FileNaming:
    def __init__(self, scheme, scale, Q_param, p_param, filename_addendum = ""):
        """
        scheme (str) : name of the scheme
        scale (str) : name of the scale
        Q_param (str) : name of the Q parametrization
        p_param (str) : name of the p parametrization
        filename_addendum (str) : optional extra string
        """
        self.scheme = scheme
        self.scale = scale
        self.Q_param = Q_param
        self.p_param = p_param
        self.filename_addendum = filename_addendum

def find_nearest_val(array, value):
    """
    Finds the value in array closest to value and returns that entry.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    """
    Finds the value in array closest to value and returns that entry.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mask_mapper(array_from, array_to, mask_from):
    """
    Converts from one mask to another by mapping the entries of the first to the nearest-in-
    value entries in the second.
    """
    mask_array = [( np.argwhere(array_to == find_nearest_val(array_to, i)) ) for i in array_from[mask_from]]
    mask = np.zeros(len(array_from))
    for i in range(len(mask_array)):
        mask[mask_array[i]] = 1
    return np.array(mask.astype(int), dtype = bool)

class PosteriorBounds:
    def __init__(self, x_lower, x_upper, x_n, y_lower, y_upper, y_n):
        """
        Class for the boundaries of the 2D posterior PDF plot and the mesh on which it is plotted.
        """
        self.x_vals = np.linspace(x_lower, x_upper, x_n)
        self.y_vals = np.linspace(y_lower, y_upper, y_n)

class RandomVariable:
    def __init__(self, var, name, label, units, ticks, logprior, logprior_name):
        self.var = var
        self.name = name
        self.label = label
        self.units = units
        self.ticks = ticks
        self.logprior = logprior
        self.logprior_name = logprior_name

class OrderInfo:
    def __init__(self, orders_array, orders_mask, colors_array, lightcolors_array, 
            orders_restricted = [], mask_restricted = [], orders_names_dict = None, 
            orders_labels_dict = None):
        """
        Class for the number of orders under consideration and the color for each.
        
        Parameters
        ----------
        orders_array (array): list of orders at which the potential CAN BE evaluated
        orders_mask (array): boolean mask corresponding to orders_array
        colors_array (array): list of colors corresponding to each order
        lightcolors_array (array): list of lighter versions of colors_array
        orders_restricted (array): list of orders at which the potential WILL BE evaluated
            Set to orders_array if no value is given.
        mask_restricted (array): boolean mask corresponding to orders_restricted
            Set to orders_mask if no value is given.
        orders_names_dict (dict): dictionary method linking the numerical indices (int) 
            of EFT orders and their corresponding abbreviations (str)
        orders_names_dict (dict): dictionary method linking the numerical indices (int) 
            of EFT orders and their corresponding math-mode-formatted labels (str)
        """
        self.orders_full = orders_array
        self.mask_full = orders_mask
        self.colors_array = colors_array
        self.lightcolors_array = lightcolors_array
        
        if orders_restricted == []:
            self.orders_restricted = self.orders_full
        else:
            self.orders_restricted = orders_restricted
        if mask_restricted == []:
            self.mask_restricted = self.mask_full
        else:
            self.mask_restricted = mask_restricted
        
        self.orders_names_dict = orders_names_dict
        self.orders_labels_dict = orders_labels_dict

def versatile_train_test_split(interp_obj, n_train, n_test_inter = 1, isclose_factor = 0.01, \
            offset_train_min = 0, offset_train_max = 0, xmin_train = None, xmax_train = None, \
            offset_test_min = 0, offset_test_max = 0, xmin_test = None, xmax_test = None, \
            train_at_ends = True, test_at_ends = False):
    """
    Returns the training and testing points in the input space and the corresponding 
    (interpolated) data values

    Parameters
    ----------
    interp_obj (InterpObj) : function generated with scipy.interpolate.interp1d(x, y), plus
        x and y
    n_train (int) : number of intervals into which to split x, with training points at the 
        edges of each interval
    n_test_inter (int) : number of subintervals into which to split the intervals between 
        training points, with testing points at the edges of each subinterval
    isclose_factor (float) : fraction of the total input space for the tolerance of making
        sure that training and testing points don't coincide
    offset_train_min (float) : value above the minimum of the input space where the first 
        potential training point ought to go
    offset_train_max (float) : value below the maximum of the input space where the last 
        potential training point ought to go
    xmin_train (float) : minimum value within the input space below which there ought not to
        be training points
    xmax_train (float) : maximum value within the input space above which there ought not to
        be training points
    offset_test_min (float) : value above the minimum of the input space where the first 
        potential testing point ought to go
    offset_test_max (float) : value below the maximum of the input space where the last 
        potential testing point ought to go
    xmin_test (float) : minimum value within the input space below which there ought not to
        be testing points
    xmax_test (float) : maximum value within the input space above which there ought not to
        be testing points
    train_at_ends (bool) : whether training points should be allowed at or near the 
        endpoints of x
    test_at_ends (bool) : whether testing points should be allowed at or near the endpoints 
        of x
    """
    # gets information from the InterpObj
    x = interp_obj.x
    y = interp_obj.y
    kind_interp = interp_obj.kind
    f_interp = interp_obj.f_interp
    
    # creates initial sets of training and testing x points
    x_train = np.linspace(np.min(x) + offset_train_min, np.max(x) - offset_train_max, \
                          n_train + 1)
    x_test = np.linspace(np.min(x) + offset_test_min, np.max(x) - offset_test_max, \
                          n_train * n_test_inter + 1)
    
    # sets the xmin and xmax values to the minima and maxima, respectively, of the 
    # input space if no other value is given
    if xmin_train == None: xmin_train = np.min(x);
    if xmax_train == None: xmax_train = np.max(x);
    if xmin_test == None: xmin_test = np.min(x);
    if xmax_test == None: xmax_test = np.max(x);
        
    # eliminates, using a mask, all values for the training and testing x points outside of 
    # x
    x_train = x_train[ np.invert([(x_train[i] < np.min(x) or x_train[i] > np.max(x)) \
                        for i in range(len(x_train))]) ]
    x_test = x_test[ np.invert([(x_test[i] < np.min(x) or x_test[i] > np.max(x)) \
                        for i in range(len(x_test))]) ]
    
    # eliminates, using a mask, all values for the training and testing x points outside of 
    # the bounds specified by xmin and xmax
    x_train = x_train[ np.invert([(x_train[i] < xmin_train or x_train[i] > xmax_train) \
                        for i in range(len(x_train))]) ]
    x_test = x_test[ np.invert([(x_test[i] < xmin_test or x_test[i] > xmax_test) \
                        for i in range(len(x_test))]) ]
    
    # eliminates, using a mask, all values in the testing x points that are close enough
    # (within some tolerance) to any value in the training x points
    mask_filter_array = [[ np.isclose(x_test[i], x_train[j], \
                    atol = isclose_factor * (np.max(x) - np.min(x))) \
                    for i in range(len(x_test))] for j in range(len(x_train))]
    mask_filter_list = np.invert(np.sum(mask_filter_array, axis = 0, dtype = bool))
    x_test = x_test[mask_filter_list]
    
    # evaluates the interpolater at the training and testing x points
    y_train = f_interp(x_train)
    y_test = f_interp(x_test)
    
    # eliminates training and/or testing points if they lie at the edges of the input space
    if not train_at_ends:
        if np.isclose(x_train[0], x[0], atol = isclose_factor * (np.max(x) - np.min(x))):
            x_train = x_train[1:]
            if y_train.ndim == 3:
                y_train = y_train[:, :, 1:]
            elif y_train.ndim == 2:
                y_train = y_train[:, 1:]
        if np.isclose(x_train[-1], x[-1], atol = isclose_factor * (np.max(x) - np.min(x))):
            x_train = x_train[:-1]
            if y_train.ndim == 3:
                y_train = y_train[:, :, :-1]
            elif y_train.ndim == 2:
                y_train = y_train[:, :-1]
    if not test_at_ends:
        if np.isclose(x_test[0], x[0], atol = isclose_factor * (np.max(x) - np.min(x))):
            x_test = x_test[1:]
            if y_test.ndim == 3:
                y_test = y_test[:, :, 1:]
            elif y_test.ndim == 2:
                y_test = y_test[:, 1:]
        if np.isclose(x_test[-1], x[-1], atol = isclose_factor * (np.max(x) - np.min(x))):
            x_test = x_test[:-1]
            if y_test.ndim == 3:
                y_test = y_test[:, :, :-1]
            elif y_test.ndim == 2:
                y_test = y_test[:, :-1]

    return x_train, x_test, y_train, y_test
        
class InputSpaceBunch:
    """
    Class for an input space (i.e., x-coordinate)
    name (string) : (abbreviated) name for the input space
    input_space (array) : x-coordinate mesh points for evaluation
    mom (array) : momenta for the purpose of calculating the ratio
    caption (string) : caption for the x-axis of the coefficient plots for that input space
    tick_marks (array) : major tick marks for the x-axis of the coefficient plots
    title_pieces (array) : information to be concatenated into the coefficient plot's title
    """
    def __init__(self, name, input_space, mom, caption, title_pieces):
        self.name = name
        self.input_space = input_space
        # self.mom = mom * np.ones(len(input_space))
        self.mom = mom
        self.caption = caption
        self.title_pieces = title_pieces
        
    def make_title(self):
        """
        Concatenates the entries of title_pieces into a plot title
        """
        self.title = ''
        for piece in self.title_pieces: self.title += str(piece)
        return self.title

class ObservableBunch:
    """
    Class for an observable
    name (string) : (abbreviated) name for the observable
    data (array) : coefficient values at each order over the mesh
    energies (array) : energies at which the observable will be evaluated (None for observables
        plotted against energy)
    title (string) : title for the coefficient plot
    ref_type (string) : tells whether the reference scale (to be divided out of the coefficient
        values) has dimension (e.g., the case of the cross section) or not (e.g., the case of the 
    spin observables). Can only be "dimensionless" or "dimensionful".
    constraint (array or None): constraint on the values of the observable, including the 
        name of the quantity for which the constraint applies.
        For dimensionful (i.e., cross-section) observables, should be None.
    """
    def __init__(self, name, data, energies, angles, title, ref_type, constraint = None):
        self.name = name
        self.data = data
        self.energies = energies
        self.angles = angles
        self.title = title
        self.ref_type = ref_type
        self.constraint = constraint
        if (ref_type != "dimensionless") and (ref_type != "dimensionful"):
            raise Exception("ref_type must be dimensionless or dimensionful.")

class Interpolation:
    """
    Class for an interpolater
    x (array) : x-coordinate data
    y (array) : y-coordinate data
    kind (string) : scipy.interpolate.interp1d interpolater 'kind'
    """
    def __init__(self, x, y, kind = 'cubic'):
        self.x = x
        self.y = y
        self.kind = kind
        self.f_interp = interp1d(self.x, self.y, kind = self.kind)

class TrainTestSplit:
    """
    Class for an input space (i.e., x-coordinate)
    
    name (str) : (abbreviated) name for the combination of training and testing masks
    n_train (int) : number of intervals into which to split x, with training points at the 
        edges of each interval
    n_test_inter (int) : number of subintervals into which to split the intervals between 
        training points, with testing points at the edges of each subinterval
    isclose_factor (float) : fraction of the total input space for the tolerance of making
        sure that training and testing points don't coincide
    offset_train_min_factor (float) : fraction above the minimum of the input space where 
        the first potential training point ought to go
    offset_train_max_factor (float) : fraction below the maximum of the input space where 
        the last potential training point ought to go
    xmin_train_factor (float) : fraction of the input space below which there ought not to
        be training points
    xmax_train_factor (float) : fraction of the input space above which there ought not to
        be training points
    offset_test_min_factor (float) : fraction above the minimum of the input space where 
        the first potential testing point ought to go
    offset_test_max_factor (float) : fraction below the maximum of the input space where 
        the last potential testing point ought to go
    xmin_test_factor (float) : fraction of the input space below which there ought not to
        be testing points
    xmax_test_factor (float) : fraction of the input space above which there ought not to
        be testing points
    train_at_ends (bool) : whether training points should be allowed at or near the 
        endpoints of x
    test_at_ends (bool) : whether testing points should be allowed at or near the endpoints 
        of x
    """
    def __init__(self, name, n_train, n_test_inter, isclose_factor = 0.01, \
                offset_train_min_factor = 0, offset_train_max_factor = 0,\
                xmin_train_factor = 0, xmax_train_factor = 1, \
                offset_test_min_factor = 0, offset_test_max_factor = 0, \
                xmin_test_factor = 0, xmax_test_factor = 1, \
                train_at_ends = True, test_at_ends = False):
        self.name = name
        self.n_train = n_train
        self.n_test_inter = n_test_inter
        self.isclose_factor = isclose_factor
        self.offset_train_min_factor = offset_train_min_factor
        self.offset_train_max_factor = offset_train_max_factor
        self.xmin_train_factor = xmin_train_factor
        self.xmax_train_factor = xmax_train_factor
        self.offset_test_min_factor = offset_test_min_factor
        self.offset_test_max_factor = offset_test_max_factor
        self.xmin_test_factor = xmin_test_factor
        self.xmax_test_factor = xmax_test_factor
        self.train_at_ends = train_at_ends
        self.test_at_ends = test_at_ends
        
    def make_masks(self, x, y):
        """Returns the training and testing points in the input space and the corresponding 
        (interpolated) data values after calculating the actual values for xmin, xmax, and 
        offsets using the corresponding factors and the input space

        Parameters
        ----------
        x (1D array) : input space
        y (ND array) : data points at each input space value, with N>1 dimensions for N 
            orders
        """
        self.x = x
        # print(self.x)
        self.y = y
        
        # calculates the actual value for each offset, xmin, and xmax
        self.offset_train_min = self.offset_train_min_factor \
                                    * (np.max(self.x) - np.min(self.x))
        self.offset_train_max = self.offset_train_max_factor \
                                    * (np.max(self.x) - np.min(self.x))
        self.xmin_train = np.min(self.x) + self.xmin_train_factor * \
                            (np.max(self.x) - np.min(self.x))
        self.xmax_train = np.min(self.x) + self.xmax_train_factor * \
                            (np.max(self.x) - np.min(self.x))
        self.offset_test_min = self.offset_test_min_factor \
                                    * (np.max(self.x) - np.min(self.x))
        self.offset_test_max = self.offset_test_max_factor \
                                    * (np.max(self.x) - np.min(self.x))
        self.xmin_test = np.min(self.x) + self.xmin_test_factor * \
                            (np.max(self.x) - np.min(self.x))
        self.xmax_test = np.min(self.x) + self.xmax_test_factor * \
                            (np.max(self.x) - np.min(self.x))
        
        self.interp_obj = Interpolation(self.x, self.y, kind = 'cubic')
        
        # creates the x and y training and testing points
        self.x_train, self.x_test, self.y_train, self.y_test = \
            versatile_train_test_split(self.interp_obj, \
                self.n_train, n_test_inter = self.n_test_inter, \
                isclose_factor = self.isclose_factor, \
                offset_train_min = self.offset_train_min, \
                offset_train_max = self.offset_train_max, \
                xmin_train = self.xmin_train, xmax_train = self.xmax_train, \
                offset_test_min = self.offset_test_min, \
                offset_test_max = self.offset_test_max, \
                xmin_test = self.xmin_test, xmax_test = self.xmax_test, \
                train_at_ends = self.train_at_ends, test_at_ends = self.test_at_ends)
        
        return self.x_train, self.x_test, self.y_train, self.y_test

class ScaleSchemeBunch:
    def __init__(self, file_name, orders_full, cmaps, potential_string, cutoff_string, 
                 dir_path = "./"):
        self.file_name = file_name
        self.orders_full  = orders_full
        self.cmaps = cmaps
        self.potential_string = potential_string
        self.cutoff_string = cutoff_string
        self.name = self.potential_string + self.cutoff_string
        self.dir_path = dir_path
        
        self.full_path = self.dir_path + self.file_name
        
        self.colors = [cmap(0.55 - 0.1 * (i==0)) for i, cmap in enumerate(self.cmaps)]
        self.light_colors = [cmap(0.35) for cmap in self.cmaps]

    def get_data(self, observable_string):
        response = h5py.File(self.full_path, "r")
        obs_data = np.array(response[observable_string][:])
        response.close()
        return obs_data

class LengthScale:
    """
    Class for setting a guess for the Gaussian process correlation length scale and its
    bounds
    x (array) : x-coordinate data
    ls_guess_factor (float) : fraction of the total input space length for the initial
        length scale guess
    ls_bound_lower_factor (float) : fraction of the initial length scale guess for the lower
        bound of fitting
    ls_bound_upper_factor (float) : fraction of the initial length scale guess for the upper
        bound of fitting
    whether_fit (bool) : should the fit procedure be performed?
    """
    def __init__(self, name, ls_guess_factor, ls_bound_lower_factor, 
                 ls_bound_upper_factor, whether_fit = True):
        self.name = name
        self.ls_guess_factor = ls_guess_factor
        self.ls_bound_lower_factor = ls_bound_lower_factor
        self.ls_bound_upper_factor = ls_bound_upper_factor
        self.whether_fit = whether_fit
    
    def make_guess(self, x):
        self.ls_guess = (np.max(x) - np.min(x)) * self.ls_guess_factor
        
        if self.whether_fit:
            self.ls_bound_lower = self.ls_bound_lower_factor * self.ls_guess
            self.ls_bound_upper = self.ls_bound_upper_factor * self.ls_guess
        else:
            self.ls_bound_lower = self.ls_guess.copy()
            self.ls_bound_upper = self.ls_guess.copy()

class GSUMDiagnostics:
    def __init__(self, observable, Lambda_b, inputspace, traintestsplit, 
                 gphyperparameters, orderinfo, filenaming, 
                 fixed_quantity = [None, None, None], 
                 x_quantity = [None, None], posteriorgrid = None):
        """
        Class for everything involving Jordan Melendez's GSUM library for observables that 
        can be plotted against angle.
        observable (ObservableBunch) : observable being plotted
        Lambda_b (float) : cutoff (MeV)
        inputspace (InputSpaceBunch) : input space against which the observable is plotted
        traintestsplit (TrainTestSplit) : training and testing masks
        gphyperparameters (GPHyperparameters) : parameters for fitted Gaussian process
        orderinfo (OrderInfo) : information about the EFT orders and their colors 
        filenaming (FileNaming) : strings for naming the save files
        E_lab (float) : lab energy (MeV) at which to evaluate the observable
        E_lab_x (float array) : lab-energy (MeV) x-coordinate mesh over which the GP is calculated, plotted, and fitted
        posteriorgrid (PosteriorBounds) : xy-grid over which to plot the Lambda-ell posterior pdf
        """
        # information on the observable
        self.observable = observable
        self.observable_name = self.observable.name
        self.data = self.observable.data
        self.ref_type = self.observable.ref_type
        self.constraint = self.observable.constraint

        # cutoff scale
        self.Lambda_b = Lambda_b

        # energy or angle at which the observable is evaluated, along with all 
        # possible energies or angles for evaluation
        self.fixed_quantity_name = fixed_quantity[0]
        self.fixed_quantity_value = fixed_quantity[1]
        self.fixed_quantity_array = fixed_quantity[2]
        self.fixed_quantity_units = fixed_quantity[3]

        # angle or energy mesh
        self.x_quantity_name = x_quantity[0]
        self.x_quantity_array = x_quantity[1]
        # print(self.x_quantity_array)
        self.x_quantity_units = x_quantity[2]
        
        # information on the input space
        self.inputspace = inputspace
        self.vs_what = self.inputspace.name
        self.x = self.inputspace.input_space(**{"deg_input" : self.x_quantity_array, 
                                                "p_input" : E_to_p(self.fixed_quantity_value, "np"),
                                                "E_lab" : self.x_quantity_array})
        self.X = self.x[:, None]
        self.caption_coeffs = self.inputspace.caption
        self.title_coeffs = self.inputspace.title

        # information on the train/test split
        self.traintestsplit = traintestsplit
        self.train_pts_loc = self.traintestsplit.name
        self.x_train = self.traintestsplit.x_train
        self.n_train_pts = len(self.x_train)
        self.x_test = self.traintestsplit.x_test
        self.n_test_pts = len(self.x_test)
        self.y_train = self.traintestsplit.y_train
        self.y_test = self.traintestsplit.y_test

        # information on the GP hyperparameters
        self.gphyperparameters = gphyperparameters
        self.ls = self.gphyperparameters.ls
        self.ls_lower = self.gphyperparameters.ls_lower
        self.ls_upper = self.gphyperparameters.ls_upper
        self.whether_fit = self.gphyperparameters.whether_fit
        self.center = self.gphyperparameters.center
        self.ratio = self.gphyperparameters.ratio
        self.nugget = self.gphyperparameters.nugget
        self.seed = self.gphyperparameters.seed
        self.df = self.gphyperparameters.df
        self.disp = self.gphyperparameters.disp
        self.std_est = self.gphyperparameters.scale
        self.sd = self.gphyperparameters.sd

        # information on the orders at which the potential is evaluated
        self.orderinfo = orderinfo
        self.nn_orders_full = self.orderinfo.orders_full
        self.nn_orders = self.orderinfo.orders_full
        self.nn_orders_mask = self.orderinfo.mask_full
        # print(self.nn_orders_mask)
        self.colors = self.orderinfo.colors_array
        self.light_colors = self.orderinfo.lightcolors_array
        self.orders_restricted = self.orderinfo.orders_restricted
        # print(self.orders_restricted)
        self.mask_restricted = self.orderinfo.mask_restricted
        # print(self.mask_restricted)
        # self.raw_data_mask = np.array([*[True], *(self.nn_orders_mask[1:])[self.mask_restricted]])
        self.raw_data_mask = np.array([*[True], *(self.mask_restricted)])
        if self.orderinfo.orders_names_dict is None:
            self.orders_names_dict = {
                6: "N4LO+",
                5: "N4LO",
                4: "N3LO",
                3: "N2LO",
                2: "NLO",
            }
        else:
            self.orders_names_dict = self.orderinfo.orders_names_dict
        if self.orderinfo.orders_labels_dict is None:
            self.orders_labels_dict = {6 : r'N$^{4}$LO$^{+}$', 5 : r'N$^{4}$LO', 
                                4 : r'N$^{3}$LO', 3 : r'N$^{2}$LO',
                                2 : r'NLO'}
        else:
            self.orders_labels_dict = self.orderinfo.orders_labels_dict
                                       
        # information for naming the file
        self.filenaming = filenaming
        self.scheme = self.filenaming.scheme
        self.scale = self.filenaming.scale
        self.Q_param = self.filenaming.Q_param
        self.p_param = self.filenaming.p_param
        self.filename_addendum = self.filenaming.filename_addendum
        
        # posterior pdf bounds
        self.posteriorgrid = posteriorgrid
        
        # for plotting observables at a fixed energy
        if self.fixed_quantity_name == "energy":
            self.fixed_idx = np.nonzero(self.fixed_quantity_array == self.fixed_quantity_value)[0][0]
            
            self.data = self.data[:, self.fixed_idx, :].T
            
            self.X_train = self.x_train[:, None]
            self.y_train = self.y_train[:, self.fixed_idx, :].T
            self.X_test = self.x_test[:, None]
            self.y_test = self.y_test[:, self.fixed_idx, :].T
            
            # determines the reference scale for the truncation-error model, including for 
            # training and testing
            if self.ref_type == "dimensionless":
                self.ref = np.ones(len(self.x)) * 1
                self.ref_train = np.ones(len(self.x_train)) * 1
                self.ref_test = np.ones(len(self.x_test)) * 1
                # if self.observable_name == "AY":
                #     self.ref = np.sin(np.radians(self.x_quantity_array))
                #     self.interp_f_ref = interp1d(self.x, self.ref)
                #     print("Radians are " + str(np.radians(self.x_quantity_array)))
                #     print("AY's ref is " + str(self.ref))
                #     self.ref_train = self.interp_f_ref(self.x_train)
                #     self.ref_test = self.interp_f_ref(self.x_test)
                # elif self.observable_name == "A":
                #     self.ref = np.sin(np.radians(self.x_quantity_array) / 2)
                #     self.interp_f_ref = interp1d(self.x, self.ref)
                #     print("Radians are " + str(np.radians(self.x_quantity_array)))
                #     print("A's ref is " + str(self.ref))
                #     self.ref_train = self.interp_f_ref(self.x_train)
                #     self.ref_test = self.interp_f_ref(self.x_test)
                # else:
                # 	self.ref = np.ones(len(self.x)) * 1
                # 	self.ref_train = np.ones(len(self.x_train)) * 1
                # 	self.ref_test = np.ones(len(self.x_test)) * 1
            elif self.ref_type == "dimensionful":
                self.ref = self.data[:, -1]
                    
                self.interp_f_ref = interp1d(self.x, self.ref)
                self.ref_train = self.interp_f_ref(self.x_train)
                self.ref_test = self.interp_f_ref(self.x_test)
               
        # for plotting observables at a fixed angle
        elif self.fixed_quantity_name == "angle":
            if self.fixed_quantity_value == 0:
                self.X_train = self.x_train[:, None]
                self.y_train = self.y_train.T
                self.X_test = self.x_test[:, None]
                self.y_test = self.y_test.T
            else:
                self.fixed_idx = np.nonzero(self.fixed_quantity_array == self.fixed_quantity_value)[0][0]
                
                self.data = self.data[:, :, self.fixed_idx].T
                
                self.X_train = self.x_train[:, None]
                self.y_train = self.y_train[:, self.fixed_idx, :].T
                self.X_test = self.x_test[:, None]
                self.y_test = self.y_test[:, self.fixed_idx, :].T
            
            # determines the reference scale for the truncation-error model, including for 
            # training and testing
            if self.ref_type == "dimensionless":
            	self.ref = np.ones(len(self.x)) * 1
            	self.ref_train = np.ones(len(self.x_train)) * 1
            	self.ref_test = np.ones(len(self.x_test)) * 1
            elif self.ref_type == "dimensionful":
                if self.fixed_quantity_value == 0:
                    self.ref = self.data[-1]
                    self.data = self.data.T
                else:
                    self.ref = self.data[:, -1]
                
                self.interp_f_ref = interp1d(self.x, self.ref)
                self.ref_train = self.interp_f_ref(self.x_train)
                self.ref_test = self.interp_f_ref(self.x_test)

        # uses interpolation to find the proper reference scales
        self.interp_f_ref = interp1d(self.x, self.ref)

        # Extract the coefficients and define kernel
        self.coeffs = gm.coefficients(self.data, ratio = self.ratio, 
                            ref = self.ref, orders = self.nn_orders)[:, self.nn_orders_mask]

        # uses interpolation to find the proper ratios for training and testing
        self.interp_f_ratio = interp1d(self.x, self.ratio * np.ones(len(self.x)))
        self.ratio_train = self.interp_f_ratio(self.x_train)
        self.coeffs_train = gm.coefficients(self.y_train, ratio = self.ratio_train, 
                                          ref = self.ref_train, 
                                          orders = self.nn_orders)[:, self.nn_orders_mask]
        self.ratio_test = self.interp_f_ratio(self.x_test)
        self.coeffs_test = gm.coefficients(self.y_test, ratio = self.ratio_test, 
                                          ref = self.ref_test, 
                                          orders = self.nn_orders)[:, self.nn_orders_mask]

        # defines the kernel
        if self.fixed_quantity_name == "energy" and \
                self.fixed_quantity_value < 70.1 and \
                self.fixed_quantity_value >= 1.:
            self.kernel = RBF(length_scale = self.ls, 
                        length_scale_bounds = (self.ls_lower, self.ls_upper)) + \
                        WhiteKernel(1e-6, noise_level_bounds = 'fixed')
        else:
            self.kernel = RBF(length_scale = self.ls, \
                        length_scale_bounds = (self.ls_lower, self.ls_upper)) + \
                        WhiteKernel(1e-10, noise_level_bounds = 'fixed')
        # print(self.kernel)

        # Define the GP
        self.gp = gm.ConjugateGaussianProcess(
            self.kernel, center = self.center, disp = self.disp, df = self.df,
            scale = self.std_est, n_restarts_optimizer = 50, random_state = self.seed, 
            sd = self.sd)
        
        # restricts coeffs and colors to only those orders desired for 
        # evaluating statistical diagnostics
        self.nn_orders = self.orders_restricted
        self.colors = list(np.array(self.colors)[self.mask_restricted])
        self.light_colors = list(np.array(self.light_colors)[self.mask_restricted])
        self.coeffs = (self.coeffs.T[self.mask_restricted]).T
        self.coeffs_train = (self.coeffs_train.T[self.mask_restricted]).T
        self.coeffs_test = (self.coeffs_test.T[self.mask_restricted]).T

    def plot_coefficients(self, ax = None, whether_save = True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        # optimizes the ConjugateGaussianProcess for the given parameters and extracts the 
        # length scale
        self.gp.fit(self.X_train, self.coeffs_train)
        self.ls_true = np.exp(self.gp.kernel_.theta)
        # X_constraint = [self.x[i] for i in self.constraint[0]]
        # mask_constraint = np.reshape(~ np.isin(self.X_train, X_constraint), len(self.X_train))
        # print(mask_constraint.shape)
        # print("mask_constraint = " + str(mask_constraint))
        # self.pred, self.std = self.gp.predict(self.X, 
        #                                 Xc = self.X_train[mask_constraint], 
        #                                 y = self.coeffs_train[mask_constraint, :], 
        #                                 return_std = True)
        self.pred, self.std = self.gp.predict(self.X, return_std = True)
        self.underlying_std = np.sqrt(self.gp.cov_factor_)

        # plots the coefficients against the given input space
        if ax is None:
            # fig, ax = plt.subplots(figsize=(3.2, 3.2))
            fig, ax = plt.subplots(figsize=(2.1, 2.1))
        
        for i, n in enumerate((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]):
            ax.fill_between(self.x, self.pred[:, i] + 2*self.std, 
                            self.pred[:, i] - 2*self.std, 
                            facecolor = self.light_colors[i], edgecolor = self.colors[i], 
                            lw = edgewidth, alpha=1, zorder = 5 * i - 4)
            ax.plot(self.x, self.pred[:, i], color = self.colors[i], ls='--', zorder = 5 * i - 3)
            ax.plot(self.x, self.coeffs[:, i], color = self.colors[i], zorder = 5 * i - 2)
            ax.plot(self.x_train, self.coeffs_train[:, i], color = self.colors[i], \
                    ls='', marker='o', label=r'$c_{}$'.format(n), zorder = 5 * i - 1)

        # Format
        ax.axhline(2*self.underlying_std, 0, 1, color = gray, zorder=-10, lw=1)
        ax.axhline(-2*self.underlying_std, 0, 1, color = gray, zorder=-10, lw=1)
        ax.axhline(0, 0, 1, color = softblack, zorder=-10, lw=1)
        if np.max(self.x) < 1.1:
            ax.set_xticks(self.x_test, minor=True)
            ax.set_xticks([round(xx, 1) for xx in self.x_train])
        else:
            ax.set_xticks(self.x_test, minor=True)
            ax.set_xticks([round(xx, 0) for xx in self.x_train])
        ax.tick_params(which='minor', bottom=True, top=False)
        # ax.set_xlabel(self.caption_coeffs)
        ax.set_yticks(ticks = [-2*self.underlying_std, 2*self.underlying_std])
        ax.set_yticklabels(labels = ['{:.1f}'.format(-2*self.underlying_std), '{:.1f}'.format(2*self.underlying_std)])
        ax.set_yticks([-1 * self.underlying_std, self.underlying_std], minor = True)
        # ax.legend(ncol=2, borderpad=0.4,# labelspacing=0.5, columnspacing=1.3,
        #           borderaxespad=0.6, loc = 'upper right',
        #           title = self.title_coeffs).set_zorder(5 * i)
        
        if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
            dX = np.array([[self.x[i]] for i in self.constraint[0]])
            # std_interp = np.sqrt(np.diag(
            #     self.gp.cov(self.X) -
            #     self.gp.cov(self.X, dX) @ np.linalg.solve(self.gp.cov(dX, dX), self.gp.cov(dX, self.X))
            # ))
            _, std_interp = self.gp.predict(self.X, 
                                            Xc = dX, 
                                            y = np.array(self.constraint[1]),
                                            return_std = True)
            # print("We calculated std_interp.")
            ax.plot(self.x, 2*std_interp, color='gray', ls='--', zorder=-10, lw=1)
            ax.plot(self.x, -2*std_interp, color='gray', ls='--', zorder=-10, lw=1)
        
        # draws length scales
        ax.annotate("", xy=(np.min(self.x), -0.65*2*self.underlying_std), 
                    xytext=(np.min(self.x) + self.ls, -0.65*2*self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + self.ls + 0.2 * (np.max(self.x) - np.min(self.x)), 
                -0.65*2*self.underlying_std, r'$\ell_{\mathrm{guess}}$', fontsize = 14, 
                horizontalalignment='right', verticalalignment='center', zorder = 5 * i)

        ax.annotate("", xy=(np.min(self.x), -0.9*2*self.underlying_std), \
                    xytext=(np.min(self.x) + self.ls_true, -0.9*2*self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + self.ls_true + 0.2 * (np.max(self.x) - np.min(self.x)), 
                -0.9*2*self.underlying_std, r'$\ell_{\mathrm{fit}}$', fontsize = 14, 
                horizontalalignment='right', verticalalignment='center', zorder = 5 * i)
        
        # draws standard deviations
        ax.annotate("", xy=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), 0), \
                    xytext=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), \
                            -1. * self.std_est),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), 
                -1.2 * self.std_est, r'$\sigma_{\mathrm{guess}}$', fontsize = 14, 
                horizontalalignment='center', verticalalignment='bottom', zorder = 5 * i)
            
        ax.annotate("", xy=(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)), 0), \
                    xytext=(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)), \
                            -1. * self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + 0.74 * (np.max(self.x) - np.min(self.x)), 
                -1.2 * self.underlying_std, r'$\sigma_{\mathrm{fit}}$', fontsize= 14, 
                horizontalalignment='center', verticalalignment='bottom', zorder = 5 * i)
        
        if 'fig' in locals() and whether_save:
            fig.tight_layout()
    
            fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                    '_' + 'interp_and_underlying_processes' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + \
                    '_' + self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                    '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                    self.train_pts_loc + '_' + self.p_param + 
                    self.filename_addendum).replace('_0MeVlab_', '_'))

    def plot_md(self, ax = None, whether_save = True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots the squared Mahalanobis distance
            self.gp.kernel_
            
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test, 
                                                Xc = dX, 
                                                y = np.array(self.constraint[1]),
                                                return_std = False, 
                                                return_cov = True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, 
                                            self.mean, 
                                            self.cov, 
                                            colors = self.colors, 
                                            gray = gray, 
                                            black = softblack)
    
            if ax is None:
                # fig, ax = plt.subplots(figsize=(1, 3.2))
                fig, ax = plt.subplots(figsize=(0.7, 4.2))
                
            self.gr_dgn.md_squared(type = 'box', trim = False, title = None, 
                            xlabel=r'$\mathrm{D}_{\mathrm{MD}}^2$', ax = ax, 
                            **{"size" : 10})
            offset_xlabel(ax)
            # ax.set_ylim(0, 100)
            
            if 'fig' in locals() and whether_save:
                fig.tight_layout();
            
                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                        '_' + 'md' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + '_' + \
                        self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))
            
        except:
            print("Error in calculating or plotting the Mahalanobis distance.")

    def plot_pc(self, ax = None, whether_save = True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots the pivoted Cholesky decomposition
            self.gp.kernel_
            
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test, 
                                                Xc = dX, 
                                                y = np.array(self.constraint[1]),
                                                return_std = False, 
                                                return_cov = True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, 
                                                     self.mean, 
                                                     self.cov, 
                                                     colors = self.colors, 
                                                     gray = gray, 
                                                     black = softblack)

            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                if ax is None:
                    # fig, ax = plt.subplots(figsize=(3.2, 3.2))
                    fig, ax = plt.subplots(figsize=(2.1, 2.1))
                    
                self.gr_dgn.pivoted_cholesky_errors(ax = ax, title = None)
                ax.set_xticks(np.arange(2, self.n_test_pts + 1, 2))
                ax.set_xticks(np.arange(1, self.n_test_pts + 1, 2), minor = True)
                ax.text(0.05, 0.95, r'$\mathrm{D}_{\mathrm{PC}}$', bbox = text_bbox, 
                        transform = ax.transAxes, va='top', ha='left')
                ax.set_ylim(-6, 6)
                
                if 'fig' in locals() and whether_save:
                    fig.tight_layout()
                    
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                            '_' + 'pc_vs_index' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + '_' + \
                            self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                            '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                            self.train_pts_loc + '_' + self.p_param + 
                            self.filename_addendum).replace('_0MeVlab_', '_'))
    
        except:
            print("Error in calculating or plotting the pivoted Cholesky decomposition.")

    def plot_posterior_pdf(self, ax_joint = None, ax_marg_x = None, 
                         ax_marg_y = None, whether_save = True):
        """
        Parameters
        ----------
        ax_joint : Axes, optional
            Joint axis. The default is None.
        ax_marg_x : Axes, optional
            Axis for marginalizing the y-coordinate. The default is None.
        ax_marg_y : Axes, optional
            Axis for marginalizing the x-coordinate. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        # functions for interpolating the ratio and reference scale in the TruncationGP
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)
        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var
        try:
            # creates the grid over which the posterior PDF will be plotted
            self.ls_vals = self.posteriorgrid.x_vals
            self.lambda_vals = self.posteriorgrid.y_vals
            
            # creates and fits the TruncationGP
            self.gp_post = gm.TruncationGP(self.kernel, 
                                           ref = lambda_interp_f_ref, 
                                        ratio = lambda_interp_f_ratio, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            
            # print("y_train has dimensions " + str(self.y_train.shape))
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                self.gp_post.fit(self.X_train, 
                                 self.y_train, 
                                 orders = self.nn_orders_full,
                                 orders_eval = self.nn_orders, 
                                 dX = np.array([[self.x[i]] for i in self.constraint[0]]), 
                                 dy = [j for j in self.constraint[1]])
            else:
                self.gp_post.fit(self.X_train, 
                                 self.y_train, 
                                 orders = self.nn_orders_full, 
                                 orders_eval = self.nn_orders)
            
            # evaluates the probability across the mesh
            self.ls_lambda_loglike = np.array([[
                self.gp_post.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders,
                                                     **{"lambda_var" : lambda_})
                    for ls_ in np.log(self.ls_vals)]
                    for lambda_ in self.lambda_vals])
    
            # Makes sure that the values don't get too big or too small
            self.ls_lambda_like = np.exp(self.ls_lambda_loglike - np.max(self.ls_lambda_loglike))
    
            # Now compute the marginal distributions
            self.lambda_like = np.trapz(self.ls_lambda_like, x = self.ls_vals, axis = -1)
            self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
    
            # Normalize them
            self.lambda_like /= np.trapz(self.lambda_like, x = self.lambda_vals, axis = 0)
            self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
            
            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                cmap_name = 'Blues'
                cmap = mpl.cm.get_cmap(cmap_name)
    
                # Setup axes
                if ax_joint == None and ax_marg_x == None and ax_marg_y == None:
                    fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(ratio=5, height=3.4)
    
                # Plot contour
                ax_joint.contour(self.ls_vals, self.lambda_vals, self.ls_lambda_like,
                                  levels=[np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999],
                                  cmap=cmap_name, vmin=-0.05, vmax=0.8, zorder=1)
    
                # Now plot the marginal distributions
                ax_marg_y.plot(self.lambda_like, self.lambda_vals, c=cmap(0.8), lw=1)
                ax_marg_y.fill_betweenx(self.lambda_vals, np.zeros_like(self.lambda_like),
                                        self.lambda_like, facecolor=cmap(0.2), lw=1)
                ax_marg_x.plot(self.ls_vals, self.ls_like, c=cmap(0.8), lw=1)
                ax_marg_x.fill_between(self.ls_vals, np.zeros_like(self.ls_vals),
                                        self.ls_like, facecolor=cmap(0.2), lw=1)
    
                # Formatting
                ax_joint.set_xlabel(r'$\ell$')
                ax_joint.set_ylabel(r'$\Lambda$')
                ax_joint.axvline(self.ls, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.axhline(self.Lambda_b, 0, 1, c=gray, lw=1, zorder=0)
                ax_joint.margins(x=0, y=0.)
                ax_joint.set_xlim(min(self.ls_vals), max(self.ls_vals))
                ax_joint.set_ylim(min(self.lambda_vals), max(self.lambda_vals))
                ax_marg_x.set_ylim(bottom=0);
                ax_marg_y.set_xlim(left=0);
                ax_joint.text(0.95, 0.95, r'pr$(\ell, \Lambda \,|\, \vec{\mathbf{y}}_k)$', ha='right', va='top',
                              transform=ax_joint.transAxes, bbox=text_bbox, fontsize = 12)
    
                plt.show()
                
                if 'fig' in locals() and whether_save:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                                '_' + 'Lambda_ell_jointplot' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + '_' + \
                                self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                                '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                                self.train_pts_loc + '_' + self.p_param + 
                                self.filename_addendum).replace('_0MeVlab_', '_'))
        
        except:
            print("Error in plotting the posterior PDF.")

    def plot_truncation_errors(self, online_data, residual_plot = True,
                             whether_save = True):
        """
        Parameters
        ----------
        online_data : array
            Summed order-by-order predictions for an observable.
        constraint : array of arrays, optional
            The constraints for the fit procedure. The default is None.
        residual_plot : bool, optional
            Whether to plot the truncation error as residuals or as full sums.
        whether_save : bool, optional
            Whether to save the output. The default is True.

        Returns
        -------
        Figure (2)
            A figure with the order-by-order residuals plot and a figure with 
            the order-by-order empirical coverage.
        """
        # sets up the data from PWA93 to which we'll compare
        self.online_data = online_data
        
        # functions for reference scale and dimensionless expansion parameter (ratio)
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)
        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var
        
        try:
            self.gp_trunc = gm.TruncationGP(self.kernel, 
                        ref = lambda_interp_f_ref, 
                        ratio = lambda_interp_f_ratio, 
                        center = self.center, 
                        disp = self.disp, 
                        df = self.df, 
                        scale = self.std_est, 
                        excluded = [0], 
                        ratio_kws = {"lambda_var" : self.Lambda_b})
            
            # fits the GP with or without a constraint
            # print("constraint is " + str(self.constraint))
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                # print("We got this far.")
                self.gp_trunc.fit(self.X_train, self.y_train, 
                        orders = self.nn_orders_full,
                        orders_eval = self.nn_orders, 
                        dX = np.array([[self.x[i]] for i in self.constraint[0]]), 
                        dy = [j for j in self.constraint[1]])
            else:
                self.gp_trunc.fit(self.X_train, self.y_train, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders)
            
            
            # creates fig with two columns of axes
            fig, axes = plt.subplots(int(np.ceil(len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]) / 2)),
                    2, sharex=False, sharey=False, figsize=(6, 7))
            # deletes extraneous axes to suit number of evaluated orders
            if 2 * np.ceil(len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]) / 2) > len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]):
                fig.delaxes(axes[int(np.ceil(len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]) / 2)) - 1, 1])
            
            for i, n in enumerate((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]):
                # calculates the standard deviation of the truncation error
                _, self.std_trunc = self.gp_trunc.predict(self.X, order = n, 
                                        return_std = True, kind = 'trunc')
                
                # gets the "true" order-by-order data from online
                if self.fixed_quantity_name == "energy":
                    data_true = self.online_data[self.fixed_quantity_value, :]
                elif self.fixed_quantity_name == "angle":
                    if self.fixed_quantity_value == 0:
                        data_true = self.online_data
                    else:
                        data_true = self.online_data[:, self.fixed_quantity_value]
                        
                for j in range(i, len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted])):
                    ax = axes.ravel()[j]
                    
                    # number of standard deviations around the dotted line to plot
                    std_coverage = 0.64
                    
                    if residual_plot:
                        # calculates and plots the residuals
                        residual = data_true - ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i]
                        ax.plot(self.x, residual, zorder=i-5, c = self.colors[i])
                        ax.fill_between(self.x, 
                                    residual + std_coverage*self.std_trunc, 
                                    residual - std_coverage*self.std_trunc, 
                                    zorder=i-5, 
                                    facecolor = self.light_colors[i], 
                                    edgecolor = self.colors[i], 
                                    lw=edgewidth)
                        ax.set_ylim(np.min(np.concatenate((residual + std_coverage*self.std_trunc, residual - std_coverage*self.std_trunc))), 
                                np.max(np.concatenate((residual + std_coverage*self.std_trunc, residual - std_coverage*self.std_trunc))))
                
                    else:
                        # calculates and plots the true data
                        ax.plot(self.x, 
                                ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i], 
                                zorder=i-5, c = self.colors[i])
                        ax.fill_between(self.x, 
                                    ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] + std_coverage*self.std_trunc, 
                                    ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] - std_coverage*self.std_trunc, 
                                    zorder=i-5, 
                                    facecolor = self.light_colors[i], 
                                    edgecolor = self.colors[i], 
                                    lw=edgewidth)
                        ax.set_ylim(np.min(np.concatenate((((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] + std_coverage*self.std_trunc, ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] - std_coverage*self.std_trunc))), 
                                np.max(np.concatenate((((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] + std_coverage*self.std_trunc, ((self.data[:, self.nn_orders_mask])[:, self.mask_restricted])[:, i] - std_coverage*self.std_trunc))))
                    
                    # # plots the testing points as vertical lines
                    # for line in self.x_test: ax.axvline(line, 0, 1, c = gray)
                    
                ax = axes.ravel()[i]
                
                if residual_plot:
                    # plots a line at y = 0
                    ax.plot(self.x, np.zeros(len(self.x)), color=softblack, lw=1, ls='--')
                else:
                    # plots the true data
                    ax.plot(self.x, data_true, color=softblack, lw=1, ls='--')
                
                # formats x-axis labels and tick marks
                ax.set_xlabel(self.caption_coeffs)
                ax.set_xticks([int(min(self.x) + (max(self.x) - min(self.x)) / 3), 
                                int(min(self.x) + (max(self.x) - min(self.x)) / 3 * 2)])
                ax.set_xticks([tick for tick in self.x_test], minor = True)
            
            # saves
            if 'fig' in locals() and whether_save:
                fig.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(self.fixed_quantity_units) + ')\,' + \
                                '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                                '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size = 20)
                fig.tight_layout()
                
                if self.constraint is None:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' + 
                        str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + 
                        '_' + 'full_pred_truncation' + '_' + self.scheme + '_' + 
                            self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))
                else:
                    fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' + 
                        str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + 
                        '_' + 'full_pred_truncation_constrained' + '_' + self.scheme + '_' + 
                            self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))
            
            # creates interpolation function for the true and theory data
            data_interp = interp1d(self.x, (self.data[:, self.nn_orders_mask])[:, self.mask_restricted].T)
            data_true_interp = interp1d(self.x, data_true)
            
            # calculates the covariance matrix and mean
            self.cov_wp = self.gp_trunc.cov(self.X_test, start = 0, end = 0)
            # print(self.cov_wp)
            self.mean_wp = self.gp_trunc.mean(self.X_test)
            
            
            # norms the residuals by factors of the ratio
            self.norm_residuals_wp = data_true_interp(self.X_test) - data_interp(self.X_test)
            denom = (np.tile(self.ratio_test, (len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]), 1)).T)**((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted] + 1) * (np.sqrt(1 - np.tile(self.ratio_test, (len((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted]), 1))**2)).T
            self.norm_residuals_wp = self.norm_residuals_wp / (denom.T)[:, :, None]
            self.gr_dgn_wp = gm.GraphicalDiagnostic(self.norm_residuals_wp.T, 
                                mean=self.mean_wp, cov=self.cov_wp, 
                                colors=self.colors, gray=gray, black=softblack)
    
            
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
    
            # creates the empirical coverage plot
            self.gr_dgn_wp.credible_interval(
                np.linspace(1e-5, 1, 100), band_perc=[0.68, 0.95], ax = ax, 
                title = "Empirical coverage (PWA93)\n" + 
                        r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(self.fixed_quantity_units) + ')\,' + \
                                            '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                                            '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', 
                xlabel = r'Credible Interval ($100\alpha\%$)', 
                ylabel = r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')
    
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])
            
            if 'fig' in locals() and whether_save:
                fig.tight_layout()
        
                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' + 
                        str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + \
                        '_' + 'truncation_error_empirical_coverage' + '_' + self.scheme + '_' + \
                            self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                        self.train_pts_loc + '_' + self.p_param +  
                        self.filename_addendum).replace('_0MeVlab_', '_'))
        
        except:
            print("Error plotting the truncation errors.")
            
    def plot_credible_intervals(self, ax = None, whether_save = True):
        """
        Parameters
        ----------
        ax : Axes, optional
            Axes object for plotting. The default is None.
        whether_save : bool, optional
            Whether to save the figure. The default is True.

        Returns
        -------
        Figure with plot.
        """
        try:
            # calculates and plots credible intervals ("weather plots")
            self.gp.kernel_
            
            if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
                dX = np.array([[self.x[i]] for i in self.constraint[0]])
                self.mean, self.cov = self.gp.predict(self.X_test, 
                                                Xc = dX, 
                                                y = np.array(self.constraint[1]),
                                                return_std = False, 
                                                return_cov = True)
            else:
                self.mean = self.gp.mean(self.X_test)
                self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, 
                                                 self.mean, 
                                                 self.cov, 
                                                 colors = self.colors, 
                                                 gray = gray, 
                                                 black=softblack)
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(3.4, 3.2))
    
            self.gr_dgn.credible_interval(
                np.linspace(1e-5, 1, 100), band_perc=[0.68, 0.95], ax = ax, title = None, \
                xlabel = r'Credible Interval ($100\alpha\%$)', \
                ylabel = r'Empirical Coverage ($\%$)\,(N = ' + str(len(self.X_test)) + r')')
    
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])
            
            if 'fig' in locals() and whether_save:
                fig.tight_layout()
        
                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + '_' + 
                        str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + \
                        '_' + 'truncation_error_credible_intervals' + '_' + self.scheme + '_' + \
                            self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))
        
        except:
            print("Error in plotting the credible intervals.")
    
    def plot_lambda_posterior_pointwise(self, SGT, DSG, AY, A, D, AXX, AYY, t_lab, degrees, 
                            ax = None, whether_save = True):
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)
        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var
        
        # try:
        # t_lab_Lb = np.array([100, 250])
        t_lab_Lb = np.array([50, 100, 150, 200, 250, 300])
        # degrees_Lb = np.array([30, 60, 90, 120, 150])
        degrees_Lb = np.array([26, 51, 77, 103, 129, 154])
        # t_lab_Lb = np.array([96, 143, 200, 300])
        # degrees_Lb = np.array([60, 120])
        X_Lb = gm.cartesian(t_lab_Lb, degrees_Lb)
        # print(X_Lb)
        Lb_colors = self.light_colors[-2:]
        # print(self.light_colors)
        # print(Lb_colors)
        Lambda_b_array = np.arange(1, 1501, 1)
        
        # scale invariant: df = 0
        Lb_model = gm.TruncationPointwise(df = 0, excluded = [0])
        
        ratios_sgt_Lb = [Q_approx(E_to_p(t_lab_Lb, "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        ratios_dsg_Lb = [Q_approx(E_to_p(X_Lb[:, 0], "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        # print("sgt ratios has shape " + str(np.shape(ratios_sgt_Lb)))
        # print(np.shape(ratios_dsg_Lb))
        # print(X_Lb[:, 0])
        logprior = Lb_logprior(Lambda_b_array)
        
        # print(self.nn_orders_mask)
        # print(self.mask_restricted)
        # print(self.raw_data_mask)
        # print(self.nn_orders)
        # print(self.nn_orders_full)
        # print(self.nn_orders_full[self.nn_orders_mask])
        # print((self.nn_orders_full[self.nn_orders_mask])[self.mask_restricted])
       
        
        # Mask unused SGT data, and compute results
        # print(SGT.shape)
        # print(SGT[self.nn_orders_mask, :].shape)
        # print((SGT[self.nn_orders_mask, :])[self.mask_restricted, :].shape)
        sgt_Lb = (SGT[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)]
        sgt_Lb_nho_result = compute_posterior_intervals(
            Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 2,
            logprior=logprior, Lb=Lambda_b_array)
        sgt_Lb_ho_result = compute_posterior_intervals(
            Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 1,
            logprior = logprior, Lb = Lambda_b_array)
        
        # Mask unused DSG data, and compute results
        dsg_Lb = np.reshape((DSG[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
        # print("dsg_Lb = " + str(dsg_Lb))
        dsg_Lb_nho_result = compute_posterior_intervals(
            Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 2,
            logprior = logprior, Lb = Lambda_b_array)
        dsg_Lb_ho_result = compute_posterior_intervals(
            Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 1,
            logprior = logprior, Lb = Lambda_b_array)
        
        # Concatenate all spin observable data into one long vector, and compute results
        spins_Lb = np.concatenate([
            np.reshape((spin[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            for spin in [AY, D, A, AXX, AYY]],
            axis=1)
        ratios_spins_Lb = np.concatenate([ratios_dsg_Lb for i in [AY, D, A, AXX, AYY]], axis=1)
        spins_Lb_nho_result = compute_posterior_intervals(
            Lb_model, spins_Lb, ratios_spins_Lb, ref = 1, 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 2,
            logprior = logprior, Lb = Lambda_b_array)
        spins_Lb_ho_result = compute_posterior_intervals(
            Lb_model, spins_Lb, ratios_spins_Lb, ref = 1, 
            orders = self.nn_orders, 
            max_idx = max(self.nn_orders) - 1,
            logprior = logprior, Lb = Lambda_b_array)
        
        # Gather the above results
        results = [
            sgt_Lb_nho_result, sgt_Lb_ho_result,
            dsg_Lb_nho_result, dsg_Lb_ho_result,
            spins_Lb_nho_result, spins_Lb_ho_result
        ]
        # results = [dsg_Lb_n4lo_result]
        
        # Plot each posterior and its summary statistics
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
        for i, (posterior, bounds, median) in enumerate(results):
            posterior = posterior / (1.2*np.max(posterior))  # Scale so they're all the same height
            # Make the lines taper off
            Lb_vals = Lambda_b_array[posterior > 1e-2]
            posterior = posterior[posterior > 1e-2]
            # Plot and fill posterior, and add summary statistics
            
            ax.plot(Lb_vals, posterior-i, c='gray')
            
            # if i == 0: pdf_label = self.orders_dict[(np.sort(self.nn_orders))[-2]]
            # elif i == 1: pdf_label = self.orders_dict[max(self.nn_orders)]
            # else: pdf_label = '_nolegend_'
                
            ax.fill_between(Lb_vals, -i, posterior-i, facecolor=Lb_colors[i % 2])
            draw_summary_statistics(*bounds, median, ax=ax, height=-i)
        
        # Plot formatting
        ax.set_yticks([-0, -2, -4])
        ax.set_yticks([-1.1, -3.1], minor=True)
        ax.set_yticklabels([r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$'])
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=7, right=True)
        ax.set_xlim(0, 1200)
        ax.set_xticks([0, 300, 600, 900, 1200])
        ax.set_xlabel(r'$\Lambda_b$ (MeV)')
        ax.legend(title = r'$\mathrm{pr}(\Lambda_{b} \, | \, \vec{\mathbf{y}}_{k}, \mathbf{f})$', 
                  handles = [Patch(facecolor=Lb_colors[0], 
                                   edgecolor='gray', 
                                   linewidth = 1, 
                                   label=self.orders_dict[(np.sort(self.nn_orders))[-2]]), 
                             Patch(facecolor=Lb_colors[1], 
                                   edgecolor='gray',  
                                   linewidth = 1, 
                                   label=self.orders_dict[max(self.nn_orders)])])
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        
        if 'fig' in locals() and whether_save:
            fig.tight_layout()
    
            fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                    'Lambdab_posterior_pdf_pointwise' + '_' + self.scheme + '_' + 
                        self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                    '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                    self.train_pts_loc + '_' + self.p_param + 
                    self.filename_addendum).replace('_0MeVlab_', '_'))
        
        # except:
        #     print("Error in plotting the pointwise posterior PDF.")

    def plot_lambda_posterior_curvewise(self, SGT, DSG, AY, A, D, AXX, AYY, t_lab, degrees, 
                        ax = None, whether_save = True):
        
        # functions for interpolating the ratio and reference scale in the TruncationGP
        # def lambda_interp_f_ref(x_):
        #     X = np.ravel(x_)
        #     return self.interp_f_ref(X)
        def lambda_interp_f_ratio_Lb_tlab(x_, lambda_var):
            X = np.ravel(x_)
            return interp_f_ratio_Lb_tlab(X) * self.Lambda_b / lambda_var
        def lambda_interp_f_ratio_Lb_degrees(x_, lambda_var):
            X = np.ravel(x_)
            return interp_f_ratio_Lb_degrees(X) * self.Lambda_b / lambda_var
        
        interp_f_ratio_Lb_tlab = interp1d(E_to_p(t_lab, "np"), 
                Q_approx(E_to_p(t_lab, "np"), self.Q_param, self.Lambda_b, interaction='np'))
        # interp_f_ratio_Lb_degrees = interp1d(-1. * np.cos(np.radians(degrees)),
        #         Q_approx(E_to_p(t_lab_prime_loop, "np"), self.Q_param, self.Lambda_b, interaction='np') * len(degrees))
        
        # try:
        # t_lab_Lb = np.array([50, 100, 150, 200, 250, 300])
        # t_lab_Lb = np.array([100, 250])
        t_lab_Lb = np.array([50, 100, 150, 200, 250, 300])
        t_lab_Lb_prime = E_to_p(t_lab_Lb, "np")
        # degrees_Lb = np.array([60, 120])
        degrees_Lb = np.array([26, 51, 77, 103, 129, 154])
        # degrees_Lb = np.array([30, 60, 90, 120, 150])
        degrees_Lb_prime = -1. * np.cos(np.radians(degrees_Lb))
        # t_lab_Lb = np.array([96, 143, 200, 300])
        # degrees_Lb = np.array([60, 120])
        X_Lb = gm.cartesian(t_lab_Lb, degrees_Lb)
        X_Lb_prime = gm.cartesian(t_lab_Lb_prime, degrees_Lb_prime)
        # print(X_Lb_prime)
        Lb_colors = self.light_colors[-2:]
        # print(self.light_colors)
        # print(Lb_colors)
        # Lambda_b_array = np.arange(1, 1501, 1)
        
        # ratios_sgt_Lb = [Q_approx(E_to_p(t_lab_Lb, "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        # print(np.shape(ratios_sgt_Lb))
        # ratios_dsg_Lb = [Q_approx(E_to_p(X_Lb[:, 0], "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        # # print(ratios_dsg_Lb[13])
        # logprior = Lb_logprior(Lambda_b_array)
        
        # creates the grid over which the posterior PDF will be plotted
        # self.ls_vals = self.posteriorgrid.x_vals
        # self.lambda_vals = self.posteriorgrid.y_vals
        lambda_vals_Lb = np.arange(1, 1501, 10)
        ls_vals_Lb = np.arange(0.02, 2.02, 0.02)
        
        lambda_logprior = Lb_logprior(lambda_vals_Lb)
        
        # # creates and fits the TruncationGP
        # self.gp_post = gm.TruncationGP(self.kernel, 
        #                             ref = lambda_interp_f_ref, 
        #                             ratio = lambda_interp_f_ratio, 
        #                             center = self.center, 
        #                             disp = self.disp, 
        #                             df = self.df, 
        #                             scale = self.std_est, 
        #                             excluded = [0], 
        #                             ratio_kws = {"lambda_var" : self.Lambda_b})
        
        # Mask unused SGT data, and compute results
        # print(SGT.shape)
        # print(SGT[self.nn_orders_mask, :].shape)
        # print((SGT[self.nn_orders_mask, :])[self.mask_restricted, :].shape)
        sgt_Lb = (SGT[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)]
        # print("sgt_Lb has shape " + str(sgt_Lb.shape))
        # creates and fits the TruncationGP
        gp_post_sgt_Lb_nho = gm.TruncationGP(self.kernel, 
                                    ref = sgt_Lb[0], 
                                    ratio = lambda_interp_f_ratio_Lb_tlab, 
                                    center = self.center, 
                                    disp = self.disp, 
                                    df = self.df, 
                                    scale = self.std_est, 
                                    excluded = [0], 
                                    ratio_kws = {"lambda_var" : self.Lambda_b})
        gp_post_sgt_Lb_nho.fit(t_lab_Lb_prime[:, None],  
                              sgt_Lb.T, 
                              orders = self.nn_orders_full, 
                              orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
        gp_post_sgt_Lb_ho = gm.TruncationGP(self.kernel, 
                                    ref = sgt_Lb[0], 
                                    ratio = lambda_interp_f_ratio_Lb_tlab, 
                                    center = self.center, 
                                    disp = self.disp, 
                                    df = self.df, 
                                    scale = self.std_est, 
                                    excluded = [0], 
                                    ratio_kws = {"lambda_var" : self.Lambda_b})
        gp_post_sgt_Lb_ho.fit(t_lab_Lb_prime[:, None],  
                              sgt_Lb.T, 
                              orders = self.nn_orders_full, 
                              orders_eval = self.nn_orders[:len(self.nn_orders)])
        # sgt_Lb_nho_result = compute_posterior_intervals(
        #     Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
        #     orders = self.nn_orders, 
        #     max_idx = max(self.nn_orders) - 2,
        #     logprior=logprior, Lb=Lambda_b_array)
        # sgt_Lb_ho_result = compute_posterior_intervals(
        #     Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
        #     orders = self.nn_orders, 
        #     max_idx = max(self.nn_orders) - 1,
        #     logprior = logprior, Lb = Lambda_b_array)
        
        # evaluates the probability across the mesh
        ls_lambda_loglike_nho = np.array([[
            gp_post_sgt_Lb_nho.log_marginal_likelihood([ls_,], 
                    orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                    **{"lambda_var" : lambda_})
                for ls_ in np.log(ls_vals_Lb)]
                for lambda_ in lambda_vals_Lb])

        # adds the log prior to the log likelihood
        ls_lambda_loglike_nho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_nho)[1], 1) ).T
        
        # Makes sure that the values don't get too big or too small
        ls_lambda_like_nho = np.exp(ls_lambda_loglike_nho - np.max(ls_lambda_loglike_nho))

        # Now compute the marginal distributions
        lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
        # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)

        # Normalize them
        lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
        # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)

        sgt_Lb_nho_result = lambda_like_nho
        
        ls_lambda_loglike_ho = np.array([[
            gp_post_sgt_Lb_ho.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders)],
                                                  **{"lambda_var" : lambda_})
                for ls_ in np.log(ls_vals_Lb)]
                for lambda_ in lambda_vals_Lb])
        
        # adds the log prior to the log likelihood
        ls_lambda_loglike_ho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_ho)[1], 1) ).T

        # Makes sure that the values don't get too big or too small
        ls_lambda_like_ho = np.exp(ls_lambda_loglike_ho - np.max(ls_lambda_loglike_ho))

        # Now compute the marginal distributions
        lambda_like_ho = np.trapz(ls_lambda_like_ho, x = ls_vals_Lb, axis = -1)
        # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)

        # Normalize them
        lambda_like_ho /= np.trapz(lambda_like_ho, x = lambda_vals_Lb, axis = 0)
        # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)

        sgt_Lb_ho_result = lambda_like_ho
            
        dsg_Lb_nho_result = np.zeros((len(lambda_vals_Lb)))
        dsg_Lb_ho_result = np.zeros((len(lambda_vals_Lb)))
        spins_Lb_nho_result = np.zeros((len(lambda_vals_Lb)))
        spins_Lb_ho_result = np.zeros((len(lambda_vals_Lb)))
        
        for t_lab_Lb_loop in zip(t_lab_Lb, t_lab_Lb_prime):
            # print(np.shape(-1. * np.cos(np.radians(degrees))))
            # print(np.shape(Q_approx(E_to_p(t_lab_Lb_loop[1], "np"), self.Q_param, self.Lambda_b, interaction='np') * np.ones((len(degrees)))))
            interp_f_ratio_Lb_degrees = interp1d(-1. * np.cos(np.radians(degrees)),
                    Q_approx(E_to_p(t_lab_Lb_loop[1], "np"), self.Q_param, self.Lambda_b, interaction='np') * np.ones((len(degrees))))
            # # Mask unused DSG data, and compute results
            dsg_Lb = np.reshape((DSG[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            # print("dsg_Lb has shape " + str(np.shape(dsg_Lb)))
            # # print("dsg_Lb = " + str(dsg_Lb))
            # dsg_Lb_nho_result = compute_posterior_intervals(
            #     Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
            #     orders = self.nn_orders, 
            #     max_idx = max(self.nn_orders) - 2,
            #     logprior = logprior, Lb = Lambda_b_array)
            # dsg_Lb_ho_result = compute_posterior_intervals(
            #     Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
            #     orders = self.nn_orders, 
            #     max_idx = max(self.nn_orders) - 1,
            #     logprior = logprior, Lb = Lambda_b_array)
            gp_post_dsg_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = dsg_Lb[0], 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_dsg_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  dsg_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_dsg_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = dsg_Lb[0], 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_dsg_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  dsg_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            # evaluates the probability across the mesh
            ls_lambda_loglike_nho = np.array([[
                gp_post_dsg_Lb_nho.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                                                      **{"lambda_var" : lambda_})
                    for ls_ in np.log(ls_vals_Lb)]
                    for lambda_ in lambda_vals_Lb])
            
            # adds the log prior to the log likelihood
            ls_lambda_loglike_nho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_nho)[1], 1) ).T

            # Makes sure that the values don't get too big or too small
            ls_lambda_like_nho = np.exp(ls_lambda_loglike_nho - np.max(ls_lambda_loglike_nho))
    
            # Now compute the marginal distributions
            lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
            # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
    
            # Normalize them
            lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
            # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
    
            dsg_Lb_nho_result += lambda_like_nho
            
            ls_lambda_loglike_ho = np.array([[
                gp_post_dsg_Lb_ho.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders)],
                                                      **{"lambda_var" : lambda_})
                    for ls_ in np.log(ls_vals_Lb)]
                    for lambda_ in lambda_vals_Lb])
            
            # adds the log prior to the log likelihood
            ls_lambda_loglike_ho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_ho)[1], 1) ).T

            # Makes sure that the values don't get too big or too small
            ls_lambda_like_ho = np.exp(ls_lambda_loglike_ho - np.max(ls_lambda_loglike_ho))
    
            # Now compute the marginal distributions
            lambda_like_ho = np.trapz(ls_lambda_like_ho, x = ls_vals_Lb, axis = -1)
            # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
    
            # Normalize them
            lambda_like_ho /= np.trapz(lambda_like_ho, x = lambda_vals_Lb, axis = 0)
            # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
    
            dsg_Lb_ho_result += lambda_like_ho
            
            # # Concatenate all spin observable data into one long vector, and compute results
            # spins_Lb = np.concatenate([
            #     np.reshape((spin[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            #     for spin in [AY, D, A, AXX, AYY]],
            #     axis=1)
            # ratios_spins_Lb = np.concatenate([ratios_dsg_Lb for i in [AY, D, A, AXX, AYY]], axis=1)
            # # spins_Lb_nho_result = compute_posterior_intervals(
            # #     Lb_model, spins_Lb, ratios_spins_Lb, ref = 1, 
            # #     orders = self.nn_orders, 
            # #     max_idx = max(self.nn_orders) - 2,
            # #     logprior = logprior, Lb = Lambda_b_array)
            # # spins_Lb_ho_result = compute_posterior_intervals(
            # #     Lb_model, spins_Lb, ratios_spins_Lb, ref = 1, 
            # #     orders = self.nn_orders, 
            # #     max_idx = max(self.nn_orders) - 1,
            # #     logprior = logprior, Lb = Lambda_b_array)
            
            # # if self.constraint is not None and self.constraint[2] == self.x_quantity_name:
            # #     self.gp_post.fit(self.X_train, 
            # #                      self.y_train, 
            # #                      orders = self.nn_orders_full,
            # #                      orders_eval = self.nn_orders, 
            # #                      dX = np.array([[self.x[i]] for i in self.constraint[0]]), 
            # #                      dy = [j for j in self.constraint[1]])
            # # else:
            # # self.gp_post.fit(self.X_train, 
            # #                      self.y_train, 
            # #                      orders = self.nn_orders_full, 
            # #                      orders_eval = self.nn_orders)
            
            ay_Lb = np.reshape((AY[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            gp_post_ay_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_ay_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  ay_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_ay_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_ay_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  ay_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            a_Lb = np.reshape((A[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            gp_post_a_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_a_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  a_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_a_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_a_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  a_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            d_Lb = np.reshape((D[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            gp_post_d_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_d_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  d_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_d_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_d_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  d_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            axx_Lb = np.reshape((AXX[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            gp_post_axx_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_axx_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  axx_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_axx_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_axx_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  axx_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            ayy_Lb = np.reshape((AYY[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
            gp_post_ayy_Lb_nho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_ayy_Lb_nho.fit(degrees_Lb_prime[:, None],  
                                  ayy_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
            gp_post_ayy_Lb_ho = gm.TruncationGP(self.kernel, 
                                        ref = np.ones((len(degrees_Lb))), 
                                        ratio = lambda_interp_f_ratio_Lb_degrees, 
                                        center = self.center, 
                                        disp = self.disp, 
                                        df = self.df, 
                                        scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            gp_post_ayy_Lb_ho.fit(degrees_Lb_prime[:, None],  
                                  ayy_Lb.T, 
                                  orders = self.nn_orders_full, 
                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
            
            gp_fits_spins_nho = [gp_post_ay_Lb_nho, gp_post_a_Lb_nho, 
                        gp_post_d_Lb_nho, gp_post_axx_Lb_nho, gp_post_ayy_Lb_nho]
            gp_fits_spins_ho = [gp_post_ay_Lb_ho, gp_post_a_Lb_ho, 
                        gp_post_d_Lb_ho, gp_post_axx_Lb_ho, gp_post_ayy_Lb_ho]
            
            spins_ls_lambda_loglike_nho = np.zeros((len(lambda_vals_Lb), len(ls_vals_Lb)))
            spins_ls_lambda_loglike_ho = np.zeros((len(lambda_vals_Lb), len(ls_vals_Lb)))
            
            for gp_fit_spins in gp_fits_spins_nho:
                # evaluates the probability across the mesh
                ls_lambda_loglike_nho = np.array([[
                    gp_fit_spins.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                                                          **{"lambda_var" : lambda_})
                        for ls_ in np.log(ls_vals_Lb)]
                        for lambda_ in lambda_vals_Lb])
                
                spins_ls_lambda_loglike_nho += ls_lambda_loglike_nho
            
            # adds the log prior to the log likelihood
            ls_lambda_loglike_nho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_nho)[1], 1) ).T
    
            # Makes sure that the values don't get too big or too small
            ls_lambda_like_nho = np.exp(spins_ls_lambda_loglike_nho - np.max(spins_ls_lambda_loglike_nho))
    
            # Now compute the marginal distributions
            lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
            # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
    
            # Normalize them
            lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
            # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
    
            spins_Lb_nho_result = lambda_like_nho
            
            for gp_fit_spins in gp_fits_spins_ho:
                # evaluates the probability across the mesh
                ls_lambda_loglike_ho = np.array([[
                    gp_fit_spins.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders)],
                                                          **{"lambda_var" : lambda_})
                        for ls_ in np.log(ls_vals_Lb)]
                        for lambda_ in lambda_vals_Lb])
                
                spins_ls_lambda_loglike_ho += ls_lambda_loglike_ho
            
            # adds the log prior to the log likelihood
            ls_lambda_loglike_ho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_ho)[1], 1) ).T
    
            # Makes sure that the values don't get too big or too small
            ls_lambda_like_ho = np.exp(spins_ls_lambda_loglike_ho - np.max(spins_ls_lambda_loglike_ho))
    
            # Now compute the marginal distributions
            lambda_like_ho = np.trapz(ls_lambda_like_ho, x = ls_vals_Lb, axis = -1)
            # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
    
            # Normalize them
            lambda_like_ho /= np.trapz(lambda_like_ho, x = lambda_vals_Lb, axis = 0)
            # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
    
            spins_Lb_ho_result = lambda_like_ho
        
        # Normalize them
        dsg_Lb_ho_result /= np.trapz(dsg_Lb_ho_result, x = lambda_vals_Lb, axis = 0)
        dsg_Lb_nho_result /= np.trapz(dsg_Lb_nho_result, x = lambda_vals_Lb, axis = 0)
        spins_Lb_ho_result /= np.trapz(spins_Lb_ho_result, x = lambda_vals_Lb, axis = 0)
        spins_Lb_ho_result /= np.trapz(spins_Lb_nho_result, x = lambda_vals_Lb, axis = 0)
        
        # Gather the above results
        results = [
            sgt_Lb_nho_result, sgt_Lb_ho_result,
            dsg_Lb_nho_result, dsg_Lb_ho_result,
            spins_Lb_nho_result, spins_Lb_ho_result
        ]
        # results = [dsg_Lb_n4lo_result]
        
        # Plot each posterior and its summary statistics
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
        # for i, (posterior, bounds, median) in enumerate(results):
        for i, posterior_raw in enumerate(results):
            posterior = posterior_raw / (1.2*np.max(posterior_raw))  # Scale so they're all the same height
            # Make the lines taper off
            Lb_vals = lambda_vals_Lb[posterior > 1e-2]
            posterior = posterior[posterior > 1e-2]
            # Plot and fill posterior, and add summary statistics
            ax.plot(Lb_vals, posterior-i, c='gray')
            
            # if i == 0: pdf_label = self.orders_dict[(np.sort(self.nn_orders))[-2]]
            # elif i == 1: pdf_label = self.orders_dict[max(self.nn_orders)]
            # else: pdf_label = '_nolegend_'
                
            ax.fill_between(Lb_vals, -i, posterior-i, facecolor=Lb_colors[i % 2])
            # draw_summary_statistics(*bounds, median, ax=ax, height=-i)
            
            bounds = np.zeros((2,2))
            for j, p in enumerate([0.68, 0.95]):
                # bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
                bounds[j] = gm.hpd_pdf(pdf=posterior_raw, alpha=p, x=lambda_vals_Lb)
                # bounds[j] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb_vals)

            median = gm.median_pdf(pdf=posterior_raw, x=lambda_vals_Lb)
            # median = gm.median_pdf(pdf=posterior, x=Lb_vals)

            draw_summary_statistics(*bounds, median, ax=ax, height=-i)
        
        # Plot formatting
        ax.set_yticks([-0, -2, -4])
        ax.set_yticks([-1.1, -3.1], minor=True)
        ax.set_yticklabels([r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$'])
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=7, right=True)
        ax.set_xlim(0, 1200)
        ax.set_xticks([0, 300, 600, 900, 1200])
        ax.set_xlabel(r'$\Lambda_b$ (MeV)')
        ax.legend(title = r'$\mathrm{pr}(\Lambda_{b} \, | \, \vec{\mathbf{y}}_{k}, \ell, \mathbf{f})$', 
                  handles = [Patch(facecolor=Lb_colors[0], 
                                   edgecolor='gray', 
                                   linewidth = 1, 
                                   label=self.orders_dict[(np.sort(self.nn_orders))[-2]]), 
                             Patch(facecolor=Lb_colors[1], 
                                   edgecolor='gray',  
                                   linewidth = 1, 
                                   label=self.orders_dict[max(self.nn_orders)])])
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        
        if 'fig' in locals() and whether_save:
            fig.tight_layout()
    
            fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                    'Lambdab_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                        self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                    '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                    self.train_pts_loc + '_' + self.p_param + 
                    self.filename_addendum).replace('_0MeVlab_', '_'))
        
        # except:
        #     print("Error in plotting the curvewise posterior PDF.")

    def plot_posteriors_curvewise(self, SGT, DSG, AY, A, D, AXX, AYY, 
                        t_lab, t_lab_pts, degrees, degrees_pts, 
                        Lambda_b_true, ls_true, mpi_true, orders = 2, ax = None, whether_plot_joint_Lbls = True, 
                        whether_plot_joint_mpils = True, whether_plot_joint_Lbmpi = True, 
                        whether_plot_lambda = True, whether_plot_mpi = True, 
                        whether_plot_corner = True, whether_save = True):
        
        # functions for interpolating the ratio in the TruncationGP
        def interp_f_ratio_Lb_mpi(x_map, x_interp, x_interp_Q, Q_param, mpi_var, lambda_var):
            X = np.ravel(x_map)
        
            return (interp1d( x_interp, Q_approx(x_interp_Q, Q_param, Lambda_b = lambda_var, m_pi = mpi_var) 
                             * np.ones(len(x_interp)) ))(X)
        
        def make_likelihood_filename(
            folder,
            order_name,
            logpriors_names, 
            random_vars_array,
        ):
            filename = (
                str(folder)
                + "/"
                + "posterior_pdf_curvewise"
                + "_"
                + str(self.observable_name)
                + "_"
                + str(self.scheme)
                + "_"
                + str(self.scale)
                + "_"
                + str(order_name)
                + "_"
                + "Q"
                + str(self.Q_param)
                + "_"
                + str(self.p_param)
                + "_"
                + str(self.vs_what)
            )

            for logprior in logpriors_names:
                filename += "_" + str(logprior)
            for random_var in random_vars_array:
                filename += (
                    "_"
                    + str(random_var.name)
                    + str(len(random_var.var))
                    + "pts"
                    # + f"{min(random_var.var):.2f}"
                    # + "to"
                    # + f"{max(random_var.var):.2f}"
                )
            print(filename)
            return str(filename.replace("__", "_") + ".txt")
        
        # sets the number of orders and the corresponding colors
        order_num = int(orders)
        Lb_colors = self.light_colors[-1 * order_num:]
        
        # sets the meshes for the random variable arrays
        mpi_vals = np.linspace(mpi_true / 3.1, mpi_true * 3.1, 24)
        if self.observable_name == "SGT":
            ls_vals = np.linspace(1, 300, 25)
        else:
            ls_vals = np.linspace(0.02, 2.00, 25)
        lambda_vals = np.linspace(0.5 * np.max(mpi_vals), 1500, 26)
        mesh_cart = gm.cartesian(lambda_vals, np.log(ls_vals), mpi_vals)
        
        # sets the "true" values for the random variables
        variables_true_list = [Lambda_b_true, ls_true, mpi_true]
        
        # sets the priors
        # lambda_logprior = Lb_logprior(lambda_vals)
        # ls_logprior = np.zeros(len(ls_vals))
        # mpi_logprior = mpieff_logprior(mpi_vals)
        # logprior_list = [lambda_logprior, ls_logprior, mpi_logprior]
        # logprior[]_name_list = ["Lambdab_uniformlogprior", "ls_nologprior", "mpieff_uniformlogprior"]
                      
        # sets the RandomVariable objects 
        LambdabVariable = RandomVariable(var = lambda_vals, 
                                         name = 'Lambdab', 
                                         label = "\Lambda_{b}", 
                                         units = "MeV", 
                                         ticks = [300, 600, 900, 1200], 
                                         logprior = Lb_logprior(lambda_vals), 
                                         logprior_name = "Lambdab_uniformlogprior")
        # this will need to change for SGT vs. other observables
        LsVariable = RandomVariable(var = ls_vals, 
                                         name = 'ls', 
                                         label = "\ell", 
                                         units = "", 
                                         ticks = [], 
                                         logprior = np.zeros(len(ls_vals)), 
                                         logprior_name = "ls_nologprior")
        MpieffVariable = RandomVariable(var = mpi_vals, 
                                         name = 'mpieff', 
                                         label = "m_{\pi}", 
                                         units = "MeV", 
                                         ticks = [50, 100, 150, 200, 250, 300, 350], 
                                         logprior = mpieff_logprior(mpi_vals), 
                                         logprior_name = "mpieff_uniformlogprior")
        variables_array = np.array([LambdabVariable, LsVariable, MpieffVariable])
        
        # initiates the ray kernel for utilizing all processors on the laptop
        ray.shutdown()
        ray.init()
        
        @ray.remote
        def log_likelihood(gp_fitted, orders, x_interp, x_interp_Q, mesh_points):
            return [gp_fitted.log_marginal_likelihood([pt[1],], 
                        orders_eval = orders,
                        **{"x_interp" : x_interp,
                            "x_interp_Q" : x_interp_Q, 
                            "Q_param" : self.Q_param, 
                            "mpi_var" : pt[2], 
                            "lambda_var" : pt[0]}) for pt in mesh_points]
        
        BATCH_SIZE = 100
        
        try:
            like_list = []
            
            # generates names for files and searches for whether they exist
            for order_counter in range(1, order_num + 1):
                order = np.max(self.nn_orders) - order_num + order_counter
                print("order = " + str(order))
                
                # if they exist, they are read in, reshaped, and appended to like_list
                np.append(like_list, np.reshape(
                    np.loadtxt(make_likelihood_filename(
                        "data",
                        self.orders_names_dict[order],
                        [variable.logprior_name for variable in variables_array], 
                        variables_array,
                    )),
                    (len(random_var.var) for random_var in variables_array),
                ))
        
        except:
            like_list = []
            
            for order_counter in range(1, order_num + 1):
                # sets the order number
                order = np.max(self.nn_orders) - order_num + order_counter
                print("order = " + str(order))
                orders_nho_ray = ray.put(self.nn_orders[:order])
                
                if self.observable_name == "SGT":
                    obs_name_corner = "SGT"
                    posterior_label = r'$\sigma$'
                    
                    # converts the points in t_lab_pts to the current input space
                    t_lab_input = self.inputspace.input_space(**{"E_lab" : t_lab})
                    t_lab_pts_input = self.inputspace.input_space(**{"E_lab" : t_lab_pts})
                    
                    # sieves the data
                    sgt_data = (SGT[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_pts)]
                    
                    # creates and fits the TruncationGP object
                    gp_post_sgt_nho = gm.TruncationGP(self.kernel, 
                                                ref = sgt_data[0], 
                                                ratio = interp_f_ratio_Lb_mpi, 
                                                center = self.center, 
                                                disp = self.disp, 
                                                df = self.df, 
                                                scale = self.std_est, 
                                                excluded = [0], 
                                                ratio_kws = {"x_interp" : t_lab_input,
                                                        "x_interp_Q" : E_to_p(t_lab, "np"), 
                                                        "Q_param" : self.Q_param, 
                                                        "mpi_var" : mpi_true,  
                                                        "lambda_var" : Lambda_b_true})
                    gp_post_sgt_nho.fit(t_lab_pts_input[:, None],  
                                          sgt_data.T, 
                                          orders = self.nn_orders_full, 
                                          orders_eval = self.nn_orders[:order])
                    
                    # puts important objects into ray objects
                    gp_post_nho_ray = ray.put(gp_post_sgt_nho)
                    t_lab_mom_ray = ray.put(E_to_p(t_lab, "np"))
                    
                    # calculates the posterior using ray
                    log_like_ids = []
                    for i in range(0, len(mesh_cart), BATCH_SIZE):
                        batch = mesh_cart[i : i + BATCH_SIZE]
                        log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                orders_nho_ray, t_lab_input, t_lab_mom_ray, batch))
                    log_like = list(itertools.chain(*ray.get(log_like_ids)))
                    obs_loglike = np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                
                elif self.observable_name == "DSG":
                    obs_name_corner = "DSG"
                    posterior_label = r'$\displaystyle\frac{d\sigma}{d\Omega}$'
                    
                    obs_loglike = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    
                    for t_lab_pt, t_lab_mom_pt in zip(t_lab_pts, E_to_p(t_lab_pts, "np")):
                        # converts the points in t_lab_pts to the current input space
                        degrees_input = self.inputspace.input_space(**{"deg_input" : degrees, 
                                                                       "p_input" : t_lab_mom_pt})
                        degrees_pts_input = self.inputspace.input_space(**{"deg_input" : degrees_pts, 
                                                                       "p_input" : t_lab_mom_pt})
                        degrees_input_ray = ray.put(degrees_input)
                        
                        # not sure why we doubly transformed the energies here before
                        ratio_points_ray = ray.put(E_to_p(t_lab_pt, "np"))
                        
                        # sieves the data
                        dsg_data = np.reshape((DSG[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_pt)][..., np.isin(degrees, degrees_pts)], (len(self.nn_orders), -1))
                        
                        # creates and fits the TruncationGP object
                        gp_post_dsg_nho = gm.TruncationGP(self.kernel, 
                                                    ref = dsg_data[0], 
                                                    ratio = interp_f_ratio_Lb_mpi, 
                                                    center = self.center, 
                                                    disp = self.disp, 
                                                    df = self.df, 
                                                    scale = self.std_est, 
                                                    excluded = [0], 
                                                    ratio_kws = {"x_interp" : degrees_input,
                                                                  "x_interp_Q" : t_lab_mom_pt, 
                                                                "Q_param" : self.Q_param, 
                                                                "mpi_var" : mpi_true,  
                                                                "lambda_var" : Lambda_b_true})
                        gp_post_dsg_nho.fit(degrees_pts_input[:, None],  
                                              dsg_data.T, 
                                              orders = self.nn_orders_full, 
                                              orders_eval = self.nn_orders[:order])
                        
                        # puts important objects into ray objects
                        gp_post_nho_ray = ray.put(gp_post_dsg_nho)
                        
                        # calculates the posterior using ray
                        log_like_ids = []
                        for i in range(0, len(mesh_cart), BATCH_SIZE):
                            batch = mesh_cart[i : i + BATCH_SIZE]
                            log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                    orders_nho_ray, degrees_input_ray, ratio_points_ray, batch))
                        log_like = list(itertools.chain(*ray.get(log_like_ids)))
                        obs_loglike += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                        
                elif self.observable_name == "A" or self.observable_name == "AY" or \
                    self.observable_name == "D" or self.observable_name == "AYY" or \
                    self.observable_name == "AXX":
                    obs_name_corner = "spins"
                    posterior_label = r'$X_{pqik}$'
                    
                    obs_loglike = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    spin_obs_list = [AY, A, D, AXX, AYY]
                    
                    for t_lab_pt, t_lab_mom_pt in zip(t_lab_pts, E_to_p(t_lab_pts, "np")):
                        # converts the points in t_lab_pts to the current input space
                        degrees_input = self.inputspace.input_space(**{"deg_input" : degrees, 
                                                                       "p_input" : t_lab_mom_pt})
                        degrees_pts_input = self.inputspace.input_space(**{"deg_input" : degrees_pts, 
                                                                       "p_input" : t_lab_mom_pt})
                        degrees_input_ray = ray.put(degrees_input)
                        
                        # not sure why we doubly transformed the energies here before
                        ratio_points_ray = ray.put(E_to_p(t_lab_pt, "np"))
                        
                        gp_fits_spins_nho = []
                        # gp_fits_spins_ho = []
                        
                        for so, spin_obs in enumerate(spin_obs_list):
                            # sieves the data
                            spin_data = np.reshape((spin_obs[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_pt)][..., np.isin(degrees, degrees_pts)], (len(self.nn_orders), -1))
                            
                            # creates and fits the TruncationGP object
                            gp_fits_spins_nho.append(gm.TruncationGP(self.kernel, 
                                                        ref = np.ones((len(degrees_pts))), 
                                                        ratio = interp_f_ratio_Lb_mpi, 
                                                        center = self.center, 
                                                        disp = self.disp, 
                                                        df = self.df, 
                                                        scale = self.std_est, 
                                                        excluded = [0], 
                                                        ratio_kws = {"x_interp" : degrees_input,
                                                                      "x_interp_Q" : t_lab_mom_pt, 
                                                                    "Q_param" : self.Q_param, 
                                                                    "mpi_var" : mpi_true,  
                                                                    "lambda_var" : Lambda_b_true}))
                            gp_fits_spins_nho[so].fit(degrees_pts_input[:, None],  
                                                  spin_data.T, 
                                                  orders = self.nn_orders_full, 
                                                  orders_eval = self.nn_orders[:order])
                        
                        for gp_fit_spins in gp_fits_spins_nho:
                            # puts important objects into ray objects
                            gp_post_nho_ray = ray.put(gp_fit_spins)
                            
                            # calculates the posterior using ray
                            log_like_ids = []
                            for i in range(0, len(mesh_cart), BATCH_SIZE):
                                batch = mesh_cart[i : i + BATCH_SIZE]
                                log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                        orders_nho_ray, degrees_input_ray, ratio_points_ray, batch))
                            log_like = list(itertools.chain(*ray.get(log_like_ids)))
                            obs_loglike += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                        
                # loglike_list.append(obs_loglike)
            
                # adds the log-priors to the log-likelihoods
                for i, logprior in enumerate([variable.logprior_name for variable in variables_array]):
                    obs_loglike += np.transpose(np.tile( logprior, 
                            (
                              np.shape(obs_loglike)[(i + 1) % len(variables_array)], 
                              np.shape(obs_loglike)[(i + 2) % len(variables_array)], 
                              1
                              ) 
                        ), np.roll(np.arange(0, len(variables_array), dtype = int), i + 1))
                    
                # Makes sure that the values don't get too big or too small
                obs_like = np.exp(obs_loglike - np.max(obs_loglike))
                like_list.append(obs_like)
                
                np.savetxt(make_likelihood_filename(
                    "data",
                    self.orders_names_dict[order],
                    [variable.logprior_name for variable in variables_array], 
                    variables_array,
                ), np.reshape(obs_like, (np.prod([len(random_var.var) for random_var in variables_array]))))
                # np.savetxt('data/posterior_pdf_curvewise_SGT_SMS_500MeV_N3LO_Qsmoothmax_Qofprel_Elab_Lambdab_uniformlogprior_ls_nologprior_mpieff_uniformlogprior_Lambdab26pts213p90to1500p00_ls25pts1p00to300p00_mpieff24pts44p52to427p80.txt', np.reshape(obs_like, (np.prod([len(random_var.var) for random_var in variables_array]))))
    
   
        marg_post_list = []
        joint_post_list = []
        
        for like_idx, like in enumerate(like_list):
            # creates the normalized fully marginalized posteriors
            for v, var in enumerate(variables_array):
                var_idx_array = np.arange(0, np.shape(variables_array)[0], 1, dtype = int)
                var_idx_array = var_idx_array[var_idx_array != v]
                var_idx_array = np.flip(var_idx_array)
                
                marg_post = np.copy(like)
                
                for idx in var_idx_array:
                    marg_post = np.trapz(marg_post, x = variables_array[idx].var, axis = idx)
                
                marg_post /= np.trapz(marg_post, x = variables_array[v].var, axis = 0)
                
                marg_post_list.append(marg_post)
            
            # creates the normalized joint posteriors
            comb_array = np.array(np.meshgrid(np.arange(0, np.shape(variables_array)[0], 1, dtype = int), 
                                              np.arange(0, np.shape(variables_array)[0], 1, dtype = int))).T.reshape(-1, 2)
            comb_array = np.delete(comb_array, [comb[0] >= comb[1] for comb in comb_array], axis = 0)
            p = np.argsort(comb_array[:, 1])
            comb_array = np.flip(comb_array[p], axis = 0)
            
            for v in np.flip(np.roll(np.arange(0, np.shape(variables_array)[0], 1, dtype = int), 1)):
                joint_post = np.trapz(like, x = variables_array[v].var, axis = v)
                
                joint_post /= np.trapz(np.trapz(joint_post, 
                                x = variables_array[comb_array[v, 1]].var, axis = 1), 
                                x = variables_array[comb_array[v, 0]].var, axis = 0
                            )
                
                joint_post_list.append(joint_post)
        
        marg_post_array = np.reshape(marg_post_list, (len(variables_array), order_num), order = 'F')
        joint_post_array = np.reshape(joint_post_list, (len(variables_array) * (len(variables_array) - 1) // 2, order_num), order = 'F')
        
        for (variable, result) in zip(variables_array, marg_post_array):
            # Plot each posterior and its summary statistics
            fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
            
            for i, posterior_raw in enumerate(result):
                # scales the posteriors so they're all the same height
                posterior = posterior_raw / (1.2*np.max(posterior_raw))
                # Make the lines taper off
                vals_restricted = variable.var[posterior > 1e-2]
                posterior = posterior[posterior > 1e-2]
                # Plot and fill posterior, and add summary statistics
                ax.plot(vals_restricted, posterior-i, c='gray')
                    
                ax.fill_between(vals_restricted, -i, posterior-i, facecolor=Lb_colors[i])
                
                bounds = np.zeros((2,2))
                for j, p in enumerate([0.68, 0.95]):
                    bounds[j] = gm.hpd_pdf(pdf=posterior_raw, alpha=p, x=variable.var)
    
                median = gm.median_pdf(pdf=posterior_raw, x=variable.var)
    
                draw_summary_statistics(*bounds, median, ax=ax, height=-i)
            
            # Plot formatting
            ax.set_yticks([0])
            ax.set_yticklabels([posterior_label])
            ax.tick_params(axis='both', which='both', direction='in')
            ax.tick_params(which='major', length=0)
            ax.tick_params(which='minor', length=7, right=True)
            ax.set_xticks(variable.ticks)
            ax.set_xlabel((r'$' + variable.label + r'$ (' + variable.units + ')').replace('()', ''))
            ax.legend(title = r'$\mathrm{pr}(' + variable.label + r' \, | \, \vec{\mathbf{y}}_{k}, \mathbf{f})$', 
                      handles = [Patch(facecolor=Lb_colors[o], 
                                       edgecolor='gray', 
                                       linewidth = 1, 
                                       label=self.orders_labels_dict[(np.sort(self.nn_orders))[-1 - o]])
                                 for o in range(0, order_num)])
            ax.grid(axis='x')
            ax.set_axisbelow(True)
            
            if 'fig' in locals() and whether_save:
                fig.tight_layout()
        
                fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                        variable.name + '_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                            self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))
        
                   
        if whether_plot_corner:
            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                cmap_name = 'Blues'
                cmap = mpl.cm.get_cmap(cmap_name)
                
                for i in range(order_num):
                    # sets up axes
                    n_plots = np.shape(variables_array)[0]
                    fig, ax_joint_array, ax_marg_array, ax_title = corner_plot(n_plots = n_plots)
                        
                    mean_list = []
                    stddev_list = []
                    
                    for variable_idx, variable in enumerate(np.roll(variables_array, 1)):
                        # Now plot the marginal distributions
                        dist_mean, dist_stddev = mean_and_stddev(variable.var, marg_post_array[variable_idx - 1, i])
                        ax_marg_array[variable_idx].set_xlim(left = np.max([0, dist_mean - 5 * dist_stddev]), 
                                                             right = dist_mean + 5 * dist_stddev)
                        mean_list.append(dist_mean)
                        stddev_list.append(dist_stddev)
                        dist_mean = sig_figs(dist_mean, 3)
                        dist_stddev = round_to_same_digits(dist_stddev, dist_mean)
                        ax_marg_array[variable_idx].set_title(rf'{dist_mean} $\pm$ {dist_stddev}', 
                                                        fontsize = 18)
                        
                        ax_marg_array[variable_idx].plot(variable.var, marg_post_array[variable_idx - 1, i], c=cmap(0.8), lw=1)
                        ax_marg_array[variable_idx].fill_between(variable.var, np.zeros_like(variable.var),
                                                marg_post_array[variable_idx - 1, i], facecolor=cmap(0.2), lw=1)
                        try:
                            ax_marg_array[variable_idx].axvline(np.roll(variables_true_list, 1)[variable_idx], 0, 1, c=gray, lw=1, zorder=0)
                        except:
                            pass
                        if variable_idx == np.shape(variables_array)[0] - 1:
                            ax_marg_array[variable_idx].set_xticklabels(variable.ticks)
                    
                    comb_array = np.array(np.meshgrid(np.arange(0, np.shape(variables_array)[0], 1, dtype = int), 
                                                      np.arange(0, np.shape(variables_array)[0], 1, dtype = int))).T.reshape(-1, 2)
                    comb_array = np.delete(comb_array, [comb[0] >= comb[1] for comb in comb_array], axis = 0)
                    p = np.argsort(comb_array[:, 1])
                    comb_array = comb_array[p]
                    
                    for joint_idx, joint in enumerate(joint_post_array[:, i]):
                        # plots contours
                        ax_joint_array[joint_idx].set_xlim(left = np.max([0, mean_list[comb_array[joint_idx, 0]] - 5 * stddev_list[comb_array[joint_idx, 0]]]), 
                                                           right = mean_list[comb_array[joint_idx, 0]] + 5 * stddev_list[comb_array[joint_idx, 0]])
                        ax_joint_array[joint_idx].set_ylim(bottom = np.max([0, mean_list[comb_array[joint_idx, 1]] - 5 * stddev_list[comb_array[joint_idx, 1]]]), 
                                                           top = mean_list[comb_array[joint_idx, 1]] + 5 * stddev_list[comb_array[joint_idx, 1]])
                        try:
                            ax_joint_array[joint_idx].contour(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var, 
                                                              np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var, 
                                                              joint,
                                              levels = [np.amax(joint) * level for level in \
                                                        ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])],
                                              cmap=cmap_name)
                            corr_coeff = correlation_coefficient(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var, 
                                                                 np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var, 
                                                                 joint)
                        except:
                            ax_joint_array[joint_idx].contour(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var, 
                                                              np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var, 
                                                              joint.T,
                                              levels = [np.amax(joint.T) * level for level in \
                                                        ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])],
                                              cmap=cmap_name)
                            corr_coeff = correlation_coefficient(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var, 
                                                                 np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var, 
                                                                 joint.T)
                        ax_joint_array[joint_idx].text(.99, .99, rf'$\rho$ = {corr_coeff:.2f}', 
                                                    ha='right', va='top', 
                                                    transform = ax_joint_array[joint_idx].transAxes, 
                                                    fontsize = 18)
                        try:
                            ax_joint_array[joint_idx].axvline(np.roll(variables_true_list, 1)[comb_array[joint_idx, 0]], 0, 1, c=gray, lw=1, zorder=0)
                        except:
                            pass
                        try:
                            ax_joint_array[joint_idx].axhline(np.roll(variables_true_list, 1)[comb_array[joint_idx, 1]], 0, 1, c=gray, lw=1, zorder=0)
                        except:
                            pass
                        
                    ax_title.text(.99, .99, 
                                obs_name_corner + '\n' + 
                                self.scheme + '\,' + self.scale + '\n' + 
                                r'' + self.orders_labels_dict[max(self.nn_orders) - 1 + (i % 2)] + '\n' + 
                                r'$Q_{\mathrm{' + self.Q_param + '}}$' + '\n' + 
                                self.p_param + '\n' + 
                                self.vs_what,
                                ha='right', va='top', 
                                transform = ax_title.transAxes, 
                                fontsize = 25)
                    
                    
                    plt.show()
            
            print("mean_list = " + str(mean_list))
            
            # generates the coefficient plots and the MC and PC with the optimal parameters instead
            ratio_optimal = Q_approx(
                self.inputspace.mom,
                self.Q_param,
                Lambda_b=mean_list[1],
                m_pi=mean_list[0],
            )
            
            # Extract the coefficients and define kernel
            self.coeffs = gm.coefficients(self.data, ratio = ratio_optimal, 
                                ref = self.ref, orders = self.nn_orders)[:, self.nn_orders_mask]
            
            # uses interpolation to find the proper ratios for training and testing
            self.interp_f_ratio = interp1d(self.x, ratio_optimal * np.ones(len(self.x)))
            self.ratio_train = self.interp_f_ratio(self.x_train)
            self.coeffs_train = gm.coefficients(self.y_train, ratio = self.ratio_train, 
                                              ref = self.ref_train, 
                                              orders = self.nn_orders)[:, self.nn_orders_mask]
            self.ratio_test = self.interp_f_ratio(self.x_test)
            self.coeffs_test = gm.coefficients(self.y_test, ratio = self.ratio_test, 
                                              ref = self.ref_test, 
                                              orders = self.nn_orders)[:, self.nn_orders_mask]

            # defines the kernel
            if self.fixed_quantity_name == "energy" and \
                    self.fixed_quantity_value < 70.1 and \
                    self.fixed_quantity_value >= 1.:
                self.kernel = RBF(length_scale = mean_list[2], 
                            length_scale_bounds = (mean_list[2], mean_list[2])) + \
                            WhiteKernel(1e-6, noise_level_bounds = 'fixed')
            else:
                self.kernel = RBF(length_scale = mean_list[2], 
                            length_scale_bounds = (mean_list[2], mean_list[2])) + \
                            WhiteKernel(1e-10, noise_level_bounds = 'fixed')
            self.ls = mean_list[2]
            # print(self.kernel)

            # Define the GP
            self.gp = gm.ConjugateGaussianProcess(
                self.kernel, center = self.center, disp = self.disp, df = self.df,
                scale = self.std_est, n_restarts_optimizer = 50, random_state = self.seed, 
                sd = self.sd)
            
            self.coeffs = (self.coeffs.T[self.mask_restricted]).T
            self.coeffs_train = (self.coeffs_train.T[self.mask_restricted]).T
            self.coeffs_test = (self.coeffs_test.T[self.mask_restricted]).T
            
            self.plot_coefficients(whether_save = False)
            self.plot_md(whether_save = False)
            self.plot_pc(whether_save = False)
            
            # except:
            #     print("Error in plotting the curvewise posterior PDF.")

    def plot_lambda_mpi_posterior_curvewise(self, SGT, DSG, AY, A, D, AXX, AYY, t_lab, degrees, 
                        Lambda_b_true, ls_true, mpi_true, orders_num, 
                        ax = None, whether_plot_joint_Lbls = True, 
                        whether_plot_joint_mpils = True, whether_plot_joint_Lbmpi = True, 
                        whether_plot_lambda = True, whether_plot_mpi = True, 
                        whether_plot_corner = True, whether_save = True):
        
        # # functions for interpolating the ratio and reference scale in the TruncationGP
        # # def lambda_interp_f_ref(x_):
        # #     X = np.ravel(x_)
        # #     return self.interp_f_ref(X)
        # def lambda_interp_f_ratio_Lb_tlab(x_, lambda_var):
        #     X = np.ravel(x_)
        #     return interp_f_ratio_Lb_tlab(X) * self.Lambda_b / lambda_var
        # def lambda_interp_f_ratio_Lb_degrees(x_, lambda_var):
        #     X = np.ravel(x_)
        #     return interp_f_ratio_Lb_degrees(X) * self.Lambda_b / lambda_var
        
        # interp_f_ratio_Lb_tlab = interp1d(E_to_p(t_lab, "np"), 
        #         Q_approx(E_to_p(t_lab, "np"), self.Q_param, self.Lambda_b, interaction='np'))
        # # interp_f_ratio_Lb_degrees = interp1d(-1. * np.cos(np.radians(degrees)),
        # #         Q_approx(E_to_p(t_lab_prime_loop, "np"), self.Q_param, self.Lambda_b, interaction='np') * len(degrees))
        
        def interp_f_ratio_Lb_mpi(x_map, x_interp, x_interp_Q, Q_param, mpi_var, lambda_var):
            X = np.ravel(x_map)
        
#         return (interpolate.interp1d(x_interp, Q_approx(x_interp, Q_param, mpi, Lambda_b_true) * Lambda_b / lambda_var))(X)
            return (interp1d( x_interp, Q_approx(x_interp_Q, Q_param, Lambda_b = lambda_var, m_pi = mpi_var) 
                             * np.ones(len(x_interp)) ))(X)
        
        
        # try:
        # t_lab_Lb = np.array([50, 100, 150, 200, 250, 300])
        # t_lab_Lb = np.array([100, 250])
        # t_lab_Lb = np.array([50, 100, 150, 200, 250, 300])
        t_lab_Lb = np.array([5, 21, 48, 85, 133, 192])
        t_lab_Lb_prime = E_to_p(t_lab_Lb, "np")
        # degrees_Lb = np.array([60, 120])
        degrees_Lb = np.array([26, 51, 77, 103, 129, 154])
        # degrees_Lb = np.array([30, 60, 90, 120, 150])
        degrees_Lb_prime = -1. * np.cos(np.radians(degrees_Lb))
        # t_lab_Lb = np.array([96, 143, 200, 300])
        # degrees_Lb = np.array([60, 120])
        # X_Lb = gm.cartesian(t_lab_Lb, degrees_Lb)
        # X_Lb_prime = gm.cartesian(t_lab_Lb_prime, degrees_Lb_prime)
        # print(X_Lb_prime)
        Lb_colors = self.light_colors[-2:]
        # print(self.light_colors)
        # print(Lb_colors)
        # Lambda_b_array = np.arange(1, 1501, 1)
        
        # ratios_sgt_Lb = [Q_approx(E_to_p(t_lab_Lb, "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        # print(np.shape(ratios_sgt_Lb))
        # ratios_dsg_Lb = [Q_approx(E_to_p(X_Lb[:, 0], "np"), self.Q_param, Lb, interaction='np') for Lb in Lambda_b_array]
        # # print(ratios_dsg_Lb[13])
        # logprior = Lb_logprior(Lambda_b_array)
        
        # creates the grid over which the posterior PDF will be plotted
        # self.ls_vals = self.posteriorgrid.x_vals
        # self.lambda_vals = self.posteriorgrid.y_vals
        # lambda_vals = np.linspace(Lambda_b_true / 2.5, Lambda_b_true * 2.5, 51)
        mpi_vals = np.linspace(mpi_true / 3.1, mpi_true * 3.1, 99)
        ls_vals = np.linspace(0.02, 2.00, 100)
        ls_vals_Elab = np.linspace(1, 300, len(ls_vals))
        # mpi_vals = np.array([131, 133, 135, 137, 139, 141, 143, 145])
        # mpi_vals = np.array([193, 195, 197, 199, 201, 203, 205, 207])
        lambda_vals = np.linspace(np.max(mpi_vals), 1500, 101)
        # lambda_vals = np.linspace(590, 650, 10)
        random_vars_list_obs_dict = {'SGT' : [lambda_vals, ls_vals_Elab, mpi_vals], 
                       'DSG' : [lambda_vals, ls_vals, mpi_vals], 
                       'spins' : [lambda_vals, ls_vals, mpi_vals]}
        random_vars_names_list = ['Lambdab', 'ls', 'mpieff']
        
        randoms_vars_true_list = [Lambda_b_true, ls_true, mpi_true]
        
        lambda_logprior = Lb_logprior(lambda_vals)
        mpi_logprior = mpieff_logprior(mpi_vals)
        logpriors_names_list = ['Lbuniformlogprior', None, 'mpieffuniformlogprior']
        
        obs_name_list = ['SGT', 'DSG', 'spins']
        obs_label_list = obs_name_list
        obs_name_data_dict = {'SGT' : [SGT], 
                              'DSG' : [DSG], 
                              'spins' : [A, AY, D, AXX, AYY]}
        
        # data_filename_SGT_nho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'SGT' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        # print("We look for data in the file " + data_filename_SGT_nho + ".")
        # data_filename_SGT_ho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'SGT' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        # data_filename_DSG_nho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'DSG' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        # data_filename_DSG_ho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'DSG' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        # data_filename_spins_nho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'spins' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        # data_filename_spins_ho = 'data/' + \
        #         'Lambdab_ls_mpieff_posterior_pdf_curvewise' + '_' + 'spins' + '_' + \
        #         self.scheme + '_' + self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + \
        #         self.p_param + '_' + self.vs_what + '_' + \
        #         'Lbuniformlogprior' + '_' + \
        #         'mpieffuniformlogprior' + '_' + \
        #         'Lambdab' + str(len(lambda_vals)) + 'pts' + f'{min(lambda_vals):.2f}' + 'to' +  f'{max(lambda_vals):.2f}' + '_' + \
        #         'ls' + str(len(ls_vals)) + 'pts' + f'{min(ls_vals_Elab):.2f}' + 'to' + f'{max(ls_vals_Elab):.2f}' + '_' + \
        #         'mpieff' + str(len(mpi_vals)) + 'pts' + f'{min(mpi_vals):.2f}' + 'to' + f'{max(mpi_vals):.2f}' + \
        #         '.txt'
        
        def make_filename_posterior(
            self,
            folder,
            name_general,
            obs_name,
            order_name,
            input_space_name,
            random_vars_names,
            random_vars_values,
        ):
            filename = (
                str(folder)
                + "/"
                + str(name_general)
                + "_"
                + str(obs_name)
                + "_"
                + str(self.scheme)
                + "_"
                + str(self.scale)
                + "_"
                + str(order_name)
                + "_"
                + "Q"
                + str(self.Q_param)
                + "_"
                + str(self.p_param)
                + "_"
                + str(input_space_name)
            )

            for logprior in logpriors_names_list:
                filename += "_" + str(logprior)
            for (random_var_name, random_var) in zip(random_vars_names_list, random_vars_list_obs_dict[obs_name]):
                filename += (
                    "_"
                    + str(random_var_name)
                    + str(len(random_var))
                    + "pts"
                    + f"{min(random_var):.2f}"
                    + "to"
                    + f"{max(random_var):.2f}"
                )
            return filename.replace("__", "_") + ".txt"
        
        ray.shutdown()
        ray.init()
        
        @ray.remote
        def log_likelihood(gp_fitted, orders, x_interp, x_interp_Q, mesh_points):
            return [gp_fitted.log_marginal_likelihood([pt[1],], 
                        orders_eval = orders,
                        **{"x_interp" : x_interp,
                            "x_interp_Q" : x_interp_Q, 
                            "Q_param" : self.Q_param, 
                            "mpi_var" : pt[2], 
                            "lambda_var" : pt[0]}) for pt in mesh_points]
        
        BATCH_SIZE = 100
        
        likelihood_array = []
        
        for obs_name in range(10):
            for order_counter in range(10):
                order = np.max(self.nn_orders) - orders_num + order_counter
                print("order = " + str(order))
                
                filename = self.make_filename(
                    "data",
                    "Lambdab_posterior_pdf_curvewise",
                    obs_name,
                    self.orders_names_dict[order]
                )
                
                try:
                    np.append(likelihood_array, np.reshape(
                        np.loadtxt(filename),
                        (len(random_var) for random_var in random_vars_list_obs_dict[obs_name]),
                    ))
                
                except:
                    # ANOTHER FOR LOOP
                    
                    # # creates and fits the TruncationGP
                    # self.gp_post = gm.TruncationGP(self.kernel, 
                    #                             ref = lambda_interp_f_ref, 
                    #                             ratio = lambda_interp_f_ratio, 
                    #                             center = self.center, 
                    #                             disp = self.disp, 
                    #                             df = self.df, 
                    #                             scale = self.std_est, 
                    #                             excluded = [0], 
                    #                             ratio_kws = {"lambda_var" : self.Lambda_b})
                    
                    # Mask unused SGT data, and compute results
                    # print(SGT.shape)
                    # print(SGT[self.nn_orders_mask, :].shape)
                    # print((SGT[self.nn_orders_mask, :])[self.mask_restricted, :].shape)
                    sgt_data = (SGT[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb)]
                    # print("sgt_Lb has shape " + str(sgt_Lb.shape))
                    # print(t_lab_Lb)
                    # print(t_lab_Lb_prime)
                    # print(E_to_p(t_lab, "np"))
                    # print(interp_f_ratio_Lb_mpi(t_lab_Lb_prime, E_to_p(t_lab, "np"), self.Q_param, mpi_true, Lambda_b_true))
                    # creates and fits the TruncationGP
                    gp_post_sgt_nho = gm.TruncationGP(self.kernel, 
                                                ref = sgt_data[0], 
                                                ratio = interp_f_ratio_Lb_mpi, 
                                                center = self.center, 
                                                disp = self.disp, 
                                                df = self.df, 
                                                scale = self.std_est, 
                                                excluded = [0], 
                                                ratio_kws = {"x_interp" : E_to_p(t_lab, "np"),
                                                        "x_interp_Q" : E_to_p(t_lab, "np"), 
                                                        "Q_param" : self.Q_param, 
                                                        "mpi_var" : mpi_true,  
                                                        "lambda_var" : Lambda_b_true})
                    gp_post_sgt_nho.fit(t_lab_Lb_prime[:, None],  
                                          sgt_data.T, 
                                          orders = self.nn_orders_full, 
                                          orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
                    gp_post_sgt_ho = gm.TruncationGP(self.kernel, 
                                                ref = sgt_data[0], 
                                                ratio = interp_f_ratio_Lb_mpi, 
                                                center = self.center, 
                                                disp = self.disp, 
                                                df = self.df, 
                                                scale = self.std_est, 
                                                excluded = [0], 
                                                ratio_kws = {"x_interp" : E_to_p(t_lab, "np"),
                                                        "x_interp_Q" : E_to_p(t_lab, "np"), 
                                                        "Q_param" : self.Q_param, 
                                                        "mpi_var" : mpi_true,  
                                                        "lambda_var" : Lambda_b_true})
                    gp_post_sgt_ho.fit(t_lab_Lb_prime[:, None],  
                                          sgt_data.T, 
                                          orders = self.nn_orders_full, 
                                          orders_eval = self.nn_orders[:len(self.nn_orders)])
                    # sgt_Lb_nho_result = compute_posterior_intervals(
                    #     Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
                    #     orders = self.nn_orders, 
                    #     max_idx = max(self.nn_orders) - 2,
                    #     logprior=logprior, Lb=Lambda_b_array)
                    # sgt_Lb_ho_result = compute_posterior_intervals(
                    #     Lb_model, sgt_Lb, ratios_sgt_Lb, ref = sgt_Lb[0], 
                    #     orders = self.nn_orders, 
                    #     max_idx = max(self.nn_orders) - 1,
                    #     logprior = logprior, Lb = Lambda_b_array)
                    
                    # evaluates the probability across the mesh
                    
                    sgt_loglike_nho = np.array([[[
                        gp_post_sgt_nho.log_marginal_likelihood([ls_,], 
                                orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                                **{"x_interp" : E_to_p(t_lab, "np"),
                                    "x_interp_Q" : E_to_p(t_lab, "np"), 
                                    "Q_param" : self.Q_param, 
                                    "mpi_var" : mpi_, 
                                    "lambda_var" : lambda_})
                            for mpi_ in mpi_vals]
                            for ls_ in np.log(ls_vals_Elab)]
                            for lambda_ in lambda_vals])
                    
                    
                    gp_post_nho_ray = ray.put(gp_post_sgt_nho)
                    orders_nho_ray = ray.put(self.nn_orders[:len(self.nn_orders) - 1])
                    t_lab_mom_ray = ray.put(E_to_p(t_lab, "np"))
                    
                    
                    # Create hyperparameter grid
                    mesh_cart = gm.cartesian(lambda_vals, np.log(ls_vals_Elab), mpi_vals)
                    print(np.shape(mesh_cart))
                    print(mesh_cart)
                    
                    log_like_ids = []
                    for i in range(0, len(mesh_cart), BATCH_SIZE):
                        batch = mesh_cart[i : i + BATCH_SIZE]
                        log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                orders_nho_ray, t_lab_mom_ray, t_lab_mom_ray, batch))
                    log_like = list(itertools.chain(*ray.get(log_like_ids)))
                    print(np.shape(log_like))
                    # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                    sgt_loglike_nho = np.reshape(log_like, (len(lambda_vals), len(ls_vals_Elab), len(mpi_vals)))
                    
            
                    # # # adds the log prior to the log likelihood
                    # # ls_lambda_loglike_nho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_nho)[1], 1) ).T
                    # # adds the log prior to the log likelihood
                    # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[2], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 2)))
                    # sgt_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( lambda_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[2], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 2)
                    # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[0], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 0)))
                    # sgt_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( mpi_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[0], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_nho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 0)
                    # print("lambda_ls_mpi_loglike_nho has shape " + str(np.shape(sgt_lambda_ls_mpi_loglike_nho)))
                    
                    # # Makes sure that the values don't get too big or too small
                    # sgt_lambda_ls_mpi_like_nho = np.exp(sgt_lambda_ls_mpi_loglike_nho - np.max(sgt_lambda_ls_mpi_loglike_nho))
            
                    # # # Now compute the marginal distributions
                    # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                    # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
            
                    # # # Normalize them
                    # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                    # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
            
                    # # sgt_Lb_nho_result = lambda_like_nho
                    
                    # # Now compute the marginal distributions
                    # sgt_lambda_like_nho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                    #         x = ls_vals, axis = 1)
                    # sgt_ls_like_nho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                    #         x = lambda_vals, axis = 0)
                    # sgt_mpi_like_nho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_nho, x = ls_vals, axis = 1), 
                    #         x = lambda_vals, axis = 0)
                    # print(np.shape(sgt_lambda_like_nho))
                    # print(np.shape(sgt_ls_like_nho))
                    # print(np.shape(sgt_mpi_like_nho))
                
                    # # Normalize them
                    # sgt_lambda_like_nho /= np.trapz(sgt_lambda_like_nho, x = lambda_vals, axis = 0)
                    # sgt_ls_like_nho /= np.trapz(sgt_ls_like_nho, x = ls_vals, axis = 0)
                    # sgt_mpi_like_nho /= np.trapz(sgt_mpi_like_nho, x = mpi_vals, axis = 0)
                
                    # sgt_Lb_nho_result = sgt_lambda_like_nho
                    # sgt_ls_nho_result = sgt_ls_like_nho
                    # sgt_mpi_nho_result = sgt_mpi_like_nho
                    
                    # ls_lambda_loglike_ho = np.array([[
                    #     gp_post_sgt_Lb_ho.log_marginal_likelihood([ls_,], orders_eval = self.nn_orders[:len(self.nn_orders)],
                    #                                           **{"lambda_var" : lambda_})
                    #         for ls_ in np.log(ls_vals_Lb)]
                    #         for lambda_ in lambda_vals_Lb])
                    
                    # # adds the log prior to the log likelihood
                    # ls_lambda_loglike_ho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_ho)[1], 1) ).T
            
                    # # Makes sure that the values don't get too big or too small
                    # ls_lambda_like_ho = np.exp(ls_lambda_loglike_ho - np.max(ls_lambda_loglike_ho))
            
                    # # Now compute the marginal distributions
                    # lambda_like_ho = np.trapz(ls_lambda_like_ho, x = ls_vals_Lb, axis = -1)
                    # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
            
                    # # Normalize them
                    # lambda_like_ho /= np.trapz(lambda_like_ho, x = lambda_vals_Lb, axis = 0)
                    # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
            
                    # sgt_Lb_ho_result = lambda_like_ho
                    
                    # start_time = time.time()
                    # print("Starting sequential...")
                    # # evaluates the probability across the mesh
                    # sgt_loglike_ho = np.array([[[
                    #     gp_post_sgt_ho.log_marginal_likelihood([ls_,], 
                    #             orders_eval = self.nn_orders[:len(self.nn_orders)],
                    #             **{"x_interp" : E_to_p(t_lab, "np"),
                    #                 "x_interp_Q" : E_to_p(t_lab, "np"), 
                    #                 "Q_param" : self.Q_param, 
                    #                 "mpi_var" : mpi_, 
                    #                 "lambda_var" : lambda_})
                    #         for mpi_ in mpi_vals]
                    #         for ls_ in np.log(ls_vals_Elab)]
                    #         for lambda_ in lambda_vals])
                    # print(f"Time elapsed sequential: {time.time()-start_time:.2f}s")
                    
                    
                    start_time = time.time()
                    print("Starting...")
                    gp_post_ho_ray = ray.put(gp_post_sgt_ho)
                    orders_ho_ray = ray.put(self.nn_orders[:len(self.nn_orders)])
                    t_lab_mom_ray = ray.put(E_to_p(t_lab, "np"))
                    
                    log_like_ids = []
                    for i in range(0, len(mesh_cart), BATCH_SIZE):
                        batch = mesh_cart[i : i + BATCH_SIZE]
                        log_like_ids.append(log_likelihood.remote(gp_post_ho_ray, 
                                orders_ho_ray, t_lab_mom_ray, t_lab_mom_ray, batch))
                    log_like = list(itertools.chain(*ray.get(log_like_ids)))
                    print(np.shape(log_like))
                    # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                    sgt_loglike_ho = np.reshape(log_like, (len(lambda_vals), len(ls_vals_Elab), len(mpi_vals)))
                    print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                    
                    # # # adds the log prior to the log likelihood
                    # # ls_lambda_loglike_nho += np.tile( lambda_logprior, (np.shape(ls_lambda_loglike_nho)[1], 1) ).T
                    # # adds the log prior to the log likelihood
                    # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[2], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 2)))
                    # sgt_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( lambda_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[2], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 2)
                    # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[0], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 0)))
                    # sgt_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( mpi_logprior, 
                    #             (
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[0], 
                    #              np.shape(sgt_lambda_ls_mpi_loglike_ho)[1], 
                    #              1
                    #              ) 
                    #         ), 0, 0)
                    # print("lambda_ls_mpi_loglike_ho has shape " + str(np.shape(sgt_lambda_ls_mpi_loglike_ho)))
                    
                    # # Makes sure that the values don't get too big or too small
                    # sgt_lambda_ls_mpi_like_ho = np.exp(sgt_lambda_ls_mpi_loglike_ho - np.max(sgt_lambda_ls_mpi_loglike_ho))
            
                    # # # Now compute the marginal distributions
                    # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                    # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
            
                    # # # Normalize them
                    # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                    # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
            
                    # # sgt_Lb_nho_result = lambda_like_nho
                    
                    # # Now compute the marginal distributions
                    # sgt_lambda_like_ho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                    #         x = ls_vals, axis = 1)
                    # sgt_ls_like_ho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                    #         x = lambda_vals, axis = 0)
                    # sgt_mpi_like_ho = np.trapz(
                    #     np.trapz(sgt_lambda_ls_mpi_like_ho, x = ls_vals, axis = 1), 
                    #         x = lambda_vals, axis = 0)
                
                    # # Normalize them
                    # sgt_lambda_like_ho /= np.trapz(sgt_lambda_like_ho, x = lambda_vals, axis = 0)
                    # sgt_ls_like_ho /= np.trapz(sgt_ls_like_ho, x = ls_vals, axis = 0)
                    # sgt_mpi_like_ho /= np.trapz(sgt_mpi_like_ho, x = mpi_vals, axis = 0)
                
                    # sgt_Lb_ho_result = sgt_lambda_like_ho
                    # sgt_ls_ho_result = sgt_ls_like_ho
                    # sgt_mpi_ho_result = sgt_mpi_like_ho
                    
                    print("We're finished with the total cross section.")
                        
                    dsg_loglike_nho = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    dsg_loglike_ho = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    spins_loglike_nho = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    spins_loglike_ho = np.zeros((len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    
                    # dsg_Lb_nho_result = np.ones((len(lambda_vals)))
                    # dsg_Lb_ho_result = np.ones((len(lambda_vals)))
                    # dsg_mpi_nho_result = np.ones((len(mpi_vals)))
                    # dsg_mpi_ho_result = np.ones((len(mpi_vals)))
                    
                    # spins_Lb_nho_result = np.ones((len(lambda_vals)))
                    # spins_Lb_ho_result = np.ones((len(lambda_vals)))
                    # spins_mpi_nho_result = np.ones((len(mpi_vals)))
                    # spins_mpi_ho_result = np.ones((len(mpi_vals)))
                    
                    for t_lab_Lb_loop in zip(t_lab_Lb, t_lab_Lb_prime):
                        # print(np.shape(-1. * np.cos(np.radians(degrees))))
                        # print(np.shape(Q_approx(E_to_p(t_lab_Lb_loop[1], "np"), self.Q_param, self.Lambda_b, interaction='np') * np.ones((len(degrees)))))
                        # interp_f_ratio_Lb_degrees = interp1d(-1. * np.cos(np.radians(degrees)),
                        #         Q_approx(E_to_p(t_lab_Lb_loop[1], "np"), self.Q_param, self.Lambda_b, interaction='np') * np.ones((len(degrees))))
                        # # Mask unused DSG data, and compute results
                        dsg_data = np.reshape((DSG[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
                        # print("dsg_Lb has shape " + str(np.shape(dsg_Lb)))
                        # # print("dsg_Lb = " + str(dsg_Lb))
                        # dsg_Lb_nho_result = compute_posterior_intervals(
                        #     Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
                        #     orders = self.nn_orders, 
                        #     max_idx = max(self.nn_orders) - 2,
                        #     logprior = logprior, Lb = Lambda_b_array)
                        # dsg_Lb_ho_result = compute_posterior_intervals(
                        #     Lb_model, dsg_Lb, ratios_dsg_Lb, ref = dsg_Lb[0], 
                        #     orders = self.nn_orders, 
                        #     max_idx = max(self.nn_orders) - 1,
                        #     logprior = logprior, Lb = Lambda_b_array)
                        gp_post_dsg_nho = gm.TruncationGP(self.kernel, 
                                                    ref = dsg_data[0], 
                                                    ratio = interp_f_ratio_Lb_mpi, 
                                                    center = self.center, 
                                                    disp = self.disp, 
                                                    df = self.df, 
                                                    scale = self.std_est, 
                                                    excluded = [0], 
                                                    ratio_kws = {"x_interp" : -1. * np.cos(np.radians(degrees)),
                                                                  "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                                                                "Q_param" : self.Q_param, 
                                                                "mpi_var" : mpi_true,  
                                                                "lambda_var" : Lambda_b_true})
                        gp_post_dsg_nho.fit(degrees_Lb_prime[:, None],  
                                              dsg_data.T, 
                                              orders = self.nn_orders_full, 
                                              orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
                        gp_post_dsg_ho = gm.TruncationGP(self.kernel, 
                                                    ref = dsg_data[0], 
                                                    ratio = interp_f_ratio_Lb_mpi, 
                                                    center = self.center, 
                                                    disp = self.disp, 
                                                    df = self.df, 
                                                    scale = self.std_est, 
                                                    excluded = [0], 
                                                    ratio_kws = {"x_interp" : -1. * np.cos(np.radians(degrees)),
                                                                  "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                                                                "Q_param" : self.Q_param, 
                                                                "mpi_var" : mpi_true,  
                                                                "lambda_var" : Lambda_b_true})
                        gp_post_dsg_ho.fit(degrees_Lb_prime[:, None],  
                                              dsg_data.T, 
                                              orders = self.nn_orders_full, 
                                              orders_eval = self.nn_orders[:len(self.nn_orders)])
                        
                        # # evaluates the probability across the mesh
                        # dsg_loglike_nho += np.array([[[
                        #     gp_post_dsg_nho.log_marginal_likelihood([ls_,], 
                        #                 orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                        #                     **{"x_interp" : -1. * np.cos(np.radians(degrees)),
                        #                         "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                        #                         "Q_param" : self.Q_param, 
                        #                         "mpi_var" : mpi_, 
                        #                         "lambda_var" : lambda_})
                        #         for mpi_ in mpi_vals]
                        #         for ls_ in np.log(ls_vals)]
                        #         for lambda_ in lambda_vals])
                        
                        gp_post_nho_ray = ray.put(gp_post_dsg_nho)
                        x_points_ray = ray.put(-1. * np.cos(np.radians(degrees)))
                        ratio_points_ray = ray.put(E_to_p(t_lab_Lb_loop[1], "np"))
                        
                        # Create hyperparameter grid
                        mesh_cart = gm.cartesian(lambda_vals, np.log(ls_vals), mpi_vals)
                        
                        log_like_ids = []
                        for i in range(0, len(mesh_cart), BATCH_SIZE):
                            batch = mesh_cart[i : i + BATCH_SIZE]
                            log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                    orders_nho_ray, x_points_ray, ratio_points_ray, batch))
                        log_like = list(itertools.chain(*ray.get(log_like_ids)))
                        # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                        dsg_loglike_nho += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                        
                        # # evaluates the probability across the mesh
                        # dsg_loglike_ho += np.array([[[
                        #     gp_post_dsg_ho.log_marginal_likelihood([ls_,], 
                        #                 orders_eval = self.nn_orders[:len(self.nn_orders)],
                        #                     **{"x_interp" : -1. * np.cos(np.radians(degrees)),
                        #                         "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                        #                         "Q_param" : self.Q_param, 
                        #                         "mpi_var" : mpi_, 
                        #                         "lambda_var" : lambda_})
                        #         for mpi_ in mpi_vals]
                        #         for ls_ in np.log(ls_vals)]
                        #         for lambda_ in lambda_vals])
                        
                        gp_post_ho_ray = ray.put(gp_post_dsg_ho)
                        
                        log_like_ids = []
                        for i in range(0, len(mesh_cart), BATCH_SIZE):
                            batch = mesh_cart[i : i + BATCH_SIZE]
                            log_like_ids.append(log_likelihood.remote(gp_post_ho_ray, 
                                    orders_ho_ray, x_points_ray, ratio_points_ray, batch))
                        log_like = list(itertools.chain(*ray.get(log_like_ids)))
                        # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                        dsg_loglike_ho += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                        
                        spin_obs_list = [AY, A, D, AXX, AYY]
                        
                        gp_fits_spins_nho = []
                        gp_fits_spins_ho = []
                        
                        for so, spin_obs in enumerate(spin_obs_list):
                            spin_data = np.reshape((spin_obs[self.raw_data_mask, :])[:, np.isin(t_lab, t_lab_Lb_loop[0])][..., np.isin(degrees, degrees_Lb)], (len(self.nn_orders), -1))
                            gp_fits_spins_nho.append(gm.TruncationGP(self.kernel, 
                                                        ref = np.ones((len(degrees_Lb))), 
                                                        ratio = interp_f_ratio_Lb_mpi, 
                                                        center = self.center, 
                                                        disp = self.disp, 
                                                        df = self.df, 
                                                        scale = self.std_est, 
                                                        excluded = [0], 
                                                        ratio_kws = {"x_interp" : -1. * np.cos(np.radians(degrees)),
                                                                      "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                                                                    "Q_param" : self.Q_param, 
                                                                    "mpi_var" : mpi_true,  
                                                                    "lambda_var" : Lambda_b_true}))
                            gp_fits_spins_nho[so].fit(degrees_Lb_prime[:, None],  
                                                  spin_data.T, 
                                                  orders = self.nn_orders_full, 
                                                  orders_eval = self.nn_orders[:len(self.nn_orders) - 1])
                            gp_fits_spins_ho.append(gm.TruncationGP(self.kernel, 
                                                        ref = np.ones((len(degrees_Lb))), 
                                                        ratio = interp_f_ratio_Lb_mpi, 
                                                        center = self.center, 
                                                        disp = self.disp, 
                                                        df = self.df, 
                                                        scale = self.std_est, 
                                                        excluded = [0], 
                                                        ratio_kws = {"x_interp" : -1. * np.cos(np.radians(degrees)),
                                                                      "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                                                                    "Q_param" : self.Q_param, 
                                                                    "mpi_var" : mpi_true,  
                                                                    "lambda_var" : Lambda_b_true}))
                            gp_fits_spins_ho[so].fit(degrees_Lb_prime[:, None],  
                                                  spin_data.T, 
                                                  orders = self.nn_orders_full, 
                                                  orders_eval = self.nn_orders[:len(self.nn_orders)])
                        
                        for gp_fit_spins in gp_fits_spins_nho:
                            # # evaluates the probability across the mesh
                            # spins_loglike_nho += np.array([[[
                            #     gp_fit_spins.log_marginal_likelihood([ls_,], 
                            #                     orders_eval = self.nn_orders[:len(self.nn_orders) - 1],
                            #                     **{"x_interp" : -1. * np.cos(np.radians(degrees)),
                            #                       "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                            #                       "Q_param" : self.Q_param, 
                            #                       "mpi_var" : mpi_, 
                            #                       "lambda_var" : lambda_})
                            #         for mpi_ in mpi_vals]
                            #         for ls_ in np.log(ls_vals)]
                            #         for lambda_ in lambda_vals])
                            gp_post_nho_ray = ray.put(gp_fit_spins)
                            
                            log_like_ids = []
                            for i in range(0, len(mesh_cart), BATCH_SIZE):
                                batch = mesh_cart[i : i + BATCH_SIZE]
                                log_like_ids.append(log_likelihood.remote(gp_post_nho_ray, 
                                        orders_nho_ray, x_points_ray, ratio_points_ray, batch))
                            log_like = list(itertools.chain(*ray.get(log_like_ids)))
                            # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                            spins_loglike_nho += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                            
                        
                        for gp_fit_spins in gp_fits_spins_ho:
                            # # evaluates the probability across the mesh
                            # spins_loglike_ho += np.array([[[
                            #     gp_fit_spins.log_marginal_likelihood([ls_,], 
                            #                     orders_eval = self.nn_orders[:len(self.nn_orders)],
                            #                     **{"x_interp" : -1. * np.cos(np.radians(degrees)),
                            #                       "x_interp_Q" : E_to_p(t_lab_Lb_loop[1], "np"), 
                            #                       "Q_param" : self.Q_param, 
                            #                       "mpi_var" : mpi_, 
                            #                       "lambda_var" : lambda_})
                            #         for mpi_ in mpi_vals]
                            #         for ls_ in np.log(ls_vals)]
                            #         for lambda_ in lambda_vals])
                            gp_post_ho_ray = ray.put(gp_fit_spins)
                            
                            log_like_ids = []
                            for i in range(0, len(mesh_cart), BATCH_SIZE):
                                batch = mesh_cart[i : i + BATCH_SIZE]
                                log_like_ids.append(log_likelihood.remote(gp_post_ho_ray, 
                                        orders_ho_ray, x_points_ray, ratio_points_ray, batch))
                            log_like = list(itertools.chain(*ray.get(log_like_ids)))
                            # print(f"Time elapsed parallel: {time.time() - start_time:.2f}s")
                            spins_loglike_ho += np.reshape(log_like, (len(lambda_vals), len(ls_vals), len(mpi_vals)))
                    
                    # # # Normalize them
                    # # dsg_Lb_ho_result /= np.trapz(dsg_Lb_ho_result, x = lambda_vals_Lb, axis = 0)
                    # # dsg_Lb_nho_result /= np.trapz(dsg_Lb_nho_result, x = lambda_vals_Lb, axis = 0)
                    # # spins_Lb_ho_result /= np.trapz(spins_Lb_ho_result, x = lambda_vals_Lb, axis = 0)
                    # # spins_Lb_ho_result /= np.trapz(spins_Lb_nho_result, x = lambda_vals_Lb, axis = 0)
                    
                    # # Gather the above results
                    # # results = [
                    # #     sgt_Lb_nho_result, sgt_Lb_ho_result,
                    # #     dsg_Lb_nho_result, dsg_Lb_ho_result,
                    # #     spins_Lb_nho_result, spins_Lb_ho_result
                    # # ]
                    
                    loglike_list = [sgt_loglike_nho, sgt_loglike_ho, 
                                    dsg_loglike_nho, dsg_loglike_ho, 
                                    spins_loglike_nho, spins_loglike_ho]
                    
                    for loglike in loglike_list:
                        # adds the log prior to the log likelihood
                        print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                                    (
                                      np.shape(loglike)[2], 
                                      np.shape(loglike)[1], 
                                      1
                                      ) 
                                ), 0, 2)))
                        loglike += np.swapaxes(np.tile( lambda_logprior, 
                                    (
                                      np.shape(loglike)[2], 
                                      np.shape(loglike)[1], 
                                      1
                                      ) 
                                ), 0, 2)
                        print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                                    (
                                      np.shape(loglike)[0], 
                                      np.shape(loglike)[1], 
                                      1
                                      ) 
                                ), 0, 0)))
                        loglike += np.swapaxes(np.tile( mpi_logprior, 
                                    (
                                      np.shape(loglike)[0], 
                                      np.shape(loglike)[1], 
                                      1
                                      ) 
                                ), 0, 0)
                        
                    # Makes sure that the values don't get too big or too small
                    like_list = [np.exp(loglike - np.max(loglike)) for loglike in loglike_list]
                    
                    # np.savetxt(data_filename_SGT_nho, 
                    #            np.reshape(like_list[0], (len(lambda_vals) * len(ls_vals_Elab) * len(mpi_vals))))
                    # np.savetxt(data_filename_SGT_ho, 
                    #            np.reshape(like_list[1], (len(lambda_vals) * len(ls_vals_Elab) * len(mpi_vals))))
                    # np.savetxt(data_filename_DSG_nho, 
                    #            np.reshape(like_list[2], (len(lambda_vals) * len(ls_vals) * len(mpi_vals))))
                    # np.savetxt(data_filename_DSG_ho, 
                    #            np.reshape(like_list[3], (len(lambda_vals) * len(ls_vals) * len(mpi_vals))))
                    # np.savetxt(data_filename_spins_nho, 
                    #            np.reshape(like_list[4], (len(lambda_vals) * len(ls_vals) * len(mpi_vals))))
                    # np.savetxt(data_filename_spins_ho, 
                    #            np.reshape(like_list[5], (len(lambda_vals) * len(ls_vals) * len(mpi_vals))))
            
                # # Now compute the marginal distributions
                # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
        
                # # Normalize them
                # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
        
                # sgt_Lb_nho_result = lambda_like_nho
                
                # Now compute the marginal distributions
                sgt_lambda_post_nho = np.trapz(
                    np.trapz(like_list[0], x = mpi_vals, axis = 2), 
                        x = ls_vals_Elab, axis = 1)
                sgt_ls_post_nho = np.trapz(
                    np.trapz(like_list[0], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                sgt_mpi_post_nho = np.trapz(
                    np.trapz(like_list[0], x = ls_vals_Elab, axis = 1), 
                        x = lambda_vals, axis = 0)
                sgt_lambda_post_ho = np.trapz(
                    np.trapz(like_list[1], x = mpi_vals, axis = 2), 
                        x = ls_vals_Elab, axis = 1)
                sgt_ls_post_ho = np.trapz(
                    np.trapz(like_list[1], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                sgt_mpi_post_ho = np.trapz(
                    np.trapz(like_list[1], x = ls_vals_Elab, axis = 1), 
                        x = lambda_vals, axis = 0)
                
                dsg_lambda_post_nho = np.trapz(
                    np.trapz(like_list[2], x = mpi_vals, axis = 2), 
                        x = ls_vals, axis = 1)
                dsg_ls_post_nho = np.trapz(
                    np.trapz(like_list[2], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                dsg_mpi_post_nho = np.trapz(
                    np.trapz(like_list[2], x = ls_vals, axis = 1), 
                        x = lambda_vals, axis = 0)
                dsg_lambda_post_ho = np.trapz(
                    np.trapz(like_list[3], x = mpi_vals, axis = 2), 
                        x = ls_vals, axis = 1)
                dsg_ls_post_ho = np.trapz(
                    np.trapz(like_list[3], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                dsg_mpi_post_ho = np.trapz(
                    np.trapz(like_list[3], x = ls_vals, axis = 1), 
                        x = lambda_vals, axis = 0)
                
                spins_lambda_post_nho = np.trapz(
                    np.trapz(like_list[4], x = mpi_vals, axis = 2), 
                        x = ls_vals, axis = 1)
                spins_ls_post_nho = np.trapz(
                    np.trapz(like_list[4], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                spins_mpi_post_nho = np.trapz(
                    np.trapz(like_list[4], x = ls_vals, axis = 1), 
                        x = lambda_vals, axis = 0)
                spins_lambda_post_ho = np.trapz(
                    np.trapz(like_list[5], x = mpi_vals, axis = 2), 
                        x = ls_vals, axis = 1)
                spins_ls_post_ho = np.trapz(
                    np.trapz(like_list[5], x = mpi_vals, axis = 2), 
                        x = lambda_vals, axis = 0)
                spins_mpi_post_ho = np.trapz(
                    np.trapz(like_list[5], x = ls_vals, axis = 1), 
                        x = lambda_vals, axis = 0)
            
                # Normalize them
                sgt_lambda_post_nho /= np.trapz(sgt_lambda_post_nho, x = lambda_vals, axis = 0)
                sgt_ls_post_nho /= np.trapz(sgt_ls_post_nho, x = ls_vals_Elab, axis = 0)
                sgt_mpi_post_nho /= np.trapz(sgt_mpi_post_nho, x = mpi_vals, axis = 0)
                sgt_lambda_post_ho /= np.trapz(sgt_lambda_post_ho, x = lambda_vals, axis = 0)
                sgt_ls_post_ho /= np.trapz(sgt_ls_post_ho, x = ls_vals_Elab, axis = 0)
                sgt_mpi_post_ho /= np.trapz(sgt_mpi_post_ho, x = mpi_vals, axis = 0)
                
                dsg_lambda_post_nho /= np.trapz(dsg_lambda_post_nho, x = lambda_vals, axis = 0)
                dsg_ls_post_nho /= np.trapz(dsg_ls_post_nho, x = ls_vals, axis = 0)
                dsg_mpi_post_nho /= np.trapz(dsg_mpi_post_nho, x = mpi_vals, axis = 0)
                dsg_lambda_post_ho /= np.trapz(dsg_lambda_post_ho, x = lambda_vals, axis = 0)
                dsg_ls_post_ho /= np.trapz(dsg_ls_post_ho, x = ls_vals, axis = 0)
                dsg_mpi_post_ho /= np.trapz(dsg_mpi_post_ho, x = mpi_vals, axis = 0)
                
                spins_lambda_post_nho /= np.trapz(spins_lambda_post_nho, x = lambda_vals, axis = 0)
                spins_ls_post_nho /= np.trapz(spins_ls_post_nho, x = ls_vals, axis = 0)
                spins_mpi_post_nho /= np.trapz(spins_mpi_post_nho, x = mpi_vals, axis = 0)
                spins_lambda_post_ho /= np.trapz(spins_lambda_post_ho, x = lambda_vals, axis = 0)
                spins_ls_post_ho /= np.trapz(spins_ls_post_ho, x = ls_vals, axis = 0)
                spins_mpi_post_ho /= np.trapz(spins_mpi_post_ho, x = mpi_vals, axis = 0)
                    
                # # adds the log prior to the log likelihood
                # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[2], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)))
                # dsg_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[2], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)
                # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[0], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)))
                # dsg_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[0], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)
                # print("lambda_ls_mpi_loglike_nho has shape " + str(np.shape(dsg_lambda_ls_mpi_loglike_nho)))
                
                # # Makes sure that the values don't get too big or too small
                # dsg_lambda_ls_mpi_like_nho = np.exp(dsg_lambda_ls_mpi_loglike_nho - np.max(dsg_lambda_ls_mpi_loglike_nho))
        
                # # # Now compute the marginal distributions
                # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
        
                # # # Normalize them
                # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
        
                # # sgt_Lb_nho_result = lambda_like_nho
                
                # # Now compute the marginal distributions
                # dsg_lambda_like_nho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                #         x = ls_vals, axis = 1)
                # dsg_ls_like_nho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                #         x = lambda_vals, axis = 0)
                # dsg_mpi_like_nho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_nho, x = ls_vals, axis = 1), 
                #         x = lambda_vals, axis = 0)
                # print(np.shape(dsg_lambda_like_nho))
                # print(np.shape(dsg_ls_like_nho))
                # print(np.shape(dsg_mpi_like_nho))
            
                # # Normalize them
                # dsg_lambda_like_nho /= np.trapz(dsg_lambda_like_nho, x = lambda_vals, axis = 0)
                # dsg_ls_like_nho /= np.trapz(dsg_ls_like_nho, x = ls_vals, axis = 0)
                # dsg_mpi_like_nho /= np.trapz(dsg_mpi_like_nho, x = mpi_vals, axis = 0)
        
                # # dsg_Lb_nho_result += dsg_lambda_like_nho
                # # dsg_mpi_nho_result += dsg_mpi_like_nho
                # dsg_Lb_nho_result += dsg_lambda_like_nho
                # dsg_ls_nho_result += dsg_ls_like_nho
                # dsg_mpi_nho_result += dsg_mpi_like_nho
                
                # # adds the log prior to the log likelihood
                # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[2], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)))
                # dsg_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[2], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)
                # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[0], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)))
                # dsg_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[0], 
                #               np.shape(dsg_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)
                # print("lambda_ls_mpi_loglike_ho has shape " + str(np.shape(dsg_lambda_ls_mpi_loglike_ho)))
                
                # # Makes sure that the values don't get too big or too small
                # dsg_lambda_ls_mpi_like_ho = np.exp(dsg_lambda_ls_mpi_loglike_ho - np.max(dsg_lambda_ls_mpi_loglike_ho))
        
                # # # Now compute the marginal distributions
                # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
        
                # # # Normalize them
                # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
        
                # # sgt_Lb_nho_result = lambda_like_nho
                
                # # Now compute the marginal distributions
                # dsg_lambda_like_ho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                #         x = ls_vals, axis = 1)
                # dsg_ls_like_ho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                #         x = lambda_vals, axis = 0)
                # dsg_mpi_like_ho = np.trapz(
                #     np.trapz(dsg_lambda_ls_mpi_like_ho, x = ls_vals, axis = 1), 
                #         x = lambda_vals, axis = 0)
            
                # # Normalize them
                # dsg_lambda_like_ho /= np.trapz(dsg_lambda_like_ho, x = lambda_vals, axis = 0)
                # dsg_ls_like_ho /= np.trapz(dsg_ls_like_ho, x = ls_vals, axis = 0)
                # dsg_mpi_like_ho /= np.trapz(dsg_mpi_like_ho, x = mpi_vals, axis = 0)
        
                # dsg_Lb_ho_result += dsg_lambda_like_ho
                # dsg_ls_ho_result += dsg_ls_like_ho
                # dsg_mpi_ho_result += dsg_mpi_like_ho
                # # dsg_Lb_ho_result *= dsg_lambda_like_ho
                # # dsg_mpi_ho_result *= dsg_mpi_like_ho
                
                # # adds the log prior to the log likelihood
                # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[2], 
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)))
                # spins_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[2], 
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)
                # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[0], 
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)))
                # spins_lambda_ls_mpi_loglike_nho += np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[0], 
                #               np.shape(spins_lambda_ls_mpi_loglike_nho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)
                # print("lambda_ls_mpi_loglike_nho has shape " + str(np.shape(spins_lambda_ls_mpi_loglike_nho)))
                
                # # Makes sure that the values don't get too big or too small
                # spins_lambda_ls_mpi_like_nho = np.exp(spins_lambda_ls_mpi_loglike_nho - np.max(spins_lambda_ls_mpi_loglike_nho))
        
                # # # Now compute the marginal distributions
                # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
        
                # # # Normalize them
                # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
        
                # # sgt_Lb_nho_result = lambda_like_nho
                
                # # Now compute the marginal distributions
                # spins_lambda_like_nho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                #         x = ls_vals, axis = -1)
                # spins_ls_like_nho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_nho, x = mpi_vals, axis = 2), 
                #         x = lambda_vals, axis = 0)
                # spins_mpi_like_nho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_nho, x = lambda_vals, axis = 0), 
                #         x = ls_vals, axis = 0)
                # print(np.shape(spins_lambda_like_nho))
                # print(np.shape(spins_ls_like_nho))
                # print(np.shape(spins_mpi_like_nho))
            
                # # Normalize them
                # spins_lambda_like_nho /= np.trapz(spins_lambda_like_nho, x = lambda_vals, axis = 0)
                # spins_ls_like_nho /= np.trapz(spins_ls_like_nho, x = ls_vals, axis = 0)
                # spins_mpi_like_nho /= np.trapz(spins_mpi_like_nho, x = mpi_vals, axis = 0)
        
                # spins_Lb_nho_result += spins_lambda_like_nho
                # spins_ls_nho_result += spins_ls_like_nho
                # spins_mpi_nho_result += spins_mpi_like_nho
                # # spins_Lb_nho_result *= spins_lambda_like_nho
                # # spins_mpi_nho_result *= spins_mpi_like_nho
                
                # # adds the log prior to the log likelihood
                # print(np.shape(np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[2], 
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)))
                # spins_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( lambda_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[2], 
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 2)
                # print(np.shape(np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[0], 
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)))
                # spins_lambda_ls_mpi_loglike_ho += np.swapaxes(np.tile( mpi_logprior, 
                #             (
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[0], 
                #               np.shape(spins_lambda_ls_mpi_loglike_ho)[1], 
                #               1
                #               ) 
                #         ), 0, 0)
                # print("lambda_ls_mpi_loglike_ho has shape " + str(np.shape(spins_lambda_ls_mpi_loglike_ho)))
                
                # # Makes sure that the values don't get too big or too small
                # spins_lambda_ls_mpi_like_ho = np.exp(spins_lambda_ls_mpi_loglike_ho - np.max(spins_lambda_ls_mpi_loglike_ho))
        
                # # # Now compute the marginal distributions
                # # lambda_like_nho = np.trapz(ls_lambda_like_nho, x = ls_vals_Lb, axis = -1)
                # # # self.ls_like = np.trapz(self.ls_lambda_like, x = self.lambda_vals, axis = 0)
        
                # # # Normalize them
                # # lambda_like_nho /= np.trapz(lambda_like_nho, x = lambda_vals_Lb, axis = 0)
                # # # self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)
        
                # # sgt_Lb_nho_result = lambda_like_nho
                
                # # Now compute the marginal distributions
                # spins_lambda_like_ho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                #         x = ls_vals, axis = -1)
                # spins_ls_like_ho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_ho, x = mpi_vals, axis = 2), 
                #         x = lambda_vals, axis = 0)
                # spins_mpi_like_ho = np.trapz(
                #     np.trapz(spins_lambda_ls_mpi_like_ho, x = lambda_vals, axis = 0), 
                #         x = ls_vals, axis = 0)
            
                # # Normalize them
                # spins_lambda_like_ho /= np.trapz(spins_lambda_like_ho, x = lambda_vals, axis = 0)
                # spins_ls_like_ho /= np.trapz(spins_ls_like_ho, x = ls_vals, axis = 0)
                # spins_mpi_like_ho /= np.trapz(spins_mpi_like_ho, x = mpi_vals, axis = 0)
        
                # spins_Lb_ho_result += spins_lambda_like_ho
                # spins_ls_ho_result += spins_ls_like_ho
                # spins_mpi_ho_result += spins_mpi_like_ho
                # # spins_Lb_ho_result *= spins_lambda_like_ho
                # # spins_ls_ho_result *= spins_ls_like_ho
                # # spins_mpi_ho_result *= spins_mpi_like_ho
                
                # # Normalize them one last time
                # dsg_Lb_ho_result /= np.trapz(dsg_Lb_ho_result, x = lambda_vals, axis = 0)
                # dsg_Lb_nho_result /= np.trapz(dsg_Lb_nho_result, x = lambda_vals, axis = 0)
                # dsg_mpi_ho_result /= np.trapz(dsg_mpi_ho_result, x = mpi_vals, axis = 0)
                # dsg_mpi_nho_result /= np.trapz(dsg_mpi_nho_result, x = mpi_vals, axis = 0)
                
                # spins_Lb_ho_result /= np.trapz(spins_Lb_ho_result, x = lambda_vals, axis = 0)
                # spins_Lb_nho_result /= np.trapz(spins_Lb_nho_result, x = lambda_vals, axis = 0)
                # spins_mpi_ho_result /= np.trapz(spins_mpi_ho_result, x = mpi_vals, axis = 0)
                # spins_mpi_nho_result /= np.trapz(spins_mpi_nho_result, x = mpi_vals, axis = 0)
                
                # results_lambda = [sgt_Lb_nho_result, sgt_Lb_ho_result, 
                #                   dsg_Lb_nho_result, dsg_Lb_ho_result, 
                #                   spins_Lb_nho_result, spins_Lb_ho_result]
                # results_ls = [sgt_ls_nho_result, sgt_ls_ho_result, 
                #                   dsg_ls_nho_result, dsg_ls_ho_result, 
                #                   spins_ls_nho_result, spins_ls_ho_result]
                # results_mpi = [sgt_mpi_nho_result, sgt_mpi_ho_result, 
                #                 dsg_mpi_nho_result, dsg_mpi_ho_result, 
                #                 spins_mpi_nho_result, spins_mpi_ho_result]
                results_joint_lambdampi = [np.trapz(dist, x = ls_vals, axis = 1) for dist in like_list]
                for dist_idx, dist in enumerate(like_list):
                    if dist_idx == 0 or dist_idx == 1:
                        results_joint_lambdampi[dist_idx] = np.trapz(dist, x = ls_vals_Elab, axis = 1)
                results_joint_lslambda = [np.trapz(dist, x = mpi_vals, axis = 2).T for dist in like_list]
                results_joint_lsmpi = [np.trapz(dist, x = lambda_vals, axis = 0) for dist in like_list]
                
                # norm_joint_lambdampi = [np.trapz(np.trapz(dist, x = lambda_vals, axis = 0), x = mpi_vals, axis = 0) for dist in results_joint_lambdampi]
                # norm_joint_lslambda = [np.trapz(np.trapz(dist, x = ls_vals, axis = 0), x = lambda_vals, axis = 0) for dist in results_joint_lslambda]
                # norm_joint_lsmpi = [np.trapz(np.trapz(dist, x = ls_vals, axis = 0), x = mpi_vals, axis = 0) for dist in results_joint_lsmpi]
                
                results_joint_lambdampi = [dist / np.amax(dist) for dist in results_joint_lambdampi]
                results_joint_lslambda = [dist / np.amax(dist) for dist in results_joint_lslambda]
                results_joint_lsmpi = [dist / np.amax(dist) for dist in results_joint_lsmpi]
                
                results_lambda = [sgt_lambda_post_nho, sgt_lambda_post_ho, 
                                  dsg_lambda_post_nho, dsg_lambda_post_ho, 
                                  spins_lambda_post_nho, spins_lambda_post_ho]
                results_ls = [sgt_ls_post_nho, sgt_ls_post_ho, 
                                  dsg_ls_post_nho, dsg_ls_post_ho, 
                                  spins_ls_post_nho, spins_ls_post_ho]
                results_mpi = [sgt_mpi_post_nho, sgt_mpi_post_ho, 
                                  dsg_mpi_post_nho, dsg_mpi_post_ho, 
                                  spins_mpi_post_nho, spins_mpi_post_ho]
                # results_lambda = [sgt_Lb_nho_result, sgt_Lb_ho_result, 
                #                   dsg_Lb_nho_result, dsg_Lb_ho_result]
                # results_mpi = [sgt_mpi_nho_result, sgt_mpi_ho_result, 
                #                dsg_mpi_nho_result, dsg_mpi_ho_result]
                
                # for i, (posterior, bounds, median) in enumerate(results):
                if whether_plot_lambda:
                    # Plot each posterior and its summary statistics
                    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
                    
                    for i, posterior_raw in enumerate(results_lambda):
                        posterior = posterior_raw / (1.2*np.max(posterior_raw))  # Scale so they're all the same height
                        # Make the lines taper off
                        # print(np.shape(lambda_vals))
                        # print(np.shape(posterior))
                        Lb_vals = lambda_vals[posterior > 1e-2]
                        posterior = posterior[posterior > 1e-2]
                        # Plot and fill posterior, and add summary statistics
                        ax.plot(Lb_vals, posterior-i, c='gray')
                        
                        # if i == 0: pdf_label = self.orders_dict[(np.sort(self.nn_orders))[-2]]
                        # elif i == 1: pdf_label = self.orders_dict[max(self.nn_orders)]
                        # else: pdf_label = '_nolegend_'
                            
                        ax.fill_between(Lb_vals, -i, posterior-i, facecolor=Lb_colors[i % 2])
                        # draw_summary_statistics(*bounds, median, ax=ax, height=-i)
                        
                        bounds = np.zeros((2,2))
                        for j, p in enumerate([0.68, 0.95]):
                            # bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
                            bounds[j] = gm.hpd_pdf(pdf=posterior_raw, alpha=p, x=lambda_vals)
                            # bounds[j] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb_vals)
            
                        median = gm.median_pdf(pdf=posterior_raw, x=lambda_vals)
                        # median = gm.median_pdf(pdf=posterior, x=Lb_vals)
            
                        draw_summary_statistics(*bounds, median, ax=ax, height=-i)
                    
                    # Plot formatting
                    ax.set_yticks([-0, -2, -4])
                    ax.set_yticks([-1.1, -3.1], minor=True)
                    ax.set_yticklabels([r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$'])
                    ax.tick_params(axis='both', which='both', direction='in')
                    ax.tick_params(which='major', length=0)
                    ax.tick_params(which='minor', length=7, right=True)
                    ax.set_xlim(0, 1200)
                    ax.set_xticks([0, 300, 600, 900, 1200])
                    ax.set_xlabel(r'$\Lambda_b$ (MeV)')
                    ax.legend(title = r'$\mathrm{pr}(\Lambda_{b} \, | \, \vec{\mathbf{y}}_{k}, m_{\pi}, \ell, \mathbf{f})$', 
                              handles = [Patch(facecolor=Lb_colors[0], 
                                               edgecolor='gray', 
                                               linewidth = 1, 
                                               label=self.orders_dict[(np.sort(self.nn_orders))[-2]]), 
                                         Patch(facecolor=Lb_colors[1], 
                                               edgecolor='gray',  
                                               linewidth = 1, 
                                               label=self.orders_dict[max(self.nn_orders)])])
                    ax.grid(axis='x')
                    ax.set_axisbelow(True)
                    
                    if 'fig' in locals() and whether_save:
                        fig.tight_layout()
                
                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                'Lambdab_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                    self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                                '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                                self.train_pts_loc + '_' + self.p_param + 
                                self.filename_addendum).replace('_0MeVlab_', '_'))
                    
                if whether_plot_mpi:
                    # Plot each posterior and its summary statistics
                    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
                    
                    for i, posterior_raw in enumerate(results_mpi):
                        posterior = posterior_raw / (1.2*np.max(posterior_raw))  # Scale so they're all the same height
                        # Make the lines taper off
                        mpi_eff_vals = mpi_vals[posterior > 1e-2]
                        posterior = posterior[posterior > 1e-2]
                        # Plot and fill posterior, and add summary statistics
                        ax.plot(mpi_eff_vals, posterior-i, c='gray')
                        
                        # if i == 0: pdf_label = self.orders_dict[(np.sort(self.nn_orders))[-2]]
                        # elif i == 1: pdf_label = self.orders_dict[max(self.nn_orders)]
                        # else: pdf_label = '_nolegend_'
                            
                        ax.fill_between(mpi_eff_vals, -i, posterior-i, facecolor=Lb_colors[i % 2])
                        # draw_summary_statistics(*bounds, median, ax=ax, height=-i)
                        
                        bounds = np.zeros((2,2))
                        for j, p in enumerate([0.68, 0.95]):
                            # bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
                            bounds[j] = gm.hpd_pdf(pdf=posterior_raw, alpha=p, x=mpi_vals)
                            # bounds[j] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb_vals)
            
                        median = gm.median_pdf(pdf=posterior_raw, x=mpi_vals)
                        # median = gm.median_pdf(pdf=posterior, x=Lb_vals)
            
                        draw_summary_statistics(*bounds, median, ax=ax, height=-i)
                    
                    # Plot formatting
                    ax.set_yticks([-0, -2, -4])
                    ax.set_yticks([-1.1, -3.1], minor=True)
                    ax.set_yticklabels([r'$\sigma$', r'$\displaystyle\frac{d\sigma}{d\Omega}$', r'$X_{pqik}$'])
                    ax.tick_params(axis='both', which='both', direction='in')
                    ax.tick_params(which='major', length=0)
                    ax.tick_params(which='minor', length=7, right=True)
                    ax.set_xlim(0, 300)
                    ax.set_xticks([50, 100, 150, 200, 250, 300, 350])
                    ax.set_xlabel(r'$m_{\pi}$ (MeV)')
                    ax.legend(title = r'$\mathrm{pr}(m_{\pi} \, | \, \vec{\mathbf{y}}_{k}, \Lambda_{b}, \ell, \mathbf{f})$', 
                              handles = [Patch(facecolor=Lb_colors[0], 
                                               edgecolor='gray', 
                                               linewidth = 1, 
                                               label=self.orders_dict[(np.sort(self.nn_orders))[-2]]), 
                                         Patch(facecolor=Lb_colors[1], 
                                               edgecolor='gray',  
                                               linewidth = 1, 
                                               label=self.orders_dict[max(self.nn_orders)])])
                    ax.grid(axis='x')
                    ax.set_axisbelow(True)
                    
                    if 'fig' in locals() and whether_save:
                        fig.tight_layout()
                
                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                'mpieff_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                    self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
                                '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
                                self.train_pts_loc + '_' + self.p_param + 
                                self.filename_addendum).replace('_0MeVlab_', '_'))
                
            #     if whether_plot_joint_Lbmpi:
            #         with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
            #             cmap_name = 'Blues'
            #             cmap = mpl.cm.get_cmap(cmap_name)
                        
            #             for posterior in zip(results_joint_lambdabmpi, results_mpi, results_lambda):
            #                 # Setup axes
            # #                 if ax_joint == None and ax_marg_x == None and ax_marg_y == None:
            #                 fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(ratio=5, height=3.4)
        
            #                 # Plot contour
            #                 ax_joint.contour(mpi_vals, lambda_vals, posterior[0],
            #                                   levels=[np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999],
            #                                   cmap=cmap_name, vmin=-0.05, vmax=0.8, zorder=1)
        
            #                 # Now plot the marginal distributions
            #                 ax_marg_y.plot(posterior[2], lambda_vals, c=cmap(0.8), lw=1)
            #                 ax_marg_y.fill_betweenx(lambda_vals, np.zeros_like(posterior[2]),
            #                                         posterior[2], facecolor=cmap(0.2), lw=1)
            #                 ax_marg_x.plot(mpi_vals, posterior[1], c=cmap(0.8), lw=1)
            #                 ax_marg_x.fill_between(mpi_vals, np.zeros_like(mpi_vals),
            #                                         posterior[1], facecolor=cmap(0.2), lw=1)
        
            #                 # Formatting
            #                 ax_joint.set_xlabel(r'$m_{\pi}$ (MeV)')
            #                 ax_joint.set_ylabel(r'$\Lambda_{\mathrm{b}}$ (MeV)')
            #                 ax_joint.axvline(mpi_true, 0, 1, c=gray, lw=1, zorder=0)
            #                 ax_joint.axhline(Lambda_b_true, 0, 1, c=gray, lw=1, zorder=0)
            #                 ax_joint.margins(x=0, y=0.)
            #                 ax_joint.set_xlim(min(mpi_vals), max(mpi_vals))
            #                 ax_joint.set_ylim(min(lambda_vals), max(lambda_vals))
            #                 ax_marg_x.set_ylim(bottom=0);
            #                 ax_marg_y.set_xlim(left=0);
            #                 ax_joint.text(0.95, 0.95, r'pr$(m_{\pi}, \Lambda_{b} \,|\, \vec{\mathbf{y}}_k, \ell)$', ha='right', va='top',
            #                               transform=ax_joint.transAxes,
            #                               bbox=text_bbox
            #                               )
            #                 plt.show()
                            
            #                 if 'fig' in locals() and whether_save:
            #                     fig.tight_layout()
                        
            #                     fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
            #                             'joint_Lambdab_mpieff_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
            #                                 self.scale + '_Q' + self.Q_param + '_' + self.vs_what + 
            #                             '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + 
            #                             self.train_pts_loc + '_' + self.p_param + 
            #                             self.filename_addendum).replace('_0MeVlab_', '_'))
                    
                    if whether_plot_corner:
                        with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                            cmap_name = 'Blues'
                            cmap = mpl.cm.get_cmap(cmap_name)
                            
                            for posterior_idx, posterior in enumerate(zip(results_joint_lambdampi, results_joint_lsmpi,
                                                  results_joint_lslambda, results_mpi, 
                                                  results_lambda, results_ls)):
                                # Setup axes
                #                 if ax_joint == None and ax_marg_x == None and ax_marg_y == None:
                                n_plots = len(posterior) // 2
                                fig, ax_joint_array, ax_marg_array, ax_title = corner_plot(n_plots = n_plots)
                                
                                if posterior_idx == 0 or posterior_idx == 1:
                                    ls_vals_corner = ls_vals_Elab
                                else:
                                    ls_vals_corner = ls_vals
        
                                # Plot contour
                                # ax_joint_array[0].contour(mpi_vals, lambda_vals, posterior[0],
                                #                   levels = [np.amax(posterior[0]) * level for level in \
                                #                             ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])],
                                #                   cmap=cmap_name, vmin=0.3, vmax=1.0, zorder=1)
                                ax_joint_array[0].contour(mpi_vals, lambda_vals, posterior[0],
                                                  levels = [np.amax(posterior[0]) * level for level in \
                                                            ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])],
                                                  cmap=cmap_name)
                                corr_coeff = correlation_coefficient(mpi_vals, lambda_vals, posterior[0])
                                ax_joint_array[0].text(.99, .99, rf'$\rho$ = {corr_coeff:.2f}', 
                                                       ha='right', va='top', 
                                                       transform = ax_joint_array[0].transAxes, 
                                                       fontsize = 18)
                                ax_joint_array[0].set_xlim(left = 100, right = 250)
                                ax_joint_array[0].set_ylim(bottom = 500, top = 750)
                                
                                ax_joint_array[1].contour(mpi_vals, ls_vals_corner, posterior[1],
                                                  levels = [np.amax(posterior[1]) * level for level in \
                                                            ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])], 
                                                  cmap=cmap_name)
                                corr_coeff = correlation_coefficient(mpi_vals, ls_vals_corner, posterior[1])
                                ax_joint_array[1].text(.99, .99, rf'$\rho$ = {corr_coeff:.2f}', 
                                                       ha='right', va='top', 
                                                       transform = ax_joint_array[1].transAxes, 
                                                       fontsize = 18)
                                ax_joint_array[1].set_xlim(left = 100, right = 250)
                                ax_joint_array[1].set_ylim(top = 0.5 * (np.max(ls_vals_corner) - np.min(ls_vals_corner)))
                                
                                ax_joint_array[2].contour(lambda_vals, ls_vals_corner, posterior[2],
                                                  levels = [np.amax(posterior[2]) * level for level in \
                                                            ([np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999])],
                                                  cmap=cmap_name)
                                corr_coeff = correlation_coefficient(lambda_vals, ls_vals_corner, posterior[2])
                                ax_joint_array[2].text(.99, .99, rf'$\rho$ = {corr_coeff:.2f}', 
                                                       ha='right', va='top', 
                                                       transform = ax_joint_array[2].transAxes, 
                                                       fontsize = 18)
                                ax_joint_array[2].set_xlim(left = 500, right = 750)
                                ax_joint_array[2].set_ylim(top = 0.5 * (np.max(ls_vals_corner) - np.min(ls_vals_corner)), 
                                                           bottom = 0.1 * (np.max(ls_vals_corner) - np.min(ls_vals_corner)))
                                
                                # Now plot the marginal distributions
                                ax_marg_array[0].plot(mpi_vals, posterior[3], c=cmap(0.8), lw=1)
                                ax_marg_array[0].fill_between(mpi_vals, np.zeros_like(mpi_vals),
                                                        posterior[3], facecolor=cmap(0.2), lw=1)
                                dist_mean, dist_stddev = mean_and_stddev(mpi_vals, posterior[3])
                                dist_mean = sig_figs(dist_mean, 3)
                                dist_stddev = round_to_same_digits(dist_stddev, dist_mean)
                                ax_marg_array[0].set_title(rf'{dist_mean} $\pm$ {dist_stddev}', 
                                                               fontsize = 18)
                                ax_marg_array[0].set_xlim(left = 100, right = 250)
                                ax_marg_array[0].set_xticklabels([])
                                
                                ax_marg_array[1].plot(lambda_vals, posterior[4], c=cmap(0.8), lw=1)
                                ax_marg_array[1].fill_between(lambda_vals, np.zeros_like(lambda_vals),
                                                        posterior[4], facecolor=cmap(0.2), lw=1)
                                dist_mean, dist_stddev = mean_and_stddev(lambda_vals, posterior[4])
                                dist_mean = sig_figs(dist_mean, 3)
                                dist_stddev = round_to_same_digits(dist_stddev, dist_mean)
                                ax_marg_array[1].set_title(rf'{dist_mean} $\pm$ {dist_stddev}', 
                                                               fontsize = 18)
                                ax_marg_array[1].set_xlim(left = 500, right = 750)
                                ax_marg_array[1].set_xticklabels([])
                                
                                ax_marg_array[2].plot(ls_vals_corner, posterior[5], c=cmap(0.8), lw=1)
                                ax_marg_array[2].fill_between(ls_vals_corner, np.zeros_like(ls_vals_corner),
                                                        posterior[5], facecolor=cmap(0.2), lw=1)
                                dist_mean, dist_stddev = mean_and_stddev(ls_vals_corner, posterior[5])
                                dist_mean = sig_figs(dist_mean, 3)
                                dist_stddev = round_to_same_digits(dist_stddev, dist_mean)
                                ax_marg_array[2].set_title(rf'{dist_mean} $\pm$ {dist_stddev}', 
                                                               fontsize = 18)
                                ax_marg_array[2].set_xlim(left = 0.1 * (np.max(ls_vals_corner) - np.min(ls_vals_corner)), 
                                                          right = 0.5 * (np.max(ls_vals_corner) - np.min(ls_vals_corner)))
                                
                                # # sets the x labels for the last marginalized distribution
                                # ax_marg_array[-1].set_xticks(ax_joint_array[1 - n_plots].get_yticks())
                                # rotated_ticklabels = []
                                # for text in ax_joint_array[1 - n_plots].get_yticklabels():
                                #     Y = text._y
                                #     rotated_ticklabels.append(Text(x = Y, y = 0.0, text = f'{Y:.1f}'))
                                # print(rotated_ticklabels)
                                # ax_marg_array[-1].set_xticklabels(rotated_ticklabels)
        
                                # Formatting
                                ax_joint_array[0].set_ylabel(r'$\Lambda_{b}$ (MeV)')
                                ax_joint_array[1].set_xlabel(r'$m_{\pi}$ (MeV)')
                                ax_joint_array[1].set_ylabel(r'$\ell$')
                                ax_joint_array[2].set_xlabel(r'$\Lambda_{b}$ (MeV)')
                                
                                ax_joint_array[0].axvline(mpi_true, 0, 1, c=gray, lw=1, zorder=0)
                                ax_joint_array[1].axvline(mpi_true, 0, 1, c=gray, lw=1, zorder=0)
                                ax_joint_array[2].axvline(Lambda_b_true, 0, 1, c=gray, lw=1, zorder=0)
                                ax_marg_array[0].axvline(mpi_true, 0, 1, c=gray, lw=1, zorder=0)
                                ax_marg_array[1].axvline(Lambda_b_true, 0, 1, c=gray, lw=1, zorder=0)
                                
                                ax_joint_array[0].axhline(Lambda_b_true, 0, 1, c=gray, lw=1, zorder=0)
                                
                                if posterior_idx == 0 or posterior_idx == 1:
                                    ax_title.text(.99, .99, 
                                            'SGT' + '\n' + 
                                            self.scheme + '\,' + self.scale + '\n' + 
                                            r'$' + self.orders_dict[max(self.nn_orders) - 1 + (posterior_idx % 2)] + '$' + '\n' + 
                                            r'$Q_{\mathrm{' + self.Q_param + '}}$' + '\n' + 
                                            self.p_param + '\n' + 
                                            self.vs_what,
                                            ha='right', va='top', 
                                            transform = ax_title.transAxes, 
                                            fontsize = 25)
                                if posterior_idx == 2 or posterior_idx == 3:
                                    ax_title.text(.99, .99, 
                                            'DSG' + '\n' + 
                                            self.scheme + '\,' + self.scale + '\n' + 
                                            r'$' + self.orders_dict[max(self.nn_orders) - 1 + (posterior_idx % 2)] + '$' + '\n' + 
                                            r'$Q_{\mathrm{' + self.Q_param + '}}$' + '\n' + 
                                            self.p_param + '\n' + 
                                            self.vs_what,
                                            ha='right', va='top', 
                                            transform = ax_title.transAxes, 
                                            fontsize = 25)
                                if posterior_idx == 4 or posterior_idx == 5:
                                    ax_title.text(.99, .99, 
                                            'spins' + '\n' + 
                                            self.scheme + '\,' + self.scale + '\n' + 
                                            r'$' + self.orders_dict[max(self.nn_orders) - 1 + (posterior_idx % 2)] + '$' + '\n' + 
                                            r'$Q_{\mathrm{' + self.Q_param + '}}$' + '\n' + 
                                            self.p_param + '\n' + 
                                            self.vs_what,
                                            ha='right', va='top', 
                                            transform = ax_title.transAxes, 
                                            fontsize = 25)
                                
                                # ax_joint.margins(x=0, y=0.)
                                # ax_joint.set_xlim(min(mpi_vals), max(mpi_vals))
                                # ax_joint.set_ylim(min(lambda_vals), max(lambda_vals))
                                # ax_marg_x.set_ylim(bottom=0);
                                # ax_marg_y.set_xlim(left=0);
                                # ax_joint.text(0.95, 0.95, r'pr$(m_{\pi}, \Lambda_{b} \,|\, \vec{\mathbf{y}}_k, \ell)$', ha='right', va='top',
                                #               transform=ax_joint.transAxes,
                                #               bbox=text_bbox
                                #               )
                                plt.show()
                                
                                if 'fig' in locals() and whether_save:
                                    fig.tight_layout()
                                    if posterior_idx == 0:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'SGT_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                                    elif posterior_idx == 2:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'DSG_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                                    elif posterior_idx == 4:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'spin_observables_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'nho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                                    elif posterior_idx == 1:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'SGT_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                                    elif posterior_idx == 3:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'DSG_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                                    elif posterior_idx == 5:
                                        fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + 
                                            'spin_observables_corner_posterior_pdf_curvewise' + '_' + self.scheme + '_' + 
                                                self.scale + '_' + 'ho' + '_Q' + self.Q_param + '_' + self.p_param + '_' + 
                                                self.vs_what + self.filename_addendum).replace('_0MeVlab_', '_'))
                    
                    # except:
                    #     print("Error in plotting the curvewise posterior PDF.")

    def plotzilla(self, whether_save = True):
        """
        Returns
        -------
        Figure with plot.
        """
        # using gridspec, plots the Mahalanobis distance, coefficient curves, credible 
        # intervals, pivoted Cholesky, and Lambda-ell posterior pdf on one figure
        fig_main = plt.figure(figsize=(12, 10))
        
        gs = mpl.gridspec.GridSpec(ncols = 30, nrows = 24, 
                                   wspace = 200, hspace = 400, 
                                   figure = fig_main)
        
        ax_pdf_joint = fig_main.add_subplot(gs[2:12, 0:10])
        ax_pdf_x = fig_main.add_subplot(gs[0:2, 0:10])
        ax_pdf_y = fig_main.add_subplot(gs[2:12, 10:12])
        
        ax_md = fig_main.add_subplot(gs[0:24, 24:30])
        ax_coeff = fig_main.add_subplot(gs[0:12, 12:24])
        ax_ci = fig_main.add_subplot(gs[12:24, 0:12])
        ax_pc = fig_main.add_subplot(gs[12:24, 12:24])
        
        try:
            self.plot_coefficients(ax = ax_coeff, whether_save = True)
        except:
            print("Error in calculating or plotting the coefficient curves.")
        try:
            self.plot_md(ax = ax_md, whether_save = True)
        except:
            print("Error in calculating or plotting the Mahalanobis distance.")
        try:
            self.plot_pc(ax = ax_pc, whether_save = True)
        except:
            print("Error in calculating or plotting the pivoted Cholesky decomposition.")
        try:
            self.plot_credible_intervals(ax = ax_ci, whether_save = True)
        except:
            print("Error in calculating or plotting the credible intervals.")
        try:
            self.plot_posterior_pdf(ax_joint = ax_pdf_joint, ax_marg_x = ax_pdf_x, 
                                 ax_marg_y = ax_pdf_y, whether_save = True)
        except:
            print("Error in calculating or plotting the posterior PDF.")
        
        # adds a title
        fig_main.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(self.fixed_quantity_units) + ')\,' + \
                        '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                        '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size = 30)
        
        if whether_save:
            fig_main.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                        '_' + 'plotzilla' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + '_' + \
                        self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                        '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                        self.train_pts_loc + '_' + self.p_param + 
                        self.filename_addendum).replace('_0MeVlab_', '_'))