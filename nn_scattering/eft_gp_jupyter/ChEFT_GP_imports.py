import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import os
import h5py
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import math
import re

import urllib
import tables

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

cmaps = [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']]
colors = [cmap(0.55 - 0.1 * (i==0)) for i, cmap in enumerate(cmaps)]
light_colors = [cmap(0.25) for cmap in cmaps]
edgewidth = 0.6

# Constants: proton/neutron masses and hbar
m_p = 938.27208  # MeV/c^2
m_n = 939.56541  # MeV/c^2
hbarc = 197.33  # Mev-fm

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

def offset_xlabel(ax):
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax

def E_to_p(E_lab, interaction):
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


def Q_approx(p, Q_parametrization, Lambda_b, interaction='np', single_expansion=False):
    """
    Returns the dimensionless expansion parameter Q.
    
    Parameters
    ----------
    p (float or array) : momentum (in MeV)
    Q_parametrization (str) : can be "poly", "max", or "sum"
    Lambda_b (float) : value for the cutoff (in MeV)
    interaction (str) : can be "np", "nn", or "pp"
    """
    if single_expansion:
        m_pi = 0
    else:
        m_pi = 138  # Set to 0 to just return p/Lambda_b
    
#     p = E_to_p(E, interaction)
    
    if Q_parametrization == "poly":
        # Interpolate to smooth the transition from m_pi to p with a polynomial
        n = 8
        q = (m_pi**n + p**n) / (m_pi**(n-1) + p**(n-1)) / Lambda_b
        return q
    
    elif Q_parametrization == "max":
        # Transition from m_pi to p with a maximum function
        m_pi_eff = 200
        
        try:
            q = [max(P, m_pi_eff) / Lambda_b for P in p]
        except:
            q = max(p, m_pi_eff) / Lambda_b
        return q
    
    elif Q_parametrization == "sum":
        # Transition from m_pi to p with a simple sum
        q = (p + m_pi) / Lambda_b
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

def deg_to_qcm(p, deg):
    """
    Returns the center-of-momentum momentum transfer q in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return p * np.sqrt( 2 * (1 - np.cos(np.radians(deg))) )

def deg_to_qcm2(p, deg):
    """
    Returns the center-of-momentum momentum transfer q squared, in MeV^2.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    degrees     = number
                  angle measure given in degrees
    """
    return (p * np.sqrt( 2 * (1 - np.cos(np.radians(deg))) ))**(2)

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

class GPHyperparameters:
    def __init__(self, ls_class, center, ratio, nugget = 1e-10, seed = None, df = np.inf, \
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

class OrderInfo:
    def __init__(self, orders_array, orders_mask, colors_array, lightcolors_array, \
            orders_restricted = [], mask_restricted = []):
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
    constraint (array or None): constraint on the values of the observable
        For dimensionful (i.e., cross-section) observables, should be None.
    """
    def __init__(self, name, data, energies, angles, title, ref_type, constraint = None):
        self.name = name
        self.data = data
        self.energies = energies
        self.angles = angles
        self.title = title
        self.ref_type = ref_type
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
        self.light_colors = [cmap(0.25) for cmap in self.cmaps]

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
    def __init__(self, name ,ls_guess_factor, ls_bound_lower_factor, 
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
                 gphyperparameters, orderinfo, filenaming, fixed_quantity = [None, None, None], 
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
        self.x_quantity_units = x_quantity[2]
        
        # information on the input space
        self.inputspace = inputspace
        self.vs_what = self.inputspace.name
        self.x = self.inputspace.input_space(self.x_quantity_array)
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
        print(self.nn_orders_mask)
        self.colors = self.orderinfo.colors_array
        self.light_colors = self.orderinfo.lightcolors_array
        self.orders_restricted = self.orderinfo.orders_restricted
        print(self.orders_restricted)
        self.mask_restricted = self.orderinfo.mask_restricted
        print(self.mask_restricted)
        
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
            	self.ref = np.ones(len(self.x))
            	self.ref_train = np.ones(len(self.x_train))
            	self.ref_test = np.ones(len(self.x_test))
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
            	self.ref = np.ones(len(self.x))
            	self.ref_train = np.ones(len(self.x_train))
            	self.ref_test = np.ones(len(self.x_test))
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
        self.coeffs = gm.coefficients(self.data, ratio = self.ratio, \
                            ref = self.ref, orders = self.nn_orders)[:, self.nn_orders_mask]

        # uses interpolation to find the proper ratios for training and testing
        self.interp_f_ratio = interp1d(self.x, self.ratio * np.ones(len(self.x)))
        self.ratio_train = self.interp_f_ratio(self.x_train)
        self.coeffs_train = gm.coefficients(self.y_train, ratio = self.ratio_train, \
                                          ref = self.ref_train, \
                                          orders = self.nn_orders)[:, self.nn_orders_mask]
        self.ratio_test = self.interp_f_ratio(self.x_test)
        self.coeffs_test = gm.coefficients(self.y_test, ratio = self.ratio_test, \
                                          ref = self.ref_test, \
                                          orders = self.nn_orders)[:, self.nn_orders_mask]

        # defines the kernel
        if self.fixed_quantity_name == "energy" and \
                self.fixed_quantity_value < 70.1 and \
                self.fixed_quantity_value >= 1.:
            self.kernel = RBF(length_scale = self.ls, \
                        length_scale_bounds = (self.ls_lower, self.ls_upper)) + \
                        WhiteKernel(1e-8, noise_level_bounds = 'fixed')
        else:
            self.kernel = RBF(length_scale = self.ls, \
                        length_scale_bounds = (self.ls_lower, self.ls_upper)) + \
                        WhiteKernel(1e-10, noise_level_bounds = 'fixed')

        # Define the GP
        self.gp = gm.ConjugateGaussianProcess(
            self.kernel, center = self.center, disp = self.disp, df = self.df,
            scale = self.std_est, n_restarts_optimizer = 10, random_state = self.seed, 
            sd = self.sd)
        
        # restricts coeffs and colors to only those orders desired for 
        # evaluating statistical diagnostics
        self.nn_orders = self.orders_restricted
        self.colors = list(np.array(self.colors)[self.mask_restricted])
        self.light_colors = list(np.array(self.light_colors)[self.mask_restricted])
        self.coeffs = (self.coeffs.T[self.mask_restricted]).T
        self.coeffs_train = (self.coeffs_train.T[self.mask_restricted]).T
        self.coeffs_test = (self.coeffs_test.T[self.mask_restricted]).T

    def PlotCoefficients(self, ax = None, whether_save = True):
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
        self.pred, self.std = self.gp.predict(self.X, return_std = True)
        self.underlying_std = np.sqrt(self.gp.cov_factor_)

        # plots the coefficients against the given input space
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
        
        for i, n in enumerate(self.nn_orders[1:]):
            ax.fill_between(self.x, self.pred[:, i] + 2*self.std, \
                            self.pred[:, i] - 2*self.std, \
                            facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                            lw = edgewidth, alpha=1, zorder = 5 * i - 4)
            ax.plot(self.x, self.pred[:, i], color = self.colors[i], ls='--', zorder = 5 * i - 3)
            ax.plot(self.x, self.coeffs[:, i], color = self.colors[i], zorder = 5 * i - 2)
            ax.plot(self.x_train, self.coeffs_train[:, i], color = self.colors[i], \
                    ls='', marker='o', label=r'$c_{}$'.format(n), zorder = 5 * i - 1)

        # Format
        ax.axhline(2*self.underlying_std, 0, 1, color = gray, zorder=-10, lw=1)
        ax.axhline(-2*self.underlying_std, 0, 1, color = gray, zorder=-10, lw=1)
        ax.axhline(0, 0, 1, color = softblack, zorder=-10, lw=1)
        ax.set_xticks(self.x_test, minor=True)
        ax.set_xticks(self.x_train)
        ax.tick_params(which='minor', bottom=True, top=False)
        ax.set_xlabel(self.caption_coeffs)
        ax.legend(ncol=2, borderpad=0.4,# labelspacing=0.5, columnspacing=1.3,
                  borderaxespad=0.6, loc = 'upper right',
                  title = self.title_coeffs).set_zorder(5 * i)
        
        
        # draws length scales
        ax.annotate("", xy=(np.min(self.x), -0.8*2*self.underlying_std), 
                    xytext=(np.min(self.x) + self.ls, -0.8*2*self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + self.ls + 0.1 * (np.max(self.x) - np.min(self.x)), 
                -0.8*2*self.underlying_std, r'$\ell_{\mathrm{guess}}$', 
                horizontalalignment='right', verticalalignment='center', zorder = 5 * i)

        ax.annotate("", xy=(np.min(self.x), -0.95*2*self.underlying_std), \
                    xytext=(np.min(self.x) + self.ls_true, -0.95*2*self.underlying_std),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + self.ls_true + 0.1 * (np.max(self.x) - np.min(self.x)), 
                -0.95*2*self.underlying_std, \
                r'$\ell_{\mathrm{fit}}$', horizontalalignment='right', verticalalignment='center', zorder = 5 * i)
        
        # draws standard deviations
        ax.annotate("", xy=(np.min(self.x) + 0.94 * (np.max(self.x) - np.min(self.x)), 0), \
                    xytext=(np.min(self.x) + 0.94 * (np.max(self.x) - np.min(self.x)), \
                            -1. * self.std_est),
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + 0.96 * (np.max(self.x) - np.min(self.x)), \
                -1.2 * self.std_est, r'$\sigma_{\mathrm{guess}}$', horizontalalignment='center', \
                verticalalignment='bottom', zorder = 5 * i)
        ax.annotate("", xy=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), 0), \
                    xytext=(np.min(self.x) + 0.90 * (np.max(self.x) - np.min(self.x)), \
                            -1. * self.underlying_std),
                        
                    arrowprops=dict(arrowstyle="<->", capstyle='projecting', lw=1,
                                    color='k'), annotation_clip=False, zorder = 5 * i)
        ax.text(np.min(self.x) + 0.88 * (np.max(self.x) - np.min(self.x)), \
                -1.2 * self.underlying_std, r'$\sigma_{\mathrm{fit}}$', horizontalalignment='center', \
                verticalalignment='bottom', zorder = 5 * i)
        
        if 'fig' in locals() and whether_save:
            fig.tight_layout()
    
            fig.savefig(('figures/' + self.scheme + '_' + self.scale + '/' + self.observable_name + \
                    '_' + 'interp_and_underlying_processes' + '_' + str(self.fixed_quantity_value) + str(self.fixed_quantity_units) + \
                    '_' + self.scheme + '_' + self.scale + '_Q' + self.Q_param + '_' + self.vs_what + \
                    '_' + str(self.n_train_pts) + '_' + str(self.n_test_pts) + '_' + \
                    self.train_pts_loc + '_' + self.p_param + 
                    self.filename_addendum).replace('_0MeVlab_', '_'))

    def PlotMD(self, ax = None, whether_save = True):
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
            self.mean = self.gp.mean(self.X_test)
            self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, 
                        self.mean, self.cov, colors = self.colors, gray=gray, 
                        black=softblack)
    
            if ax is None:
                fig, ax = plt.subplots(figsize=(1, 3.2))
                
            self.gr_dgn.md_squared(type = 'box', trim = False, title = None, \
                            xlabel=r'$\mathrm{D}_{\mathrm{MD}}^2$', ax = ax)
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

    def PlotPC(self, ax = None, whether_save = True):
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
            self.mean = self.gp.mean(self.X_test)
            self.cov = self.gp.cov(self.X_test)
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, \
                                                     self.mean, self.cov, \
                                                     colors = self.colors, gray=gray, black=softblack)

            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
                if ax is None:
                    fig, ax = plt.subplots(figsize=(3.2, 3.2))
                    
                self.gr_dgn.pivoted_cholesky_errors(ax = ax, title = None)
                ax.set_xticks(np.arange(2, self.n_test_pts + 1, 2))
                ax.set_xticks(np.arange(1, self.n_test_pts + 1, 2), minor = True)
                ax.text(0.05, 0.95, r'$\mathrm{D}_{\mathrm{PC}}$', bbox = text_bbox, \
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

    def PlotPosteriorPDF(self, ax_joint = None, ax_marg_x = None, 
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
            self.gp_post = gm.TruncationGP(self.kernel, ref = lambda_interp_f_ref, 
                                        ratio = lambda_interp_f_ratio, 
                                        center = self.center, 
                                        disp = self.disp, df = self.df, scale = self.std_est, 
                                        excluded = [0], 
                                        ratio_kws = {"lambda_var" : self.Lambda_b})
            self.gp_post.fit(self.X_train, self.y_train, orders = self.nn_orders_full, 
                             orders_eval = self.nn_orders, dX = np.array([[0]]), dy=[0])
            
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
                              transform=ax_joint.transAxes,
                              bbox=text_bbox
                              );
    
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

    def PlotTruncationErrors(self, online_data, constraint = None, whether_save = True):
        """
        Parameters
        ----------
        online_data : array
            Summed order-by-order predictions for an observable.
        constraint : array of arrays, optional
            The constraints for the fit procedure. The default is None.
        whether_save : bool, optional
            Whether to save the output. The default is True.

        Returns
        -------
        Figure (2)
            A figure with the order-by-order residuals plot and a figure with 
            the order-by-order empirical coverage.
        """
        self.online_data = online_data
        
        def lambda_interp_f_ref(x_):
            X = np.ravel(x_)
            return self.interp_f_ref(X)
        def lambda_interp_f_ratio(x_, lambda_var):
            X = np.ravel(x_)
            return self.interp_f_ratio(X) * self.Lambda_b / lambda_var
        
        # try:
        # without the constraint
        # print(self.kernel)
        # print(self.center)
        # print(self.disp)
        # print(self.df)
        # print(self.std_est)
        # print(self.Lambda_b)
        self.gp_trunc = gm.TruncationGP(self.kernel, ref = lambda_interp_f_ref, 
                    ratio = lambda_interp_f_ratio, 
                    center = self.center, disp = self.disp, df = self.df, 
                    scale = self.std_est, excluded = [0], 
                    ratio_kws = {"lambda_var" : self.Lambda_b})
        
        # fits the GP with or without a constraint
        # print(self.y_train.shape)
        if constraint is None:
            self.gp_trunc.fit(self.X_train, self.y_train, orders = self.nn_orders)
        else:
            self.gp_trunc.fit(self.X_train, self.y_train, orders = self.nn_orders, 
                              dX = np.array(constraint[0]), dy=constraint[1])

        fig, axes = plt.subplots(3, 2, sharex=True, sharey=False, figsize=(6, 7))
        
        # eliminates useless subplots from the truncation error figure
        if self.scheme == "RKE":
            fig.delaxes(axes[2, 1])
            print("We deleted it.")
        elif self.scheme == "EKM":
            fig.delaxes(axes[2, 0])
            fig.delaxes(axes[2, 1])
        elif self.scheme == "EMN":
            fig.delaxes(axes[2, 0])
            fig.delaxes(axes[2, 1])
        elif self.scheme == "GT":
            fig.delaxes(axes[1, 0])
            fig.delaxes(axes[1, 1])
            fig.delaxes(axes[2, 0])
            fig.delaxes(axes[2, 1])
            
        if True:
            # plots the order-by-order residuals
            # print(self.nn_orders)
            for i, n in enumerate(self.nn_orders[self.nn_orders_mask]):
                # print("i = " + str(i))
                # print("n = " + str(n))
                _, self.std_trunc = self.gp_trunc.predict(self.X, order = n, 
                                        return_std = True, kind = 'trunc')
                # print("std_trunc = " + str(self.std_trunc))
                
                # gets the "true" order-by-order data from online
                if self.fixed_quantity_name == "energy":
                    data_true = self.online_data[self.fixed_quantity_value, :]
                elif self.fixed_quantity_name == "angle":
                    if self.fixed_quantity_value == 0:
                        data_true = self.online_data
                    else:
                        data_true = self.online_data[:, self.fixed_quantity_value]
                        
                for j in range(i, max(self.nn_orders[self.nn_orders_mask])):
                    # print("j = " + str(j))
                    ax = axes.ravel()[j]
                    # print("There are " + str(len(axes.ravel())) + " axes.")
                    
                    # calculates the residuals
                    # print("data_true has shape " + str(data_true.shape))
                    # print("self.data has shape " + str(self.data.shape))
                    residual = data_true - (self.data[:, self.nn_orders_mask])[:, i]
                    # print("residual has shape " + str(residual.shape))
                    
                    ax.plot(self.x, residual, zorder=i-5, c = self.colors[i])
                    for line in self.x_test: ax.axvline(line, 0, 1, c = gray);
                    # number of standard deviations around the residual to plot
                    std_coverage = 0.64
                    ax.fill_between(self.x, residual + std_coverage*self.std_trunc, 
                                    residual - std_coverage*self.std_trunc, zorder=i-5, 
                                    facecolor = self.light_colors[i], edgecolor = self.colors[i], 
                                    lw=edgewidth)
                    ax.set_ylim(np.min(np.concatenate((residual + std_coverage*self.std_trunc, residual - std_coverage*self.std_trunc))), 
                                np.max(np.concatenate((residual + std_coverage*self.std_trunc, residual - std_coverage*self.std_trunc))))
                
                ax = axes.ravel()[i]
                
                ax.plot(self.x, np.zeros(len(self.x)), color=softblack, lw=1, ls='--')
                
                ax.set_xticks([min(self.x) + (max(self.x) - min(self.x)) / 3, 
                               min(self.x) + (max(self.x) - min(self.x)) / 3 * 2])
    
            # Format
            axes[1, 0].set_xlabel(self.caption_coeffs)
            axes[1, 1].set_xlabel(self.caption_coeffs)
            
        # else:
        #     for i, n in enumerate(self.nn_orders[1:]):
        #         _, self.std_trunc = self.gp_trunc.predict(self.X, order = n, 
        #                                 return_std = True, kind = 'trunc')
    
        #         for j in range(i, max(self.nn_orders)):
        #             ax = axes.ravel()[j]
        #             ax.plot(self.x, self.data[:, i+1], zorder=i-5, c = self.colors[i])
        #             # ax.fill_between(self.x, self.data[:, i+1] + 2*self.std_trunc, 
        #             #                 self.data[:, i+1] - 2*self.std_trunc, zorder=i-5, 
        #             #                 facecolor = self.light_colors[i], edgecolor = self.colors[i], 
        #             #                 lw=edgewidth)
        #             percent_coverage = 2
        #             ax.fill_between(self.x, self.data[:, i+1] + percent_coverage*self.std_trunc, 
        #                             self.data[:, i+1] - percent_coverage*self.std_trunc, zorder=i-5, 
        #                             facecolor = self.light_colors[i], edgecolor = self.colors[i], 
        #                             lw=edgewidth)
        #         ax = axes.ravel()[i]
        #         if self.fixed_quantity_name == "energy":
        #             data_true = self.online_data[self.fixed_quantity_value, :]
        #             ax.plot(self.x, data_true, color=softblack, 
        #                 lw=1, ls='--')
        #         elif self.fixed_quantity_name == "angle":
        #             if self.fixed_quantity_value == 0:
        #                 data_true = self.online_data
        #                 ax.plot(self.x, data_true, color=softblack, 
        #                     lw=1, ls='--')
        #             else:
        #                 data_true = self.online_data[:, self.fixed_quantity_value]
        #                 ax.plot(self.x, data_true, color=softblack, 
        #                     lw=1, ls='--')
        #         ax.set_xticks([min(self.x) + (max(self.x) - min(self.x)) / 3, 
        #                        min(self.x) + (max(self.x) - min(self.x)) / 3 * 2])
        #         if self.ref_type == "dimensionless":
        #             ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
        #         else:
        #             ax.set_yticks(np.linspace(min(data_true), max(data_true), 
        #                                       num = 5, dtype = 'int'))
        #     ax.set_ylim(min(data_true) - (max(data_true) - min(data_true)) / 2, 
        #                 max(data_true) + (max(data_true) - min(data_true)) / 2)
    
        #     # Format
        #     axes[1, 0].set_xlabel(self.caption_coeffs)
        #     axes[1, 1].set_xlabel(self.caption_coeffs)
            
        if 'fig' in locals() and whether_save:
            fig.suptitle(r'$\mathrm{' + self.observable_name + '\,(' + str(self.fixed_quantity_value) + '\,' + str(self.fixed_quantity_units) + ')\,' + \
                            '\,for\,' + self.scheme + '\,' + self.scale + '}' + '\,(Q_{\mathrm{' + self.Q_param + \
                            '}},\,\mathrm{' + self.p_param + '},\,\mathrm{' + self.vs_what + '})$', size = 20)
            fig.tight_layout()
            
            if constraint is None:
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
        
        # self.gp_trunc.kernel_
        # this is copied from PlotCredibleIntervals() and modified only slightly
        # print("data has shape " + str((self.data[:, 1:]).shape))
        # print("data_true has shape " + str(data_true.shape))
        data_interp = interp1d(self.x, self.data[:, self.nn_orders_mask].T)
        # print((self.data[:, 1:]).shape)
        data_true_interp = interp1d(self.x, data_true)
        # self.data_wp = np.reshape(data_interp(self.X_test), (len(self.colors), len(self.X_test)))
        # print("data_wp has shape " + str(self.data_wp.shape))
        self.cov_wp = self.gp_trunc.cov(self.X_test, start = 0, end = np.inf)
        # print("cov_wp = " + str(self.cov_wp))
        self.mean_wp = self.gp_trunc.mean(self.X_test)
        # print("mean_wp = " + str(self.mean_wp))
        
        
        # norms the residuals by factors of the ratio
        # print(self.X_test)
        self.norm_residuals_wp = data_true_interp(self.X_test) - data_interp(self.X_test)
        # self.norm_residuals_wp = data_true_interp(self.X_test) - self.data_wp[:, :, None]        
        # print(self.norm_residuals_wp)
        # print(data_true_interp(self.X_test).shape)
        # print(data_interp(self.X_test).shape)
        # print(self.data_wp[:, :, None].shape)
        # print(self.norm_residuals_wp.shape)
        # print(str(self.data_wp[:, :, None] == data_interp(self.X_test)))
        # print(self.data_wp[:, :, None])
        # print(data_interp(self.X_test))
        denom = (np.tile(self.ratio_test, (len(self.nn_orders[self.nn_orders_mask]), 1)).T)**(self.nn_orders[self.nn_orders_mask] + 1) * (np.sqrt(1 - np.tile(self.ratio_test, (len(self.nn_orders[self.nn_orders_mask]), 1))**2)).T
        # print((np.tile(self.ratio_test, (len(self.nn_orders[1:]), 1)).T)**(self.nn_orders[1:] + 1))
        # print((np.sqrt(1 - np.tile(self.ratio_test, (len(self.nn_orders[1:]), 1))**2)).T)
        # print((denom.T)[:, :, None])
        self.norm_residuals_wp = self.norm_residuals_wp / (denom.T)[:, :, None]
        # print(self.norm_residuals_wp.T)
        self.gr_dgn_wp = gm.GraphicalDiagnostic(self.norm_residuals_wp.T, 
                            mean=self.mean_wp, cov=self.cov_wp, 
                            colors=self.colors, gray=gray, black=softblack)
        # norm_trunc_cov = gp_trunc.cov(X[x_valid_mask], start=0, end=0)
        # norm_residuals = (data_true[x_valid_mask, None] - data[x_valid_mask]) / (ratio**(orders+1) / np.sqrt(1 - ratio**2))
        # gr_dgn_trunc = gm.GraphicalDiagnostic(norm_residuals, mean=np.zeros(x[x_valid_mask].shape[0]), cov=norm_trunc_cov,
        #                               colors=colors, gray=gray, black=softblack)

        
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
        
        # except:
        #     print("Error plotting the truncation errors.")
            
    def PlotCredibleIntervals(self, ax = None, whether_save = True):
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
            self.mean = self.gp.mean(self.X_test)
            self.cov = self.gp.cov(self.X_test)
            
            self.gr_dgn = gm.GraphicalDiagnostic(self.coeffs_test, \
                                self.mean, self.cov, colors = self.colors, gray = gray, \
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

    def Plotzilla(self, whether_save = True):
        """
        Parameters
        ----------
        whether_save : bool, optional
            Whether to save the figure. The default is True.

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
            self.PlotCoefficients(ax = ax_coeff, whether_save = True)
        except:
            print("Error in calculating or plotting the coefficient curves.")
        try:
            self.PlotMD(ax = ax_md, whether_save = True)
        except:
            print("Error in calculating or plotting the Mahalanobis distance.")
        try:
            self.PlotPC(ax = ax_pc, whether_save = True)
        except:
            print("Error in calculating or plotting the pivoted Cholesky decomposition.")
        try:
            self.PlotCredibleIntervals(ax = ax_ci, whether_save = True)
        except:
            print("Error in calculating or plotting the credible intervals.")
        try:
            self.PlotPosteriorPDF(ax_joint = ax_pdf_joint, ax_marg_x = ax_pdf_x, 
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