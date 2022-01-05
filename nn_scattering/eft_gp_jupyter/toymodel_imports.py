import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import h5py
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import math
import urllib
import tables
import colorsys
import ipywidgets as widgets
from IPython.display import display
import warnings

def scale_lightness(rgb, scale_l):
    """
    Scales the lightness of a color. Takes in a color defined in RGB, converts to HLS, lightens
    by a factor, and then converts back to RGB.
    """
    # converts rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulates h, l, s values and returns as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

class GPHyperparameters:
    def __init__(self, ls, sd, center, ref, ratio, nugget, seed, df = np.inf, disp = 0, \
                 scale = 1):
        """
        Class for the hyperparameters of a Gaussian process.
        """
        self.ls = ls
        self.sd = sd
        self.center = center
        self.ref = ref
        self.ratio = ratio
        self.nugget = nugget
        self.seed = seed
        self.df = df
        self.disp = disp
        self.scale = scale

class order_info:
    def __init__(self, n_orders, n_final_order):
        """
        Class for information about the orders of calculation (the highest order of interest, the 
        final order for summing the "true" value, and arrays of the integers from 0 to these numbers)
        """
        self.n_orders = n_orders
        self.n_final_order = n_final_order
        self.orders_array = np.arange(0, self.n_orders)
        self.orders_all_array = np.arange(0, self.n_final_order + 1)

def regular_train_test_split(x, dx_train, dx_test, offset_train = 0, offset_test = 0, \
                                 xmin = None, xmax = None):
    """
    Sets the mask for the x-values for training and testing the Gaussian process.
    """
    train_mask = np.array([(i - offset_train) % dx_train == 0 for i in range(len(x))])
    test_mask = np.array([(i - offset_test) % dx_test == 0 for i in range(len(x))])
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    train_mask = train_mask & (x >= xmin) & (x <= xmax)
    test_mask = test_mask  & (x >= xmin) & (x <= xmax) & (~ train_mask)
    return train_mask, test_mask

def find_nearest(array, value):
    """
    Finds the value in array closest to value and returns that entry.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def mask_mapper(array_from, array_to, mask_from):
    """
    Converts from one mask to another by mapping the entries of the first to the nearest-in-
    value entries in the second.
    """
    mask_array = [( np.argwhere(array_to == find_nearest(array_to, i)) ) for i in array_from[mask_from]]
    mask = np.zeros(len(array_from))
    for i in range(len(mask_array)):
        mask[mask_array[i]] = 1
    return np.array(mask.astype(int), dtype = bool)

def offset_xlabel(ax):
    """
    Sets the offset for the x-axis label.
    """
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax

class PosteriorBounds:
    def __init__(self, x_lower, x_upper, x_n, y_lower, y_upper, y_n):
        """
        Class for the boundaries of the 2D posterior PDF plot and the mesh on which it is plotted.
        """
        self.x_vals = np.linspace(x_lower, x_upper, x_n)
        self.y_vals = np.linspace(y_lower, y_upper, y_n)

def joint_plot(ratio=1, height=3):
    """
    Taken from Seaborn JointGrid
    """
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

class GSUMobj:
    def __init__(self, gphyperparameters, orderinfo, x, fullyrandomcolors = False, \
                 color_seed = None, constrained = False, x_power = 1):
        """
        Class for everything involving Jordan Melendez's GSUM library.
        gphyperparameters (GPHyperparameters) : parameters for fitted Gaussian process
        orderinfo (order_info) : information on the calculated and plotted orders
        x (float array) : x-coordinate mesh over which the GP is calculated, plotted, and fitted
        fullyrandomcolors (boolean) : are all the colors randomly generated?
        color_seed : value of the seed from which the colors are randomly generated
        constrained (bool) : is the GP fitting process constrained?
        x_power : power by which the x-coordinate is scaled (to test stationarity)
        """
        # reads the hyperparameters to the class
        self.hyp = gphyperparameters
        self.ls = self.hyp.ls
        self.sd = self.hyp.sd
        self.center = self.hyp.center
        self.ref = self.hyp.ref
        self.ratio = self.hyp.ratio
        self.nugget = self.hyp.nugget
        self.seed = self.hyp.seed
        self.df = self.hyp.df
        self.disp = self.hyp.disp
        self.scale = self.hyp.scale
        
        # creates a kernel that defines the Gaussian process (GP)
        self.kernel = RBF(length_scale = self.ls, length_scale_bounds = 'fixed') + \
                WhiteKernel(noise_level = self.nugget, noise_level_bounds= 'fixed')

        # reads the order information to the class
        self.orderinfo = orderinfo
        self.n_orders = self.orderinfo.n_orders
        self.n_final_order = self.orderinfo.n_final_order
        self.orders_array = self.orderinfo.orders_array
        self.orders_all_array = self.orderinfo.orders_all_array
        
        # reads whether the colors will be fully randomly chosen or not and what seed will be 
        # used to generate the random ones
        self.fullyrandomcolors = fullyrandomcolors
        self.color_seed = color_seed
        self.color_randomstate = np.random.RandomState(self.color_seed)
        
        if self.fullyrandomcolors:
            # creates an array of random colors
            self.colors = []
            for i in range(0, self.n_final_order + 1):
                self.colors.append(self.color_randomstate.rand(3,))
            self.light_colors = [scale_lightness(color[:3], 1.5) for color in self.colors]
        else:
            # sets the arrays for the colors and the light colors, keeping Jordan Melendez's 
            # scheme for the first five orders and randomizing the colors for higher orders
            cmaps = [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples']]
            self.colors = [cmap(0.55 - 0.1 * (i==0)) for i, cmap in enumerate(cmaps)]
            for i in range(len(self.colors), self.n_final_order + 1):
                self.colors.append(self.color_randomstate.rand(3,))
            self.light_colors = [scale_lightness(color[:3], 1.5) for color in self.colors]
        
        # takes in the array of x-values over which the kernel generates the toy curves
        self.x_underlying = x
        self.X_underlying = self.x_underlying[:, None]
        
        # scales the x-axis by some factor, with the resulting x and X arrays being used for all
        # plotting and fitting
        if x_power == 1:
            self.x = self.x_underlying
            self.X = self.X_underlying
        else:
            self.x = (self.x_underlying)**(x_power)
            self.X = self.x[:, None]
        
        # is the GP constrained? The default answer is No
        self.constrained = constrained
        
        # creates the masks for training and testing the GPs
        self.x_train_mask, self.x_valid_mask = regular_train_test_split(self.x_underlying, \
                                dx_train = 24, dx_test = 6, offset_train = 1, offset_test = 1)
        
        # creates the masks for training and testing the GPs, taking into account any scaling 
        # power for the x-coordinate
        self.x_train_mask, self.x_valid_mask = \
                            mask_mapper(self.x_underlying, self.x, self.x_train_mask), \
                            mask_mapper(self.x_underlying, self.x, self.x_valid_mask)
        
        if not constrained:
            # for the given hyperparameters, orders, and x-variable, generates the data to all 
            # orders and extracts the coefficient curves at all orders
            self.gp = gm.ConjugateGaussianProcess(kernel = self.kernel, center = self.center, \
                                              df = np.inf, scale = self.sd, nugget = 0)
            self.coeffs_all = - self.gp.sample_y(self.X_underlying, \
                                                 n_samples = self.n_final_order + 1, \
                                                 random_state = self.seed)
            self.data_all = gm.partials(self.coeffs_all, self.ratio, ref = self.ref, \
                                        orders = self.orders_all_array)
            self.diffs_all = np.array([self.data_all[:, 0], *np.diff(self.data_all, axis=1).T]).T
            self.data_true = self.data_all[:, -1]
            
            self.coeffs = self.coeffs_all[:, :self.n_orders]
            self.data = self.data_all[:, :self.n_orders]
            self.diffs = self.diffs_all[:, :self.n_orders]
            
        else:
            # given constraints, extracts the coefficient curves at all orders
            self.gp_constrained = gm.ConjugateGaussianProcess(kernel = self.kernel, \
                        optimizer = None).fit(np.array([[0], [1]]), np.array([0, 0]))
            self.cn_constrained = self.gp_constrained.sample_y(self.X_underlying, \
                                    n_samples = self.n_orders, random_state = 5)
            self.yn_constrained = gm.partials(self.cn_constrained, ratio = self.ratio)
    
    def PlotCurvesFull(self):
        # plots the data summed to each order, up to the order we are interested in
        fig, ax = plt.subplots(1, 1, figsize=(2.45, 2.6))

        for i, curve in enumerate(self.data.T):
            ax.plot(self.x, curve, label = r'$y_{}$'.format(i), c = self.colors[i])

        ax.text(0.95, 0.95, 'Predictions', ha = 'right', va = 'top',
                transform = ax.transAxes)
        legend = ax.legend(**top_legend_kwargs)
        ax.set_xlabel(r'$x$')
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticks([0.25, 0.75], minor = True)
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_xlim(0, 1)
        fig.tight_layout()
        
    def PlotCurvesDiffs(self):
        # plots the differences between each order and the next, up to the order we are interested
        # in
        with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
#         with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(1, 1, figsize=(2.45, 2.6))

            ax.axhline(0, lw = 1, ls = '--', c = softblack)
            # For the zeroth-order, only, use the data itself
            ax.plot(self.x, self.data[:, 0], label = r'$y_0$', c = self.colors[0])
            for i in range(1, self.n_orders):
                # Beyond that, use differences
                ax.plot(self.x, self.diffs[:, i], label = r'$\Delta y_{}$'.format(i), c = self.colors[i])

            ax.text(0.95, 0.95, 'Differences', ha='right', va='top',
                       transform=ax.transAxes)

            legend = ax.legend(**top_legend_kwargs)

            # Format
            ax.set_xlabel(r'$x$')
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticks([0.25, 0.75], minor=True)
            ax.set_xticklabels([0, 0.5, 1])
            ax.set_xlim(0, 1)
            fig.tight_layout()
            plt.show()
        
    def PlotCurvesCoeffs(self):
        # plots the coefficient curves for each order we are interested in
        fig, ax = plt.subplots(1, 1, figsize=(2.45, 2.6))

        for i in range(self.n_orders):
            ax.plot(self.x, self.coeffs[:, i], label = r'$c_{}$'.format(i), c = self.colors[i])

        ax.text(0.95, 0.95, 'Coefficients', ha = 'right', va = 'top', transform = ax.transAxes)
        legend = ax.legend(**top_legend_kwargs)
        ax.set_xlabel(r'$x$')
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticks([0.25, 0.75], minor = True)
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_xlim(0, 1)
        fig.tight_layout()
    
    def PlotPointwiseVariance(self, mask):
        # plots the variance for each order we're interested in at each of a given set of points 
        # (the "mask") and compares to the expected variance
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

        for i, c in enumerate(self.coeffs.T):
            ax.plot(self.x, c, color = self.colors[i], zorder = 0)
            ax.plot(self.x[mask], c[mask], ls = '', marker = 'o', color = self.colors[i])

        # Indicate the pointwise errors  
        for i, xi in enumerate(self.x[mask]):
            # Fit a Gaussian to the orders at this x_i
            sd_fit = stats.norm.fit(self.coeffs[mask][i], floc = 0)[1]
            # Draw a vertical arrow showing mean +/- sd
            ax.annotate(
                "", xy = (xi-0.035, sd_fit), xytext = (xi-0.035, -sd_fit), zorder = 1,
                arrowprops = dict(arrowstyle = "<->", capstyle = 'projecting', lw = 1, \
                                  color = softblack),)
            ax.text(xi-0.07, 0.65, r'$\bar c_{}$'.format(i), horizontalalignment = 'center',
                    verticalalignment = 'center', zorder = 1)    

        # Draw length scale
        ax.annotate("", xy = (self.x[mask][2], 2 * self.sd), xytext=(self.x[mask][2] + self.ls, 2 * self.sd),
                    arrowprops = dict(arrowstyle = "<->", capstyle = 'projecting', lw = 1,
                                    color = softblack), zorder = 1)
        ax.text(self.x[mask][2] + self.ls / 2, 1.79 * self.sd, r'$\ell$', horizontalalignment = 'center',
                    verticalalignment = 'center', zorder = 1)

        # Format plot
        ax.axhline(0, 0, 1, c = softblack, lw = 1, zorder = -1)
        ax.axhline(1, 0, 1, c = gray, lw = 1, zorder = -1)
        ax.axhline(-1, 0, 1, c = gray, lw = 1, zorder = -1)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([r'$-\bar c$', r'$\mu$', r'$\bar c$'])
        ax.tick_params(length = 0, axis = 'y')
        ax.set_xlabel(r'$x$')
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticks([0.25, 0.75], minor = True)
        ax.set_xticklabels([0, 0.5, 1])

        fig.tight_layout()
        
    def PlotPointwiseFit(self, mask, expensive = True, constrained = False):
        # plots sums of data curves up to each order we're interested in. If the system is 
        # inexpensive, we plot the full sum for each x-value; if the system is expensive, we 
        # plot the full sum of the curves when fit a subset of x-values
        
        # By setting disp=0 and df=inf, no updating of hyperparameters occurs
        # The priors become Dirac delta functions at mu=center and cbar=scale
        # But this assumption could be relaxed, if desired
        trunc_gp = gm.TruncationGP(kernel = self.kernel, ref = self.ref, ratio = self.ratio, \
                                   disp = 0, df = np.inf, scale = 1, optimizer = None)
        # Still only fit on a subset of all data to update mu and cbar!
        # We must beware of numerical issues of using data that are "too close"
        trunc_gp.fit(self.X[mask], self.data[mask], orders = self.orders_array)

        fig, axes = plt.subplots(math.ceil(n_orders / 2), 2, sharex = True, sharey = True, \
                                 figsize = (5, 8))
        
        for i, n in enumerate(self.orders_array):
            if expensive:
                # Only get the uncertainty due to truncation (kind='trunc')
                pred_exp, std_trunc_exp = trunc_gp.predict(self.X, order = n, \
                                                               return_std = True)

                for j in range(i, self.n_orders):
                    ax = axes.ravel()[j]
                    ax.plot(self.x, pred_exp, zorder = i-5, c = self.colors[i])
                    ax.plot(self.x[mask], self.data[mask, i], ls = '', c = self.colors[i], \
                            marker = 'o', zorder = i-5)
                    ax.fill_between(self.x, pred_exp + 2 * std_trunc_exp, \
                                    pred_exp - 2 * std_trunc_exp, zorder = i-5, \
                                    facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                                    lw = edgewidth)
                ax = axes.ravel()[i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-15, 37)
            else:
                # Only get the uncertainty due to truncation (kind='trunc')
                _, std_trunc = trunc_gp.predict(self.X, order = n, return_std = True, kind = 'trunc')

                for j in range(i, self.n_orders):
                    ax = axes.ravel()[j]
                    ax.plot(self.x, self.data[:, i], zorder = i-5, c = self.colors[i])
                    ax.fill_between(self.x, self.data[:, i] + 2 * std_trunc, \
                                    self.data[:, i] - 2 * std_trunc, zorder = i-5, \
                                    facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                                    lw = edgewidth)
                ax = axes.ravel()[i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-15, 37)
                
        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        
    def PlotCurvesConstrained(self, constrained):
        # plots sums of data curves up to each order we're interested in, subject to some 
        # constraint
        if constrained:
            trunc_gp_sym = gm.TruncationGP(kernel = self.kernel, ref=1, ratio = self.ratio, \
                            disp = 0, df = np.inf, scale = 1, optimizer = None)
            # fits GP given constraints
            trunc_gp_sym.fit(self.X[::10], self.yn_constrained[::10], orders = self.orders_array, \
                             dX = np.array([[0], [1]]), dy = np.array([0, 0]))

            fig, axes = plt.subplots( math.ceil(self.n_orders / 2), 2, sharex = True, \
                                     sharey = True, figsize = (5, 8) )
            for i, n in enumerate(self.orders_array):
                # Again, only consider the truncation errors for this plot
                _, std_sym = trunc_gp_sym.predict(self.X, order = n, return_std = True, \
                                    kind = 'trunc')

                for j in range(i, self.n_orders):
                    ax = axes.ravel()[j]
                    ax.plot(self.x, self.yn_constrained[:, i], zorder = i-5, c = self.colors[i])
                    ax.fill_between(self.x, self.yn_constrained[:, i] + 2 * std_sym, \
                                    self.yn_constrained[:, i] - 2 * std_sym, zorder = i-5, \
                                    facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                                    lw = edgewidth)
                ax = axes.ravel()[i]
                ax.axhline(0, 0, 1, ls = '--', lw = 0.5, c = softblack, zorder = 0)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout(h_pad=0.3, w_pad=0.3);
        else:
            return 0
    
    def PlotGPCurvesFit(self):
        # fits the coefficient curves for the orders we're interested in to a GP at training 
        # points, and then plots (with error bars) alongside true curves at each order
        
        # kernel for fit
        self.kernel_fit = RBF(length_scale = self.ls) + \
                            WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

        # fits to GP and extracts error bars
        self.gp_diagnostic = gm.ConjugateGaussianProcess(kernel = self.kernel_fit, center = self.center, \
                    disp = self.disp, df = self.df, scale = self.scale, n_restarts_optimizer = 2, \
                    random_state = 32)
        self.gp_diagnostic.fit(self.X[self.x_train_mask], self.coeffs[self.x_train_mask])
        self.pred, self.std = self.gp_diagnostic.predict(self.X, return_std = True)
        self.underlying_std = np.sqrt(self.gp_diagnostic.cov_factor_)
        
        fig, ax = plt.subplots(figsize = (3.2, 3.2))
        
        for i, n in enumerate(self.orders_array):
            # plots true and predicted coefficient curves, mask points, and error bars
            ax.plot(self.x, self.pred[:, i], c = self.colors[i], zorder = i-5, ls = '--')
            ax.plot(self.x, self.coeffs[:, i], c = self.colors[i], zorder = i-5)
            ax.plot(self.x[self.x_train_mask], self.coeffs[self.x_train_mask, i], \
                    c = self.colors[i], zorder = i-5, ls = '', marker = 'o', \
                    label = r'$c_{}$'.format(n))
            ax.fill_between(self.x, self.pred[:, i] + 2 * self.std, self.pred[:, i] - 2 * self.std, \
                    zorder = i-5, facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                            lw = edgewidth, alpha = 1)
        
        ax.axhline(2 * self.underlying_std, 0, 1, c = gray, zorder = -10, lw = 1)
        ax.axhline(-2 * self.underlying_std, 0, 1, c = gray, zorder = -10, lw = 1)
        ax.axhline(0, 0, 1, c = softblack, zorder = -10, lw = 1)
        ax.set_xticks(self.x[self.x_valid_mask], minor = True)
        ax.set_xlabel(r'$x$')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(which = 'minor', bottom = True, top = False)
        ax.legend(ncol = 2, borderaxespad = 0.5, borderpad = 0.4)
        fig.tight_layout()
        
    def PlotSingleCurveFit(self, curve_idx, kernel = None, center = None, disp = None, \
                           df = None, scale = None, nugget = None, mask = np.array([True])):
        # for a single order, plots the true coefficient curve and the curve fit to a GP (with
        # error bars) for some mask
        
        # reads in hyperparameters for the kernel, changing them if they are different from the 
        # original
        if kernel == None:
            kernel = self.kernel
        if center == None:
            center = self.center
        if disp == None:
            disp = self.disp
        if df == None:
            df = self.df
        if scale == None:
            scale = self.scale
        if nugget == None:
            nugget = self.nugget
        if all(mask):
            mask = self.x_train_mask
        
        # interpolates between training points
        interpolater = gm.ConjugateGaussianProcess(kernel = kernel, center = center, disp = disp, \
                        df = df, scale = scale, nugget = nugget)
        interpolater.fit(self.X[mask], self.coeffs[mask, [curve_idx]])
        pred_interp, std_interp = interpolater.predict(self.X, return_std = True)
        
        fig, ax = plt.subplots(figsize=(3.5, 3))

        # Interpolating curve
        ax.plot(self.x, self.coeffs[:, [curve_idx]], c = self.colors[curve_idx],
                label = r'$c_{}$ ($\sigma_n = 0$)'.format(curve_idx), zorder = 0)
        ax.plot(self.x, pred_interp, c = self.colors[curve_idx], ls = '--', zorder = 0)
        ax.plot(self.x[mask], self.coeffs[mask, curve_idx], ls = '', marker = 'o', \
                c = self.colors[curve_idx], markersize = 7, zorder = 0)
        ax.fill_between(self.x, pred_interp - 2 * std_interp, pred_interp + 2 * std_interp, \
                        facecolor = self.light_colors[curve_idx],
                        edgecolor = self.colors[curve_idx], lw = edgewidth, zorder = 0)
        
        # Format plot
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticks([0.25, 0.75], minor = True)
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_xlabel(r'$x$')
        ax.legend()
        fig.tight_layout()
        
    def PlotMD(self, plot_type = 'box'):
        # plots the Mahalanobis distance in one of two ways (box-and-whisker or histogram)
        try:
            # kernel for GP fit
            self.kernel_fit = RBF(length_scale = self.ls) + \
                            WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

            # fits GP and extracts error bars
            self.gp_diagnostic = gm.ConjugateGaussianProcess(kernel = self.kernel_fit, \
                        center = self.center, disp = self.disp, df = self.df, scale = self.scale, \
                        n_restarts_optimizer = 2, random_state = 32)
            self.gp_diagnostic.fit(self.X[self.x_train_mask], self.coeffs[self.x_train_mask])
            self.pred, self.std = self.gp_diagnostic.predict(self.X, return_std = True)
            self.underlying_std = np.sqrt(self.gp_diagnostic.cov_factor_)

            # extracts underlying covariance matrix and calculates the diagnostics
            self.mean_underlying = self.gp_diagnostic.mean(self.X[self.x_valid_mask])
            self.cov_underlying = self.gp_diagnostic.cov(self.X[self.x_valid_mask])
            self.gdgn = gm.GraphicalDiagnostic(self.coeffs[self.x_valid_mask], \
                        self.mean_underlying, self.cov_underlying, colors = self.colors,
                        gray = gray, black = softblack)
            
            # plots the Mahalanobis distance
            if plot_type == 'box':
                fig, ax = plt.subplots(figsize = (1.5, 3.0))
                ax = self.gdgn.md_squared(type = plot_type, trim = False, title = None, \
                    xlabel = r'$\mathrm{D}_{\mathrm{MD}}^2$')
            elif plot_type == 'hist':
                fig, ax = plt.subplots(figsize=(9, 3.2))
                ax = self.gdgn.md_squared(type = plot_type, title = None, \
                    xlabel = r'$\mathrm{D}_{\mathrm{MD}}^2$')
                ax.set_ylim(0, 25)
            else:
                return 0

            offset_xlabel(ax)
#             fig.tight_layout()
        except:
            print("The Mahalanobis distance could not be calculated at one or more orders.")
    
    def PlotPC(self):
        # plots the pivoted Cholesky decomposition in one of two ways
        
        try:
            # kernel for GP fit
            self.kernel_fit = RBF(length_scale = self.ls) + \
                                WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

            # fits GP and extracts error bars
            self.gp_diagnostic = gm.ConjugateGaussianProcess(kernel = self.kernel_fit, center = self.center, \
                        disp = self.disp, df = self.df, scale = self.scale, n_restarts_optimizer = 2, \
                        random_state = 32)
            self.gp_diagnostic.fit(self.X[self.x_train_mask], self.coeffs[self.x_train_mask])
            self.pred, self.std = self.gp_diagnostic.predict(self.X, return_std = True)
            self.underlying_std = np.sqrt(self.gp_diagnostic.cov_factor_)

            # extracts underlying covariance matrix and calculates the diagnostics
            self.mean_underlying = self.gp_diagnostic.mean(self.X[self.x_valid_mask])
            self.cov_underlying = self.gp_diagnostic.cov(self.X[self.x_valid_mask])
            self.gdgn = gm.GraphicalDiagnostic(self.coeffs[self.x_valid_mask], \
                        self.mean_underlying, self.cov_underlying, colors = self.colors,
                        gray = gray, black = softblack)

            # plots the pivoted Cholesky decomposition
            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
#             with plt.rc_context({"text.usetex": True}):
                fig, ax = plt.subplots(figsize = (3.2, 3.2))
                self.gdgn.pivoted_cholesky_errors(ax = ax, title = None)
                ax.set_xticks([2, 4, 6, 8, 10, 12])
                ax.set_xticks([1, 3, 5, 7, 9, 11], minor = True)
                ax.set_yticks([-2, -1, 0, 1, 2])
                ax.text(0.04, 0.967, r'$\mathrm{D}_{\mathrm{PC}}$', bbox = text_bbox, \
                        transform = ax.transAxes, va = 'top', ha = 'left')
                fig.tight_layout()
                plt.show()
        except:
            print("The pivoted Cholesky decomposition could not be calculated at one or more orders.")
    
    def PlotTruncations(self):
        # plots the data summed to each order we're interested in
        
        try:
            # kernel for fit
            self.kernel_fit = RBF(length_scale = self.ls) + \
                                WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

            # fits truncation GP to data given a mask
            self.gp_trunc = gm.TruncationGP(kernel = self.kernel_fit, ref = self.ref, \
                        ratio = self.ratio, center = self.center, disp = self.disp, \
                        df = self.df, scale = self.scale)
            self.gp_trunc.fit(self.X[self.x_train_mask], y = self.data[self.x_train_mask], \
                        orders = self.orders_array)

            # extracts truncation error for each x-value
            self.norm_trunc_cov = self.gp_trunc.cov(self.X[self.x_valid_mask], start = 0, end = 0)
            self.norm_residuals = (self.data_true[self.x_valid_mask, None] - \
                                   self.data[self.x_valid_mask]) / \
                    (self.ratio**(self.orders_array+1) / np.sqrt(1 - self.ratio**2))
            self.gr_dgn_trunc = gm.GraphicalDiagnostic(self.norm_residuals, \
                            mean = np.zeros(self.x[self.x_valid_mask].shape[0]), \
                            cov = self.norm_trunc_cov, colors = self.colors, gray = gray, \
                            black = softblack)

            fig, axes = plt.subplots(math.ceil(self.n_orders / 2), 2, sharex = True, sharey = True, \
                figsize = (3.9, 3.2))

            # plots curves with error
            for i, n in enumerate(self.orders_array):
                _, std_trunc = self.gp_trunc.predict(self.X, order = n, return_std = True, \
                                    kind = 'trunc')

                for j in range(i, self.n_orders):
                    ax = axes.ravel()[j]
                    ax.plot(self.x, self.data[:, i], zorder = i-5, c = self.colors[i])
                    ax.fill_between(self.x, self.data[:, i] + 2 * std_trunc, \
                                    self.data[:, i] - 2 * std_trunc, zorder = i-5, \
                                    facecolor = self.light_colors[i], edgecolor = self.colors[i], \
                                    lw = edgewidth)
                ax = axes.ravel()[i]
                ax.plot(self.x, self.data_true, color = softblack, lw = 1, ls = '--')
                ax.set_xticks([0.25, 0.5, 0.75])
                ax.set_xticks(self.x[self.x_valid_mask], minor = True)
                ax.set_xticklabels([0.25, 0.5, 0.75])
                ax.set_yticks([0, 10, 20])
                ax.set_yticks([-10, 0, 10, 20, 30])
                ax.set_ylim(-15, 37)

            axes[1, 0].set_xlabel(r'$x$')
            axes[1, 1].set_xlabel(r'$x$')
            fig.tight_layout(h_pad=0.3, w_pad=0.3)
            
        except:
            print("The truncation error curves could not be calculated at one or more orders.")
    
    def PlotCredibleIntervals(self):
        # plots credible intervals ("weather plot") for each order we're interested in
        
        try:
            # kernel for fit
            self.kernel_fit = RBF(length_scale = self.ls) + \
                                WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

            # truncation GP
            self.gp_trunc = gm.TruncationGP(kernel = self.kernel_fit, ref = self.ref, \
                        ratio = self.ratio, center = self.center, disp = self.disp, \
                        df = self.df, scale = self.scale)
            self.gp_trunc.fit(self.X[self.x_train_mask], y = self.data[self.x_train_mask], \
                        orders = self.orders_array)

            # extracts truncation error for each x-value
            self.norm_trunc_cov = self.gp_trunc.cov(self.X[self.x_valid_mask], start = 0, end = 0)
            self.norm_residuals = (self.data_true[self.x_valid_mask, None] - \
                                   self.data[self.x_valid_mask]) / \
                    (self.ratio**(self.orders_array+1) / np.sqrt(1 - self.ratio**2))
            self.gr_dgn_trunc = gm.GraphicalDiagnostic(self.norm_residuals, \
                            mean = np.zeros(self.x[self.x_valid_mask].shape[0]), \
                            cov = self.norm_trunc_cov, colors = self.colors, gray = gray, \
                            black = softblack)

            fig, ax = plt.subplots(figsize = (3.4, 3.2))

            # plots the curves
            for i, n in enumerate(self.orders_array):
                norm_residuals_alt = self.data_true[self.x_valid_mask] - \
                                            self.data[self.x_valid_mask][:,i]
                norm_trunc_cov_alt = self.gp_trunc.cov(self.X[self.x_valid_mask], start = n+1)
                gr_dgn_trunc_alt = gm.GraphicalDiagnostic(
                    norm_residuals_alt, mean = np.zeros(self.x[self.x_valid_mask].shape[0]), \
                    cov = norm_trunc_cov_alt, colors = [self.colors[i]], gray = gray, black = softblack)   
                gr_dgn_trunc_alt.credible_interval(
                   np.linspace(1e-5, 1, 100), band_perc = [0.68, 0.95], ax = ax, title = None,
                   xlabel = r'Credible Interval ($100\alpha\%$)', ylabel = r'Empirical Coverage ($\%$)')
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0, 20, 40, 60, 80, 100])
            fig.tight_layout()
        
        except:
            print("The credible intervals could not be calculated at one or more orders.")
        
    def PlotPosteriorPDF(self, posteriorgrid):
        # plots the posterior PDF for the ratio and correlation length of the fit GP
        
        try:
            # kernel for fit
            self.kernel_fit = RBF(length_scale = self.ls) + \
                                WhiteKernel(noise_level = self.nugget, noise_level_bounds = 'fixed')

            # fits the GP at the mask
            self.gp_trunc = gm.TruncationGP(kernel = self.kernel_fit, ref = self.ref, \
                        ratio = self.ratio, center = self.center, disp = self.disp, \
                        df = self.df, scale = self.scale)
            self.gp_trunc.fit(self.X[self.x_train_mask], y = self.data[self.x_train_mask], \
                        orders = self.orders_array)

            # reads the posterior grid points to the class
            self.posteriorgrid = posteriorgrid
            self.ls_vals = self.posteriorgrid.x_vals
            self.ratio_vals = self.posteriorgrid.y_vals

            # Compute the log likelihood for values on this grid. 
            self.ls_ratio_loglike = np.array([[
                self.gp_trunc.log_marginal_likelihood(theta = [ls_,], ratio = ratio_val) \
                for ls_ in np.log(self.ls_vals)]
                for ratio_val in self.ratio_vals])

            # Makes sure that the values don't get too big or too small
            self.ls_ratio_like = np.exp(self.ls_ratio_loglike - np.max(self.ls_ratio_loglike))

            # Now compute the marginal distributions
            self.ratio_like = np.trapz(self.ls_ratio_like, x = self.ls_vals, axis = -1)
            self.ls_like = np.trapz(self.ls_ratio_like, x = self.ratio_vals, axis = 0)

            # Normalize them
            self.ratio_like /= np.trapz(self.ratio_like, x = self.ratio_vals, axis = 0)
            self.ls_like /= np.trapz(self.ls_like, x = self.ls_vals, axis = 0)

            with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
#             with plt.rc_context({"text.usetex": True}):
                cmap_name = 'Blues'
                cmap = mpl.cm.get_cmap(cmap_name)

                # Setup axes
                fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(ratio = 5, height = 3.4)

                # Plot contour
                ax_joint.contour(self.ls_vals, self.ratio_vals, self.ls_ratio_like,
                                 levels = [np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999],
                                 cmap = cmap_name, vmin = -0.05, vmax = 0.8, zorder = 1)

                # Now plot the marginal distributions
                ax_marg_y.plot(self.ratio_like, self.ratio_vals, c = cmap(0.8), lw = 1)
                ax_marg_y.fill_betweenx(self.ratio_vals, np.zeros_like(self.ratio_like),
                                        self.ratio_like, facecolor = cmap(0.2), lw = 1)
                ax_marg_x.plot(self.ls_vals, self.ls_like, c = cmap(0.8), lw = 1)
                ax_marg_x.fill_between(self.ls_vals, np.zeros_like(self.ls_vals),
                                       self.ls_like, facecolor = cmap(0.2), lw = 1)

                # Formatting
                ax_joint.set_xlabel(r'$\ell$')
                ax_joint.set_ylabel(r'$Q$')
                ax_joint.axvline(self.ls, 0, 1, c = gray, lw = 1, zorder = 0)
                ax_joint.axhline(self.ratio, 0, 1, c = gray, lw = 1, zorder = 0)
                ax_joint.margins(x = 0, y = 0.)
                ax_joint.set_xlim(0.05, 0.35)
                ax_joint.set_xticks([0.1, 0.2, 0.3])
                ax_joint.set_xticks([0.15, 0.25], minor = True)
                ax_joint.set_yticks([0.4, 0.5, 0.6])
                ax_joint.set_yticks([0.35, 0.45, 0.55, 0.65], minor = True)
                ax_marg_x.set_ylim(bottom = 0);
                ax_marg_y.set_xlim(left = 0);
                ax_joint.text(0.95, 0.95, r'pr$(\ell, Q \,|\, \vec{\mathbf{y}}_k)$', ha='right', \
                              va='top', transform = ax_joint.transAxes, bbox = text_bbox)
                plt.show()
        except:
            print("The posterior probability distribution could not be calculated.")
        
    def change_order(self, change_order_array, change_ratio_array, change_ls_array, \
                     change_sd_array, seed_array):
        # can create coefficient curves for some order(s) with a different correlation length, 
        # ratio, variance, etc., from the GP from which all other orders were calculated
        
        # reads the information about the changed orders to the class for ease of access
        self.change_order_array = change_order_array
        self.change_ratio_array = change_ratio_array
        self.change_ls_array = change_ls_array
        self.change_sd_array = change_sd_array
        self.seed_array = seed_array
        
        coeffs_all = self.coeffs_all
        
        # calculates the new curve(s) for some seed(s) and swaps them into the array of 
        # coefficients
        for i, order in enumerate(change_order_array):
            kernel_bad = RBF(length_scale=change_ls_array[i], length_scale_bounds='fixed') + \
                            WhiteKernel(noise_level=self.nugget, noise_level_bounds='fixed')
            gp_bad = gm.ConjugateGaussianProcess(kernel=kernel_bad, center=self.center, \
                            df=np.inf, scale=change_sd_array[i], nugget=0)
            coeffs_bad = - gp_bad.sample_y(self.X, n_samples = 1, random_state=seed_array[i])
            coeffs_all[:, order] = coeffs_bad[:, 0]
            
            self.colors[order] = np.array([0, 0, 0])
            self.light_colors[order] = np.array([128 / 255., 128 / 255., 128 / 255.])
        
        # with new coefficients, calculates the data, differences, etc., for all orders
        self.coeffs_all = coeffs_all
        self.data_all = gm.partials(self.coeffs_all, self.ratio, ref=self.ref, \
                                    orders=self.orders_all_array)
        self.diffs_all = np.array([self.data_all[:, 0], *np.diff(self.data_all, axis=1).T]).T
        # Get the "all-orders" curve
        self.data_true = self.data_all[:, -1]
        self.coeffs = self.coeffs_all[:, :self.n_orders]
        self.data = self.data_all[:, :self.n_orders]
        self.diffs = self.diffs_all[:, :self.n_orders]
    
    def print_changed_orders(self):
        # prints out all the information on the orders whose hyperparameters have been changed
        print("Adjusted orders: " + str(self.change_order_array))
        print("Adjusted ratios: " + str(self.change_ratio_array))
        print("Adjusted correlation lengths: " + str(self.change_ls_array))
        print("Adjusted variances: " + str(self.change_sd_array))
        print("Adjusted seeds: " + str(self.seed_array))
        return 0