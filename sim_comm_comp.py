import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Type
from copy import deepcopy
from nlmpy import nlmpy
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from scipy.stats import norm, multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import rasterio as rio
import xarray as xr
import rioxarray as rxr

# TODO:

    # DEBUGGING:
        # check again how GDM splines/etc are typically plotted

        # get rid of alpha? once decided, update functions as well as
        # docstrings and plot

        # add type annotation!

    # may need to reconsider how alpha div is determined/alpha raster is used
    # because right now there are still a lot of cells with 0 species niche
    # centers located there, and thus no guarantee that at least 1 species will
    # occur there
    # --> OKAY TO JUST LEAVE ALPHA AS EMERGENT PROP AFTER ALL?

    # I wonder if GDM won't capture the patterns until survey is actually cast
    # as proper abundances rather than simple binary pres/abs?? though I think
    # GDM should work just fine with pres/abs...

    # work through numerical artefacts:
        # using normal distribution across the [0,1] interval, so many will
        # extend outside it

        # artefacts caused by the 1-inverse logic approach to determining
        # sigma? is that approach justifiable?

        # need to use gamma to somehow constrain min and max alpha values that
        # occur on the map? (i.e., least and most diverse communities)

        # does ignoring spatial autocorrelation in presence/absence
        # determination create any major problems?

        # does ignoring the chance correlation between environmental layers
        # (e.g., by calculating overall presence prob as the prod of
        # independent presence probs on each axis) create any problems?

        # should max prob presence, even at the cell defining the niche center,
        # be <1.0?

    # thoroughly review ChatGPT-derived I-spline code and improve or replace

    # add sampling schemes

    # add algorithm for change over time


'''
simulate community composition at any or all points in a simulated landscape

input parameters include:
    - landscape dimensions
    - number of environmental layers
    - spatial autocorrelation of each of those layers
    - splines describing relation of compositional turnover to each of those
      layers
    - an α-diversity layer (map of local species richness)
      (defaults to a layer parameterized same as the first environmental layer)
    - γ-diversity (total species count on landscape)
'''

########################
# FUNCTIONS AND CLASSES:
########################

#-------------
# type aliases
#-------------

numerical = Union[float, int]
vectorlike = Union[List[numerical], Tuple[numerical], np.ndarray]
rasterlike = Union[np.ndarray, xr.core.dataarray.DataArray]


#--------
# classes
#--------

class ISpline:
    '''
    I-Spline class with method for getting its approximate slope at any x value
    '''
    def __init__(self,
                 x: Union[List[numerical], np.ndarray],
                 y: Union[List[numerical], np.ndarray],
                 i: int,
                 knots: np.ndarray = np.linspace(0, 1, 8),
                 degree: int = 3,
                ) -> None:
        self.i = i
        # validate and save input x and y values
        self._input_x = np.array(x)
        self._input_y = np.array(y)
        # check bounds
        assert np.all(self._input_x>=0) and np.all(self._input_x<=1)
        assert np.all(self._input_y>=0) and np.all(self._input_y<=1)
        # fit and validate the I-spline x and y values
        spline_x, spline_y = make_I_spline(x=self._input_x,
                                           y=self._input_y,
                                           knots=knots,
                                           degree=degree,
                                          )
        self.x  = spline_x
        self.y = spline_y
        # placeholders for GDM-fitted values
        self.x_gdm = None
        self.y_gdm = None
        # check bounds and monotonicity
        assert np.all(self.x>=0) and np.all(self.x<=1)
        assert np.all(self.y>=0) and np.all(self.y<=1)
        assert np.all((self.y[1:] - self.y[:-1])>=0)

    def get_approx_slope(self,
                         x: numerical,
                        ) -> float:
        '''
        calculate simple local approximation of slope at value x
        '''
        ind = np.argmin(np.abs(self.x - x))
        assert ind>=0 and ind<= len(self.x-1)
        ind_lo, ind_hi = np.clip([ind-1, ind+1], a_min=0, a_max=len(self.x)-1)
        Δx = self.x[ind_hi] - self.x[ind_lo]
        Δy = self.y[ind_hi] - self.y[ind_lo]
        return Δy/Δx

    def add_GDM_fit(self,
                    fitted_spline_x: np.ndarray,
                    fitted_spline_y: np.ndarray,
                   ) -> None:
        '''
        add attributes to store a GDM-fitted spline
        '''
        self.x_gdm = fitted_spline_x
        self.y_gdm = fitted_spline_y

    def plot(self,
             ax: Optional[mpl.axes._axes.Axes] = None,
             legend: bool = False,
            ) -> None:
        '''
        make simple plot of the fitted spline
        '''
        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
        ax.plot(self._input_x,
                self._input_y,
                'or',
                label='monotonic input data',
               )
        ax.plot(self.x, self.y,
                label='Monotonic I-spline Fit',
                linewidth=2,
                color='black'
               )
        if self.x_gdm is not None and self.y_gdm is not None:
            ax.plot(self.x_gdm, self.y_gdm, ':',
                    label='GDM-fitted I-spline',
                    linewidth=2,
                    color='blue',
                    alpha=0.5,
                   )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"env. spline {self.i}", size=14)
        if legend:
            ax.legend()
        ax.grid(True)


class Sim:
    '''
    overall class for the simulation
    '''
    def __init__(self,
                 env: List[rasterlike],
                 splines: list[Type[ISpline]],
                 alpha: rasterlike,
                 gamma: int,
                 survey_points: List[Tuple[float]],
                 use_multivar_normal_niche: bool = False,
                 prob_pres_thresh_round_to_1: Optional[float] = None,
                 max_poisson_lambda_val: int = 1000,
                 gdm_pres_abund: bool = True,
                 verbose: bool = False,
                 debug: bool = False,
                 timeit: bool = True,
                ) -> None:
        # store behavioral params
        self._verbose = verbose
        self._debug = debug
        self._timeit = timeit
        self._use_multivar_normal_niche = use_multivar_normal_niche
        self._prob_pres_thresh_round_to_1 = prob_pres_thresh_round_to_1
        # store fixed params
        self.env = env
        self.splines = splines
        self.alpha = alpha
        self.gamma = gamma
        self.points = survey_points
        # run and save the simulation
        if self._timeit:
            start = time.time()
        (spp,
         spp_max_poisson_lambdas,
         surveys,
         samples,
         gdm_splines,
         gdm_pca_rast) = run_sim(env=ENV,
                                 splines=SPLINES,
                                 alpha=ALPHA,
                                 gamma=GAMMA,
                                 use_multivar_normal_niche=self._use_multivar_normal_niche,
                                 prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                 max_poisson_lambda_val=max_poisson_lambda_val,
                                 gdm_pres_abund=gdm_pres_abund,
                                 survey_points=POINTS,
                                 verbose=self._verbose,
                                 debug=self._debug,
                                )
        if self._timeit:
            stop = time.time()
            self._runtime_sec = stop-start
            print((f"\n\nSIMULATION RAN IN {np.round(self._runtime_sec/60, 2)} "
                   "MINUTES.\n\n"))
        self.spp = spp
        self.spp_max_poisson_lambdas = spp_max_poisson_lambdas
        self.surveys = surveys
        self.samples = samples
        self.gdm_splines = gdm_splines
        self.gdm_pca_rast = gdm_pca_rast
        # useful hidden attributes
        self._env_cmaps = ['Reds', 'Greens', 'Blues']


    def plot(self,
             scatter_survey_points: bool = True,
             save: bool = False,
            ) -> None:
        '''
        plot the results of a simulation
        '''
        fig = plt.figure(figsize=(16,16))
        gs = fig.add_gridspec(80, 100)

        # plot environment rasters
        axwidth = int(100/len(self.env))-1
        for i, e in enumerate(self.env):
            ax = fig.add_subplot(gs[:20,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
            img = ax.imshow(e, vmin=0, vmax=1, cmap=self._env_cmaps[i])
            plt.colorbar(img)
            # add survey points
            if scatter_survey_points:
                for point in self.points:
                    ax.scatter(point[0],
                               point[1],
                               color='white',
                               edgecolor='black',
                               alpha=0.8,
                               s=24,
                              )
            ax.set_title(f"env. variable {i}", size=14)

        # plot their splines
        for i, spline in enumerate(self.splines):
            spline.add_GDM_fit(self.gdm_splines[f"x.env_rast_{i+1}"],
                               self.gdm_splines[f"y.env_rast_{i+1}"],
                              )
            ax = fig.add_subplot(gs[25:40,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
            spline.plot(ax=ax,
                       legend=i==(len(self.env)-1),
                       )

        # TODO: DELETE ME IF DROPPING ALPHA RAST APPROACH
        # plot input alpha raster
        #ax = fig.add_subplot(gs[50:, :30])
        #ax.imshow(self.alpha, vmin=0, vmax=1)
        #ax.set_title('input α-diversity raster (scaled [0,1])', size=14)

        # plot raster of observed alpha values at all surveyed cells
        ax = fig.add_subplot(gs[50:, 35:65])
        survey_len_arr = np.ones(self.env[0].shape)*np.nan
        for pt, survey in zip(self.points, self.surveys):
            survey_len_arr[int(pt[0]), int(pt[1])] = len(survey)
        survey_lengths = [len(survey) for survey in self.surveys]
        img = ax.imshow(survey_len_arr,
                        vmin=min(survey_lengths),
                        vmax=max(survey_lengths),
                       )
        plt.colorbar(img)
        ax.set_title('α-diversity at surveyed sites', size=14)

        # plot PCA rast from GDM transform
        ax = fig.add_subplot(gs[50:, 70:])
        self.gdm_pca_rast.plot.imshow(ax=ax)
        # add survey points
        if scatter_survey_points:
            for point in self.points:
                ax.scatter(point[0],
                           point[1],
                           color='white',
                           edgecolor='black',
                           alpha=0.8,
                           s=24,
                          )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('top 3 PCs from GDM transform')

        # format plot and save
        fig.subplots_adjust(hspace=25,
                            wspace=25,
                           )
        fig.show()
        if save:
           fig.savefig('comm_sim_res.png',
                        dpi=500,
                       )

    def plot_expec_vs_obser_distr(self,
                                  sp: int,
                                  cmap: str = 'viridis',
                                  save: bool = False,
                                 ) -> None:
        '''
        plot both the expected and observed distribution the species
        '''
        # get species' niche
        niche = self.spp[sp]
        # calculate map of expected distribution
        expec = np.zeros(self.env[0].shape)
        # calculate presence probability at all cells
        for i in range(self.env[0].shape[0]):
            for j in range(self.env[0].shape[1]):
                prob = calc_pres_prob(env_vals=[e[i, j] for e in self.env],
                                      niche=self.spp[sp],
                                      use_multivar_normal=self._use_multivar_normal_niche,
                                      prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                     )
                expec[i, j] = prob
        # calculate map of all pixels where species is observed
        # (setting pixels without surveys to NaNs)
        obser = np.zeros(self.env[0].shape)
        for pt, survey in zip(self.points, self.surveys):
            if sp in survey:
                obser[int(pt[0]), int(pt[1])] = survey[sp]
        for i in range(self.env[0].shape[0]):
            for j in range(self.env[0].shape[1]):
                if (i+0.5, j+0.5) not in self.points:
                    obser[i, j] = np.nan
        # plot both
        show_fig = False
        fig = plt.figure(figsize=(14,8))
        gs = fig.add_gridspec(nrows=80, ncols=140)
        axs_env = [fig.add_subplot(gs[:25,
                        (i*20)+(i*5):(i+1)*20+(i*5)]) for i in range(len(self.env))]
        ax_expec = fig.add_subplot(gs[25:, :55])
        ax_obser = fig.add_subplot(gs[25:, 85:])
        for i, e in enumerate(self.env):
            ax = axs_env[i]
            img = ax.imshow(e, vmin=0, vmax=1, cmap=self._env_cmaps[i])
            plt.colorbar(img)
            # get and plot niche center loc
            niche_cent = np.unravel_index(np.argmin(np.abs(self.env[i]-
                                                           self.spp[sp][i][0])),
                                          self.env[i].shape)
            ax.scatter(niche_cent[0],
                       niche_cent[1],
                       marker='*',
                       s=15,
                       c='yellow',
                       edgecolor='black',
                       linewidth=0.25,
                       alpha=0.8,
                      )
            ax.set_title(f"env. variable {i}", size=14)
        img = ax_expec.imshow(expec,
                              cmap=cmap,
                              vmin=0,
                              vmax=1,
                             )
        plt.colorbar(img)
        ax_expec.set_title(f'expected distribution for species {sp}')
        ax_obser.imshow(obser,
                         cmap=cmap,
                         vmin=0,
                         vmax=1,
                        )
        ax_obser.set_title(f'observed distribution for species {sp}')
        fig.subplots_adjust(hspace=0.25,
                            wspace=0.25,
                           )
        fig.show()
        if save:
           fig.savefig(f'comm_sim_sp{sp}_expec_vs_obser_distr.png',
                        dpi=500,
                       )


#----------
# functions
#----------

def one_minus_inv_logit(x: Union[int, float]) -> float:
    '''
    returns 1 - inverse logit of x
    '''
    return 1 - (np.exp(x)/(1+np.exp(x)))


def minmax_scale_rast(rast: rasterlike,
                      by_band : bool = True) -> Union[np.ndarray,
                                                xr.core.dataarray.DataArray]:
    '''
    recast an array of values to the [0, 1] interval using min-max scaling
    (defaults to rescaling each band separately, assuming bands are the zeroth
    index)
    '''
    out = deepcopy(rast)
    if by_band:
        for i in range(out.shape[0]):
            out[i] = ((out[i]-np.nanmin(out[i]))/
                       (np.nanmax(out[i])-np.nanmin(out[i])))
    else:
        out = (out-np.nanmin(out))/(np.nanmax(out)-np.nanmin(out))
    return out


def draw_random_survey_points(dims: Tuple[Union[float, int]],
                              n: int) -> List[Tuple[float]]:
    '''
    draw a set of random survey points within a raster whose coordinates range
    from 0 to dim-1 in both axes in dims; each point will be in a separate
    pixel, so n must not exceed the number of pixels
    '''
    assert n > 0 and n<= np.prod(dims)
    X, Y = np.meshgrid(range(dims[0]), range(dims[1]))
    xs = X.ravel()
    ys = Y.ravel()
    pts = [*zip(xs, ys)]
    np.random.shuffle(pts)
    idxs = np.random.choice(range(len(pts)), replace=False, size=n)
    # NOTE: add 0.5 to all points, to place them in cell centers
    rand_pts = [tuple(np.array(pts[idx])+0.5) for idx in idxs]
    return rand_pts


def make_I_spline_basis(x: Union[List[numerical], np.ndarray],
                        knots: np.ndarray,
                        degree: int,
                       ) -> Tuple[np.ndarray]:
    # TODO: MORE THOROUGHLY REVIEW CODE GENERATED BY CHATGPT
    '''
    NOTE: QUICK I-SPLINE CODE GENERATED BY CHATGPT!
          works for now but not prefect (e.g., sometimes fits spline y values
          above [0, 1] interval). Works to get me moving forward now, but needs
          to be more carefully reviewed and then either integrated or replaced.

    Generate I-spline basis from B-spline basis by integrating.
    '''
    n_bases = len(knots) - degree - 1
    b_splines = [BSpline.basis_element(knots[i:i+degree+2], extrapolate=False) for i in range(n_bases)]
    x_eval = np.linspace(min(x), max(x), 200)
    dx = x_eval[1] - x_eval[0]
    i_splines = np.zeros((len(x_eval), n_bases))
    for i, b in enumerate(b_splines):
        y_b = np.nan_to_num(b(x_eval))
        i_splines[:, i] = np.cumsum(y_b) * dx
        if np.max(i_splines[:, i]) > 0:
            i_splines[:, i] /= np.max(i_splines[:, i])
    return x_eval, i_splines


def make_I_spline(x: Union[List[numerical], np.ndarray],
                  y: Union[List[numerical], np.ndarray],
                  knots: np.ndarray,
                  degree: int,
                 ) -> Tuple[np.ndarray]:
    '''
    fit I-spline to provided x and y arrays
    '''
    # I-spline basis
    x_eval, i_splines = make_I_spline_basis(x, knots, degree)
    assert np.all((x_eval[1:]-x_eval[:-1])>=0)
    # evaluate basis functions at original x
    X_basis = np.zeros((len(x), i_splines.shape[1]))
    for i in range(i_splines.shape[1]):
        X_basis[:, i] = np.interp(x, x_eval, i_splines[:, i])
    # fit monotonic spline with non-negative least squares
    res = lsq_linear(X_basis, y, bounds=(0, np.inf))
    coefs = res.x
    # predict over x_eval
    y_fit = i_splines @ coefs
    # NOTE: clipping y values to the [0, 1] interval, but not constraining to
    #       vary between 0 and 1 because that would allow no insignificant
    #       relationships (i.e., flat lines)
    y_fit = np.clip(y_fit, a_min=0, a_max=1)
    return x_eval, y_fit


def create_species(gamma: int,
                   alpha: rasterlike,
                   env: List[rasterlike],
                   splines: Type[ISpline],
                   min_niche_sigma: float = 0.05,
                   max_niche_sigma: float = 0.50,
                   max_poisson_lambda_val: int = 1000,
                   debug: bool = False,
                  ) -> Tuple[Dict[int, List[Tuple[float]]], Dict[int, int]]:
    '''
    create a dict of all species' ecological niches
    (i.e., μ and σ values for all environmental layers)
    '''
    # melt alpha raster vals and transform so they sum to 1
    alpha_ravel = alpha.ravel()
    alpha_trans = alpha_ravel/np.sum(alpha_ravel)
    assert np.allclose(np.sum(alpha_trans), 1)
    # melt env rasters' vals too
    env_ravel = [e.ravel() for e in env]
    # use alpha vals to draw all species niche centers
    # (centers will always be a vector of values occurring on each of the
    # environmental layers in the landscape)
    spp_mus = []
    for sp in range(gamma):
        mu_inds = np.random.choice(a=range(len(alpha_trans)),
                                   size=len(env),
                                   replace=True,
                                   # TODO: DECIDE IF LEAVE THIS OUT...
                                   #p=alpha_trans,
                                  )
        spp_mus.append([e[i] for e, i in zip(env_ravel, mu_inds)])
    # use spline slope at each μ to draw each σ
    # (Bush et al. 2018 calculate niche width as 3.09/slope,
    # where 3.09 is the ecological distance at which two communities are
    # expected to have 95% dissimilarity (because 1-(1/exp(3.09))=0.954);
    # I think they're dealing with standardized values, whereas we're dealing
    # with values constrained to [0, 1] and we want our niche widths expressed
    # in those terms;
    # TODO: in lieu of a better solution that I should come up with later I'm
    #       just jamming in a simple function (1-inv_logit) to bound sigmas
    #       between min and max values
    spp = {}
    for sp, mus in zip(range(gamma), spp_mus):
        niche = []
        for i, mu in enumerate(mus):
            slope = splines[i].get_approx_slope(mu)
            sigma = one_minus_inv_logit(slope)
            sigma = np.clip(sigma, a_min=min_niche_sigma, a_max=max_niche_sigma)
            niche.append((mu, sigma))
        spp[sp] = niche
    # draw species' lambdas for Poisson distributions determining survey
    # results (will be multiplied by probability of presence at a location, so
    # this is the maximum value that a Poisson draw will take in a location
    # where probability of presence goes to 1.0)
    spp_poisson_lambda_vals = dict(zip(spp,
                                       np.random.uniform(1,
                                                         max_poisson_lambda_val,
                                                         len(spp))))
    return spp, spp_poisson_lambda_vals


def calc_pres_prob(env_vals: vectorlike,
                   niche: Tuple[float],
                   use_multivar_normal: bool = False,
                   prob_pres_thresh_round_to_1: Optional[float] = None,
                  ) -> float:
    '''
    for a location described by the given environmental values,
    calculate the probability of presence of a species with the given niche
    '''
    if not use_multivar_normal:
        # list of probability densities extracted from the normal distributions
        # describing the species' niches on each environmental axis
        probs = []
        for n, e in enumerate(env_vals):
            # get probability of presence for this axis by determining the
            # probability of drawing, from the species' niche distribution
            # on this environmental axis, a value equally or more extreme
            # than the survey position's environmental value
            # NOTE: calculating and then subtracting difference between survey
            #       location's environmental value and niche center, then
            #       subtracting that from the niche center in the CDF
            #       calculation, thus getting the probability of a value being
            #       that far below the niche center; then multiply by two to
            #       get two-tailed probability of a value as extreme or more so
            diff = np.abs(e-niche[n][0])
            prob_n = 2 * (norm.cdf(x=niche[n][0]-diff,
                                   loc=niche[n][0],
                                   scale=niche[n][1],
                                  ))
            assert 0 <= prob_n <= 1
            probs.append(prob_n)
        # determine overall probability of presence as the product of all
        # probabilities (i.e., the joint probability across all
        # environmental axes, treating the axes as if they are independent...
        # NOTE: ... even though in reality we could actually fold in cross-layer
        #       correlation to account for chance non-independence between
        #       environmental axes...)
        # NOTE: ... we also ignore spatial autocorrelation of presence in real
        #       species by ignoring any information about whether or not the
        #       species has been determined present in proximal locations...
        prob = np.prod(probs)**(1/3)
    else:
        # get arrays of niche centers and niche widths
        mus = np.array([n[0] for n in niche])
        sigmas = np.array([n[1] for n in niche])
        # construct covariance matrix (NOTE: without covariance between layers!)
        covar = np.zeros([len(mus)]*2)
        covar[np.diag_indices_from(covar)] = sigmas
        # get probability of presence using the cumulative distribution
        # function of the multivariate normal described by the species' niche
        # distributions on all axes (modeled as the probability of drawing
        # from within the species' multivariate normal niche space
        # a series of environmental values equally extreme as or more extreme
        # than the environmental values observed as the survey position)
        # NOTE: calculating and then subtracting difference between survey
        #       location's environmental values and multivariate niche center,
        #       then subtracting that from the niche center in the CDF
        #       calculation, thus getting the probability of a value being
        #       that far below the niche center; then multiplying by two to
        #       get the two-tailed probability of a value as extreme or more so)
        diffs = np.abs(env_vals-mus)
        distr = multivariate_normal(mean=mus, cov=covar, allow_singular=False)
        prob = 2 * distr.cdf(x=mus-diffs)
        assert 0 <= prob <= 1
    # round values to 1 above a certain value, if required
    if (prob_pres_thresh_round_to_1 is not None and
        prob >= prob_pres_thresh_round_to_1):
        prob = 1
    return prob


def do_survey(env: List[rasterlike],
              spp: Dict[int, List[Tuple[float]]],
              spp_max_poisson_lambdas: Dict[int, int],
              i: float,
              j: float,
              max_prob_pres: float = 1.0,
              prob_pres_thresh_round_to_1: Optional[float] = None,
              use_multivar_normal: bool = False,
              debug: bool = False,
             ) -> Dict[int, int]:
    '''
    use the list of environmental layers provided and the dict of species
    and their niches to survey species composition at grid cell i,j
    '''
    # list to store all species present
    survey = {}
    # get environmental values at point grid cell i,j
    # NOTE: points sit at cell centers, so int() converts to their cell indices
    env_vals = [e[int(i), int(j)] for e in env]
    for sp, niche in spp.items():
        # calculate presence probability
        prob = calc_pres_prob(env_vals,
                              niche,
                              use_multivar_normal=use_multivar_normal,
                              prob_pres_thresh_round_to_1=prob_pres_thresh_round_to_1,
                             )
        # determine presence as a Bernoulli draw on that probability
        # NOTE: ... treating all layers as independent, even though
        #       in reality we could actually fold in cross-layer
        #       correlation to account for chance non-independence between
        #       environmental axes...)
        # NOTE: ... we also ignore spatial autocorrelation of presence in real
        #       species by ignoring any information about whether or not the
        #       species has been determined present in proximal locations...
        if np.random.binomial(1, prob):
            # if present, draw abundance from Poisson
            # NOTE: altogether, this models survey results as an
            # environmentally conditional zero-inflated Poisson
            survey[sp] = np.random.poisson(prob * spp_max_poisson_lambdas[sp])
    return survey


def prep_GDM_input_data(gamma: int,
                        surveys: List[Dict[int, int]],
                        survey_points: List[Tuple[float]],
                        env: List[rasterlike],
                        pres_abund: bool = True,
                        site_survey_filename: str = 'site_survey.csv',
                        env_rast_filename: str = 'env_rast.tif',
                       ) -> None:
    '''
    prep a set of files to input into a basic R script for running a GDM model
    '''
    # create and save 'site-survey' table
    # (sites in rows, species in columns)
    n_spp = gamma
    n_sites = len(surveys)
    # NOTE: adding 3 to include a site column and x and y columns
    add_cols = 3
    site_surv_mat = np.zeros((n_sites, n_spp+add_cols))
    # NOTE: add site column
    site_surv_mat[:, 0] = [*range(len(surveys))]
    for i, survey in enumerate(surveys):
        # NOTE: adding x and y columns (points are expressed as (i, j) matrix
        #       indices, so flip them express as (x, y) geographic coordinates)
        pt = survey_points[i]
        site_surv_mat[i, 1] = pt[1]
        site_surv_mat[i, 2] = pt[0]
        for j, abund in survey.items():
            if pres_abund:
                site_surv_mat[i, j+add_cols] = abund
            else:
                site_surv_mat[i, j+add_cols] = 1
    site_surv_df = pd.DataFrame(site_surv_mat)
    site_surv_df.columns = ['site', 'x', 'y'] + [f'spp{i}' for i in range(gamma)]
    site_surv_df.to_csv(site_survey_filename, index=False)
    # create and save environmental raster
    stack = np.stack(env)
    ydim, xdim = stack.shape[1], stack.shape[2]
    n_bands = stack.shape[0]
    dtype = stack.dtype
    crs = 'EPSG:3857' # just a stand-in projected EPSG, to avoid CRS issues
    transform = rio.transform.from_origin(0, ydim, 1, 1) # top-left corner
    with rio.open(env_rast_filename,
                  'w',
                  driver='GTiff',
                  height=ydim,
                  width=xdim,
                  count=n_bands,
                  dtype=dtype,
                  crs=crs,
                  transform=transform) as dst:
        for n in range(n_bands):
            dst.write(stack[n], n + 1)
    print("\nGDM INPUTS SAVED TO DISK.\n")


def run_GDM(gamma: int,
            surveys: List[Dict[int, int]],
            survey_points: List[Tuple[float]],
            env: List[rasterlike],
            pres_abund: bool = True,
            site_survey_filename: str = 'site_survey.csv',
            env_rast_filename: str = 'env_rast.tif',
            spline_filename: str = 'GDM_splines.csv',
            pca_rast_filename: str = 'GDM_env_rast_PCA.tif',
            ) -> Tuple[pd.core.frame.DataFrame, rasterlike]:
    '''
    prep GDM data, save to disk, execute R script to run GDM, then read in and
    return results
    '''
    # prep and save GDM input data
    prep_GDM_input_data(gamma=gamma,
                        surveys=surveys,
                        survey_points=survey_points,
                        env=env,
                        pres_abund=pres_abund,
                        site_survey_filename=site_survey_filename,
                        env_rast_filename=env_rast_filename,
                       )
    # run R script
    R_cmd = (f"Rscript --vanilla run_gdm.r {site_survey_filename} "
             f"{env_rast_filename} {str(pres_abund).upper()} "
             f"{spline_filename} {pca_rast_filename}")
    print(R_cmd)
    os.system(R_cmd)
    # read and return results
    splines = pd.read_csv(spline_filename)
    pca_rast = rxr.open_rasterio(pca_rast_filename)
    # min-max scale raster (comes in as 0-255)
    pca_rast_rescaled = minmax_scale_rast(pca_rast, by_band=True)
    return splines, pca_rast_rescaled


def draw_sample(survey: Dict[int, int],
                scheme: str,
                efforts: vectorlike,
                rel_abunds: vectorlike,
                obs_probs: vectorlike,
               ) -> None:
    '''
    TODO: WRITE ME
    draw sample from given survey using the given sampling scheme and its args,
    including:
        *efforts*: measures of effort (per site; used to extract y-values from
                   a rarefaction curve)
        *rel_abunds*: relative abundances (per species; used to extract species
                      from a rarefaction curve result)
        *obs_probs*: observation probabilities (per species; used to update
                     species results from the rarefaction curve to account for
                     uneven likelihoods of detection)
    '''
    print("OTHER SAMPLING SCHEMES NOT YET IMPLEMENTED! TRY AGAIN.")
    return

def run_sim(env: List[rasterlike],
            splines: List[Type[ISpline]],
            alpha: rasterlike,
            gamma: int,
            survey_points: List[Tuple[float]],
            min_niche_sigma: float = 0.05,
            max_niche_sigma: float = 0.50,
            use_multivar_normal_niche: bool = False,
            prob_pres_thresh_round_to_1: Optional[float] = None,
            max_poisson_lambda_val: int = 1000,
            spp_rel_abund: vectorlike = None,
            spp_observ_prob: vectorlike = None,
            sampling_schemes: List[str] = ['all'],
            gdm_pres_abund: bool = True,
            site_survey_filename: str = 'site_survey.csv',
            env_rast_filename: str = 'env_rast.tif',
            spline_filename: str = 'GDM_splines.csv',
            pca_rast_filename: str = 'GDM_env_rast_PCA.tif',
            verbose: bool = False,
            debug: bool = False,
           ) -> Type[Sim]:
    # handle sampling-scheme arguments
    assert len(sampling_schemes) > 0
    for scheme in sampling_schemes:
        assert scheme in ['all',
                          'random',
                          'rel_abund_weighted',
                          'observ_prob_biased',
                         ]
    if 'rel_abund_weighted' in sampling_schemes:
        assert spp_rel_abund is not None
        assert len(spp_rel_abund) == gamma
    if 'observ_prob_biased' in sampling_schemes:
        assert spp_observ_prob is not None
        assert len(spp_observ_prob) == gamma
    # create all species
    print(f"\n\nCREATING SPECIES...\n\n")
    spp, spp_max_poisson_lambdas = create_species(gamma=gamma,
                                                  alpha=alpha,
                                                  env=env,
                                                  splines=splines,
                                                  min_niche_sigma=min_niche_sigma,
                                                  max_niche_sigma=max_niche_sigma,
                                                  max_poisson_lambda_val=max_poisson_lambda_val,
                                                  debug=debug,
                                                 )

    # create the full survey at each point
    print(f"\n\nDOING SURVEYS...\n\n")
    surveys = []
    ct = 0
    for i, j in survey_points:
        if verbose:
            if ct%25 == 0:
                print(f"\n\t{np.round(100*(ct/len(survey_points)), 1)}% complete...\n")
        survey = do_survey(env,
                           spp,
                           spp_max_poisson_lambdas,
                           i,
                           j,
                           use_multivar_normal=use_multivar_normal_niche,
                           prob_pres_thresh_round_to_1=prob_pres_thresh_round_to_1,
                           debug=debug,
                          )
        surveys.append(survey)
        ct+=1

    # create samples for each sampling scheme
    print(f"\n\nDRAWING SAMPLES...\n\n")
    samples = {}
    for scheme in sampling_schemes:
        if scheme == 'all':
            samples[scheme] = surveys
        else:
            samples = draw_sample(surveys, scheme)
            return

    # run GDM and return results
    print(f"\n\nRUNNING GDM...\n\n")
    gdm_splines, gdm_pca_rast = run_GDM(gamma,
                                        surveys,
                                        survey_points,
                                        env,
                                        pres_abund=gdm_pres_abund,
                                        site_survey_filename=site_survey_filename,
                                        env_rast_filename=env_rast_filename,
                                        spline_filename=spline_filename,
                                        pca_rast_filename=pca_rast_filename,
                                       )
    return spp, spp_max_poisson_lambdas, surveys, samples, gdm_splines, gdm_pca_rast



####################
# SET PARAMS AND RUN
####################
# behavioral params
VERBOSE = True
DEBUG = True
TIMEIT = True
PLOT_IT = True
SAVEPLOTS = True

USE_MULTIVAR_NORMAL_NICHE = False
PROB_PRES_THRESH_ROUND_TO_1 = 0.7
MAX_POISSON_LAMBDA_VAL = 1000
GDM_PRES_ABUND = True

SEED = 2
if SEED is not None:
    np.random.seed(SEED)

# landscape params
DIMS = (50, 50)
ENV_H = (0.5, 0.5, 0.5)
ADD_NOISE = True
#ENV = [nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=h) for h in ENV_H]
dist_source = np.zeros(DIMS)
dist_source[int(DIMS[0]/2-1):int(DIMS[0]/2+1),
            int(DIMS[1]/2-1):int(DIMS[1]/2+1)] = 1
ENV = [nlmpy.edgeGradient(nRow=DIMS[0], nCol=DIMS[1], direction=0),
       nlmpy.edgeGradient(nRow=DIMS[0], nCol=DIMS[1], direction=90),
       nlmpy.distanceGradient(dist_source),
      ]
if ADD_NOISE:
    NOISE = [nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=h) for h in ENV_H]
    ENV = [nlmpy.blendArrays([e, n]) for e, n in zip(ENV, NOISE)]


# splines
spline_vals = ([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.0, 0.05, 0.1, 0.7, 0.8, 0.85, 0.87, 0.9, 0.91, 0.99, 1.0]],
               [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.0, 0.04, 0.08, 0.19, 0.29, 0.39, 0.52, 0.66, 0.71, 0.89, 1.0]],
               [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.2, 0.24, 0.28, 0.30, 0.33, 0.36, 0.39, 0.40, 0.41, 0.42, 0.43]],
              )
KNOTS = np.linspace(0, 1, 8)
DEGREE = 3
SPLINES = [ISpline(x=v[0],
                   y=v[1],
                   i=i,
                   knots=KNOTS,
                   degree=DEGREE) for i, v in enumerate(spline_vals)]

# 'inventory' diversities (a la Whittaker)
# TODO: how to constrain min and max alpha values vis-a-vis gamma??
ALPHA = None
if ALPHA is None:
    ALPHA = nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=ENV_H[0])
else:
    assert isinstance(ALPHA, np.ndarray)
GAMMA=10000

# points to collect full surveys and samples at
N_POINTS = None
if N_POINTS is None:
    N_POINTS = np.prod(DIMS)
POINTS = draw_random_survey_points(DIMS, N_POINTS)

sim = Sim(env=ENV,
          splines=SPLINES,
          alpha=ALPHA,
          gamma=GAMMA,
          survey_points=POINTS,
          use_multivar_normal_niche=USE_MULTIVAR_NORMAL_NICHE,
          prob_pres_thresh_round_to_1=PROB_PRES_THRESH_ROUND_TO_1,
          max_poisson_lambda_val=MAX_POISSON_LAMBDA_VAL,
          gdm_pres_abund=GDM_PRES_ABUND,
          verbose=VERBOSE,
          debug=DEBUG,
          timeit=TIMEIT,
         )

# plot and save results
if PLOT_IT:
    sim.plot(scatter_survey_points=False,
             save=SAVEPLOTS,
            )
    # plot expected vs. observed distribution for random species
    sp = 0
    sim.plot_expec_vs_obser_distr(sp=sp,
                                  cmap='viridis',
                                  save=SAVEPLOTS,
                                 )

