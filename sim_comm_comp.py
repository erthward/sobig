import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Type
from copy import deepcopy
from nlmpy import nlmpy
from sklearn.neighbors import KernelDensity
from dms_variants.ispline import Isplines
import dms_variants
from scipy.stats import norm, multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
from math import comb
import pandas as pd
import rasterio as rio
import xarray as xr
import rioxarray as rxr

'''
simulate community composition at any or all points in a simulated landscape

input parameters include:
    - landscape dimensions
    - number of environmental layers
    - spatial autocorrelation of each of those layers
    - knots and coefficients for functions defining the relationship between
      environmental and ecological turnover (i.e., f(Env) functions)
    - an α-diversity layer (map of local species richness)
      (defaults to a layer parameterized same as the first environmental layer)
    - γ-diversity (total species count on landscape)



TODO:
    - pick up with TODO: DELETE comments
    - move other functions to methods of Sim
    - add functionality for removing species (based on niche width/specialization or something like that)
    - R gdm pkg:
        - FIGURE OUT HOW TO INSTALL!
        - either make it an optional dependency
        - or figure out way to prep whole thing to be conda isntalled
        - or port GDM to python?
    - add sampling function!
    - teaser scenarios:
        - show effects of sampling and sample site density on GAM richness model
        - b4/af env change scenario (CC, hab loss, combo)
        - nestedness vs turnover (fix all niche mus to landscape mean but let sigmas still vary)
    - allow both detection probabilities and lambda values (and ideally just
      whole user-built species, including niches) to be proapgated through or rest on Sim
    - GDM fits getting assigned to the sim leaves only space for one, which
      doesn't facilitate comparing results; reconfigure this (perhaps just
      create GDM class to return results in and give it a plot fn?)
    - prevent our f(Env) functions from being anchored at 0 on y-axis?
    - finalize input vs GDM-fitted f(Env) plotting issues
    - which is more justifiable, product of univariate normals or multivar normal?
    - is it a major problem that we're ignoring spatial autocorr in pres/abs determination? if so, add Gaussian random field?
    - add ability for knots and splines to be fed through as args to R's gdm()

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
raster_or_vectorlike = Union[List[numerical], Tuple[numerical], np.ndarray, xr.core.dataarray.DataArray]


#--------
# classes
#--------

class fEnv:
    '''
    class for `f(Env)` function that relates environmental and ecological
    distances as a monotonic function (i.e., a linear combination of I-spline
    basis functions with non-negative coefficients)

    Includes a method for getting the function's approximate slope at any value
    along the range of the environmental variable.
    '''
    def __init__(self,
                 id: int,
                 knots: vectorlike,
                 coeffs: vectorlike,
                 order: int = 3,
                 env_x_vals: Optional[vectorlike] = None,
                ) -> None:
        self.knots = np.array(knots)
        self.coeffs = np.array(coeffs)
        # validate and process args
        assert np.all((self.knots[1:] - self.knots[:-1]) >= 0), ("knots "
                                                       "must be ordered "
                                                       "from low to high.")
        assert len(self.coeffs) == len(self.knots)+1, ("length of coeffs must "
                                        "be 1 greater than number of knots.")
        assert self.coeffs[-1] == 0, ("to ensure slope of 0 at high end of "
                                "f(Env), the final coefficient must be 0.0.")
        # space environmental values evenly between 0 and 1000, if not provided
        if env_x_vals is None:
            env_x_vals = np.linspace(np.min(self.knots),
                                     np.max(self.knots), 1000)
        else:
            assert type(env_x_vals) in [list, tuple, np.ndarray]
            assert np.all((env_x_vals[1:] - env_x_vals[:-1]) >= 0), ("ispline_x"
                                                   "must be an ordered "
                                                   "range of x values "
                                                   "(i.e., environmental "
                                                   "values).")
        # save integer ID of this spline
        self.id = id
        # create the I-splines and the f(Env) that is their linear combination
        fenv, isplines = self._make_fEnv(knots=self.knots,
                                         coeffs=self.coeffs,
                                         ispline_order=order,
                                         ispline_x=env_x_vals,
                                        )
        # check length and monotonicity of resulting f(Env)
        assert len(fenv) == len(env_x_vals)
        assert np.all((fenv[1:] - fenv[:-1])>=0)
        # save x (i.e., env) and y (i.e., f(Env))
        self.x  = env_x_vals
        self.y = fenv
        # save min and max values
        self._x_min = np.min(self.x)
        self._x_max = np.max(self.x)
        self._y_min = np.min(self.y)
        self._y_max = np.max(self.y)
        self._x_minmax = (self._x_min, self._x_max)
        self._y_minmax = (self._y_min, self._y_max)
        # save iSplines object
        self.splines = isplines
        # placeholders for GDM-fitted values
        self.x_gdm_fit = None
        self.y_gdm_fit = None
        # total range of the environmental gradient
        self._x_range = np.max(self.x) - np.min(self.x)
        # get min and max slope values
        self._slope_min = np.min((self.y[1:]-self.y[:-1])/(self.x[1:]-self.x[:-1]))
        self._slope_max = np.max((self.y[1:]-self.y[:-1])/(self.x[1:]-self.x[:-1]))


    def _make_fEnv(self,
                   knots,
                   coeffs,
                   ispline_order: Optional[int] = 3,
                   ispline_x: Optional[vectorlike] = None,
                  ) -> np.ndarray:
        '''
        use the knots and coefficients provided to create and return a numpy
        array approximating f(Env) (a linear combination of a series of
        I-spline basis functions), as well as the dms_variants.isplines.Isplines
        object that provides the Ispline basis functions
        '''
        # if x is not provided, create it as 1000-point linearly spaced array
        # of values between the environmental values of the lowest and highest
        # knots
                # create the I-spline basis functions
        isplines = self._make_Ispline_basis(mesh=knots,
                                            x=ispline_x,
                                            order=ispline_order,
                                           )
        # create function as linear combination
        fenv = np.stack([coeffs[i-1] * isplines.I(i) for i in range(1,
                                        isplines.n+1)]).sum(axis=0)/isplines.n
        return fenv, isplines


    def _make_Ispline_basis(self,
                            mesh: vectorlike,
                            x: vectorlike,
                            order: Optional[int] = 3,
                           ) -> dms_variants.ispline.Isplines:
        '''
        create set of I-splines basis functions and return them (as a
        `dms_variants.ispline.Isplines object)
        '''
        isplines = Isplines(order=order,
                            mesh=mesh,
                            x=x,
                           )
        return isplines


    def _get_approx_slope(self,
                         x: numerical,
                        ) -> float:
        '''
        calculate simple local approximation of slope at value x
        (NOTE: for values of x that fall  below or above the
        minimum and maximum x values specified on the f(Env) function,
        the values returned are simply the slopes at the minimum and
        maximum x values)
        '''
        assert pd.notnull(x), "x must not be null"
        assert not np.isinf(x), "x must not be infinite"
        ind = np.argmin(np.abs(self.x - x))
        assert ind>=0 and ind<= len(self.x-1)
        ind_lo, ind_hi = np.clip([ind-1, ind+1], a_min=0, a_max=len(self.x)-1)
        Δx = self.x[ind_hi] - self.x[ind_lo]
        Δy = self.y[ind_hi] - self.y[ind_lo]
        return Δy/Δx


    def _add_GDM_fit(self,
                    x_gdm_fit: np.ndarray,
                    y_gdm_fit: np.ndarray,
                   ) -> None:
        '''
        add attributes to store a GDM-fitted f(Env) function
        '''
        self.x_gdm_fit = x_gdm_fit
        self.y_gdm_fit = y_gdm_fit
        self._y_gdm_min = np.min(self.y_gdm_fit)
        self._y_gdm_max = np.max(self.y_gdm_fit)


    def plot(self,
             ax: Optional[mpl.axes._axes.Axes] = None,
             include_input: bool = True,
             legend: bool = False,
            ) -> None:
        '''
        make simple plot of f(Env) and its GDM fit
        '''
        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("$Env_%i$" % self.id, size=14)
        if self.x_gdm_fit is not None and self.y_gdm_fit is not None:
            new_scale = (self._y_gdm_min, self._y_gdm_max)
            y_plot = _rescale_arr(arr=self.y, new_scale=new_scale)
        else:
            y_plot = self.y[:]
        if include_input:
            ax.plot(self.x,
                    y_plot,
                    ':',
                    label='input',
                    linewidth=2,
                    color='blue'
                   )
        if self.x_gdm_fit is not None and self.y_gdm_fit is not None:
            ax.plot(self.x_gdm_fit,
                    self.y_gdm_fit,
                    '-',
                    label='GDM fit',
                    linewidth=2,
                    color='black',
                    alpha=0.5,
                   )
        ax.set_xlabel("$Env_%i$" % self.id)
        ax.set_ylabel("$f(Env_%i)$" % self.id)
        if legend:
            ax.legend()
        ax.grid(True)


class Species:
    '''
    class for a simulated species
    '''
    def __init__(self,
                 niche: list[vectorlike],
                 max_poisson_lambda: float,
                 prob_detect: float,
                ) -> None:
        # validate args
        for i in range(len(niche)):
            assert len(niche[i]) == 2 # mu and sigma
        assert prob_detect is None or 0 <= prob_detect <= 1
        # assign attributes
        self.niche = niche
        self.max_poisson_lambda = max_poisson_lambda
        self.prob_detect = prob_detect


class Sim:
    '''
    overall class for the simulation
    '''
    def __init__(self,
                 env: List[rasterlike],
                 fenvs: list[Type[fEnv]],
                 gamma: int,
                 n_survey_sites: Optional[int] = None,
                 survey_sites: Optional[List[Tuple[float]]] = None,
                 min_niche_sigma: float = 0.001,
                 max_niche_sigma: float = 0.1,
                 use_multivar_normal_niche: bool = False,
                 prob_pres_thresh_round_to_1: Optional[float] = None,
                 max_poisson_lambdas: Optional[vectorlike] = None,
                 max_poisson_lambda_across_spp: int = 1000,
                 detect_probs: Optional[vectorlike] = None,
                 verbose: bool = False,
                 debug: bool = False,
                 timeit: bool = True,
                ) -> None:
        # validate args
        # store behavioral params
        self._verbose = verbose
        self._debug = debug
        self._timeit = timeit
        self._use_multivar_normal_niche = use_multivar_normal_niche
        self._prob_pres_thresh_round_to_1 = prob_pres_thresh_round_to_1
        # store hidden utility attributes
        self._env_cmaps = ['Reds', 'Greens', 'Blues']
        self._n_lyrs = len(env)
        self._dims = env[0].shape
        # store fixed params
        self.min_niche_sigma = min_niche_sigma
        self.max_niche_sigma = max_niche_sigma
        self.max_poisson_lambda_across_spp = max_poisson_lambda_across_spp
        assert len(fenvs) == self._n_lyrs
        self.fenvs = fenvs
        self._fenvs_slope_min = np.min([s._slope_min for s in self.fenvs])
        self._fenvs_slope_max = np.max([s._slope_max for s in self.fenvs])
        self.gamma = gamma
        # set the environment
        self.update_env(env,
                        verbose=self._verbose,
                        debug=self._debug,
                       )
        # make niche KDE
        self._make_niche_kde()
        # handle survey_sites
        if survey_sites is None:
            if n_survey_sites is None:
                # create a point for every raster cell, if n_survey_sites not
                # specified
                n_survey_sites = np.prod(self._dims)
            survey_sites = _draw_random_survey_sites(self._dims, n_survey_sites)
        else:
            assert n_survey_sites is None, ("If survey_sites are provided "
                                            "then n_survey_sites must be None.")
        self.sites = survey_sites
        self.n_sites = len(self.sites)
        # create all species
        self._make_all_species_runtime = None
        if self._timeit:
            start = time.time()
        if self._verbose:
            print(f"\n\nCREATING SPECIES...\n\n")
        self._make_all_species(max_poisson_lambdas=max_poisson_lambdas,
                               detect_probs=detect_probs,
                              )
        if self._timeit:
            stop = time.time()
            runtime_sec = stop-start
            self._make_all_species_runtime = runtime_sec
            if self._verbose:
                print(("\n\nALL SPECIES CREATED IN "
                       f"{np.round(self._make_all_species_runtime/60, 2)} "
                       "MINUTES.\n\n"))
        # simulate the communities
        self._runtime_sim_comms = None
        self._sim_comms(verbose=self._verbose,
                        timeit=self._timeit,
                        debug=self._debug,
                       )
        # save empty attributes that will be refilled/replaced as the Sim
        # object is used
        self.surveys = None
        self.gdm_fits = None
        self.gdm_pca_rast = None


    def update_env(self,
                   env: List[rasterlike],
                   verbose: Optional[bool] = None,
                   timeit: Optional[bool] = None,
                   debug: Optional[bool] = None,
                   recalc_std: bool = False,
                  ) -> None:
        '''
        update the environment of a Sim object (e.g., to model the effects of
        environmental change)
        NOTE: defaults to not updating the standard deviation of the layers
              (which are used to rescale species niche widths)
        '''
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        assert len(env) == self._n_lyrs
        assert np.all(env[0].shape == self._dims)
        # check if the environment is different
        env_changed = hasattr(self, 'env') and (not np.all(self.env == env))
        # convert environment to np.ndarray and store it
        self.env = np.array(env)
        # add a depth-1 0th dimension, if env is just a single raster
        if self._n_lyrs == 1:
            self.env = self.env.reshape(-1, *self.env.shape)
        self._env_min_vals = [np.min(e) for e in self.env]
        self._env_max_vals = [np.max(e) for e in self.env]
        # set the environmetn's standard deviation, if doesn't yet exist and/or
        # if recalc_std is True
        if not hasattr(self, '_env_stds') or recalc_std:
            self._env_stds = [np.std(e) for e in self.env]
        # redraw communities, if the environment has changed
        if env_changed:
            del self.comms
            self._sim_comms(verbose=verbose, timeit=timeit, debug=debug)


    def _make_niche_kde(self,
                        kernel='gaussian',
                        bandwidth='scott',
                       ):
        # got rid of failed alpha idea; just sampling whole environment evenly
        draw_cts = np.ones(self.env[0].shape)
        assert np.all(draw_cts % 1 == 0)
        draw_lists = []
        for e in self.env:
            draws = []
            for i, ct in enumerate(draw_cts.ravel()):
                for n in range(int(ct)):
                    draws.append(e.ravel()[i])
            draw_lists.append(draws)
        kde = KernelDensity(kernel=kernel,
                            bandwidth=bandwidth).fit(np.array(np.array(draw_lists).T))
        self._niche_kde = kde


    def _make_all_species(self,
                          max_poisson_lambdas: Optional[vectorlike] = None,
                          detect_probs: Optional[vectorlike] = None,
                         ) -> None:
        '''
        create a dict of all species' ecological niches
        (i.e., μ and σ values for all environmental layers)
        '''
        # draw species' niche centers from the niche KDE
        spp_mus = self._niche_kde.sample(self.gamma)

        # draw species' lambdas for Poisson distributions determining survey
        # results (will be multiplied by probability of presence at a location, so
        # this is the maximum value that a Poisson draw will take in a location
        # where probability of presence goes to 1.0)
        if max_poisson_lambdas is None:
            max_poisson_lambdas = np.random.uniform(1,
                                                    self.max_poisson_lambda_across_spp,
                                                    self.gamma,
                                                   )
        else:
            assert np.all(max_poisson_lambdas > 0)
        # draw detection probabilities randomly, if not provided
        if detect_probs is None:
            detect_probs = np.random.uniform(low=0, high=1, size=self.gamma)
        else:
            assert type(detect_probs) in [list, tuple, np.ndarray]
            assert len(detect_probs) == gamma
            assert np.all(detect_probs >= 0)
            assert np.all(detect_probs <= 1)
        # use inverse of f(Env) slope at each μ to draw each σ
        # (following a rationale derived independently but that aligns with
        # Bush et al. 2019)
        spp = {}
        for s, mus in zip(range(self.gamma), spp_mus):
            niche = []
            for i, mu in enumerate(mus):
                slope = self.fenvs[i]._get_approx_slope(mu)
                # use min-max scaling to determine slope proportional position
                # between min and max slope values, then remap to interval between
                # user-specified max and min niche widths
                slope_prop = ((slope - self._fenvs_slope_min)/
                              (self._fenvs_slope_max - self._fenvs_slope_min))
                if self.max_niche_sigma is None:
                    max_sigma_i = self.fenvs[i]._x_range/2
                else:
                    max_sigma_i = self.max_niche_sigma
                # NOTE: max niche width defaults to half the range of this
                #       environmental variable if not user-specified
                sigma = max_sigma_i - (
                        slope_prop*(max_sigma_i - self.min_niche_sigma))
                sigma = np.clip(sigma,
                                a_min=self.min_niche_sigma,
                                a_max=max_sigma_i,
                               )
                # now rescale sigma (currently expressed in standard deviations)
                # to the native distribution of the environmental layer (by
                # multiplying by its standard deviation)
                sigma_scaled = sigma * self._env_stds[i]
                niche.append((mu, sigma_scaled))
                # now multiply that by the standard deviation of the
                # environmental layer
            # create and save the Species
            sp = Species(niche=niche,
                         # TODO: DELETE
                         #niche_cent=niche_cents[s],
                         max_poisson_lambda=max_poisson_lambdas[s],
                         prob_detect=detect_probs[s],
                        )
            spp[s] = sp
        self.spp = spp


    def _sim_comms(self,
                   verbose: Optional[bool] = None,
                   timeit: Optional[bool] = None,
                   debug: Optional[bool] = None,
                  ) -> None:
        '''
        Produces a dict of observation lists, keyed to sampling schemes, if more
        than one sampling scheme provided. Otherwise, returns an observation
        list for the single sampling scheme.
        '''
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        if timeit:
            start = time.time()
        if verbose:
            print(f"\n\nSIMULATING COMMUNITIES AT SURVEY POINTS...\n\n")
        # create the simulated communities at each point
        if not hasattr(self, 'comms'):
            comms = []
            ct = 0
            for i, j in self.sites:
                if verbose:
                    if ct%25 == 0:
                        print(f"\n\t{np.round(100*(ct/len(self.sites)), 1)}% complete...\n")
                survey = _sim_comm(self.env,
                                   self.spp,
                                   i,
                                   j,
                                   use_multivar_normal=self._use_multivar_normal_niche,
                                   prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                   debug=debug,
                                  )
                comms.append(survey)
                ct+=1
                # store the full communities
                self.comms = comms
                # NOTE: flip the _env_changed flag to False (it will stay
                # that way unless and until the env is updated again)
                self._env_changed = False
        else:
            pass
        # store and report runtime, as needed
        if timeit:
            stop = time.time()
            runtime_sec = stop-start
            self._runtime_sim_comms = runtime_sec
            if verbose:
                print(("\n\nALL COMMUNITIES SIMULATED IN "
                       f"{np.round(self._runtime_sim_comms/60, 2)} "
                       "MINUTES.\n\n"))


    def sim_obs(self,
                scheme: str = 'perfect',
                efforts: Optional[vectorlike] = None,
                by_rel_abund: bool = False,
                by_detect_prob: bool = False,
                verbose: Optional[bool] = None,
                timeit: Optional[bool] = None,
                debug: Optional[bool] = None,
               ) -> List[Dict[int, int]]:
        '''
        Simulate observations at all survey_sites using the given scheme
        (defaults to 'perfect', which simply returns the complete
        simulated communities at each site), site-specific measures of effort
        (defaults to None, which returns a single 'opportunistic' sighting),
        and whether sampling probabilities should be determined as a function of
        relative abundances and/or species' intrinsic detection probabilities
        (both default to None, which yields uniform sampling probabilities
        across all individuals)

        Returns a list of observation dicts, one per survey sites, with each
        dict containing key:value pairs of species_id:count
        '''
        if efforts is not None:
            assert type(efforts) in [list, tuple, np.ndarray]
            assert len(efforts) == len(self.sites)
            assert np.all(efforts >= 0)
            assert np.all(efforts <= 1)
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        if timeit:
            start = time.time()
        if verbose:
            if scheme == 'perfect':
                label = 'PERFECT '
            elif scheme == 'sample':
                label = ''
            print(f"\n\nSIMULATING {label}SAMPLING "
                  "AT SURVEY POINTS...\n\n")
        # handle sampling-scheme arguments
        assert scheme in ['perfect',
                          'sample',
                         ]
        # return communities, if scheme is 'perfect'...
        if scheme == 'perfect':
            return self.comms
        # ...otherwise, return list of simulated observations at each site
        elif scheme == 'sample':
            obs = []
            tot = len(self.comms)
            for i, comm in enumerate(self.comms):
                if verbose and i % 100 == 0:
                    print(f"\t{np.round((i+1)/tot*100, 1)}% complete...")
                if efforts is not None:
                    effort = efforts[i]
                else:
                    effort = None
                ob = _sim_sample(spp=self.spp,
                                 comm=comm,
                                 effort=effort,
                                 by_rel_abund=by_rel_abund,
                                 by_detect_prob=by_detect_prob,
                                )
                obs.append(ob)
        return obs


    def run_GDM(self,
                surveys: Optional[List[Dict[int, int]]] = None,
                gdm_data_type: str = 'abund',
                site_survey_filename: str = 'site_survey.csv',
                env_rast_filename: str = 'env_rast.tif',
                fits_filename: str = 'GDM_fits.csv',
                pca_rast_filename: str = 'GDM_env_rast_PCA.tif',
                plot_it: bool = False,
                plot_fenv_input: bool = True,
                plot_title: str = '',
                verbose: bool = False,
               ) -> None:
        '''
        prep GDM data, save to disk, execute R script to run GDM,
        then read in and return results
        '''
        assert isinstance(gdm_data_type, str)
        assert gdm_data_type in ['abund', 'pres_abs']
        print(f"\n\nRUNNING GDM...\n\n")
        # use the complete communities, if surveys were not provided
        if surveys is None:
            surveys = self.comms
        # prep and save GDM input data
        _prep_GDM_input_data(gamma=self.gamma,
                             surveys=surveys,
                             survey_sites=self.sites,
                             env=self.env,
                             bio_data_type=gdm_data_type,
                             site_survey_filename=site_survey_filename,
                             env_rast_filename=env_rast_filename,
                            )
        # run R script
        if gdm_data_type == 'abund':
            abund = 'TRUE'
        else:
            abund = 'FALSE'
        R_cmd = (f"Rscript --vanilla run_gdm.r {site_survey_filename} "
                 f"{env_rast_filename} {abund} "
                 f"{fits_filename} {pca_rast_filename}")
        if verbose:
            print(f"\tNOW RUNNING: > {R_cmd}\n")
        os.system(R_cmd)
        # read and return results
        gdm_fits = pd.read_csv(fits_filename)
        pca_rast = rxr.open_rasterio(pca_rast_filename)
        # min-max scale raster (comes in as 0-255)
        pca_rast_rescaled = _rescale_arr(pca_rast, by_rast_band=True)
        # save the output GDM fits and raster to their Sim attributes
        self.gdm_fits = gdm_fits
        for i, fenv in enumerate(self.fenvs, start=1):
            fenv._add_GDM_fit(x_gdm_fit=self.gdm_fits[f"x.env_rast_{i}"],
                              y_gdm_fit=self.gdm_fits[f"y.env_rast_{i}"],
                             )
        # extend the first axis of the GDM PC raster to length 3, if necessary,
        # by providing layers of all 0s
        n_lyrs_add = 3 - pca_rast_rescaled.shape[0]
        if n_lyrs_add > 0:
            pca_rast_rescaled = xr.concat([pca_rast_rescaled,
                    pca_rast_rescaled[:n_lyrs_add, :, :]*0], dim='band')
            # NOTE: update the 'long_name' field
            pca_rast_rescaled = pca_rast_rescaled.assign_attrs({'long_name':
                                                        ['PC1', 'PC2', 'PC3']})
        self.gdm_pca_rast = pca_rast_rescaled
        if plot_it:
            sim.plot(scatter_survey_sites=False,
                     plot_fenv_input=plot_fenv_input,
                     title=plot_title,
                     save=False,
                    )
        return gdm_fits, pca_rast_rescaled


    def plot(self,
             scatter_survey_sites: bool = True,
             plot_fenv_input: bool = True,
             title: str = '',
             save: bool = False,
            ) -> None:
        '''
        plot the results of a simulation
        '''
        fig = plt.figure(figsize=(16,16))
        fig.suptitle(title)
        gs = fig.add_gridspec(80, 100)

        # plot environment rasters
        axwidth = int(100/self.env.shape[0])-1
        for i, e in enumerate(self.env):
            ax = fig.add_subplot(gs[:20,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
            img = ax.imshow(e,
                            vmin=self._env_min_vals[i],
                            vmax=self._env_max_vals[i],
                            cmap=self._env_cmaps[i],
                           )
            plt.colorbar(img)
            # add survey sites
            if scatter_survey_sites:
                for point in self.sites:
                    ax.scatter(point[0],
                               point[1],
                               color='white',
                               edgecolor='black',
                               alpha=0.8,
                               s=24,
                              )
            ax.set_title("$Env_%s$" % i, size=14)

        # plot their fEnvs
        fenv_axs = []
        for i, fenv in enumerate(self.fenvs):
                        ax = fig.add_subplot(gs[25:40,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
                        fenv.plot(ax=ax,
                                  legend=i==(self.env.shape[0]-1),
                                  include_input=plot_fenv_input,
                                 )
                        fenv_axs.append(ax)
        fenv_ax_max_ylim = np.max([np.max(ax.get_ylim()) for ax in fenv_axs])
        for ax in fenv_axs:
            ax.set_ylim(0, fenv_ax_max_ylim)

        # plot raster of observed alpha values at all surveyed cells
        ax = fig.add_subplot(gs[50:, 35:65])
        survey_len_arr = np.ones(self.env[0, :, :].shape)*np.nan
        for pt, survey in zip(self.sites, self.comms):
            survey_len_arr[int(pt[0]), int(pt[1])] = len(survey)
        survey_lengths = [len(survey) for survey in self.comms]
        img = ax.imshow(survey_len_arr,
                        vmin=min(survey_lengths),
                        vmax=max(survey_lengths),
                       )
        plt.colorbar(img)
        ax.set_title('α-diversity at surveyed sites', size=14)

        # plot PCA rast from GDM transform
        ax = fig.add_subplot(gs[50:, 70:])
        self.gdm_pca_rast.plot.imshow(ax=ax)
        # add survey sites
        if scatter_survey_sites:
            for point in self.sites:
                ax.scatter(point[0],
                           point[1],
                           color='white',
                           edgecolor='black',
                           alpha=0.8,
                           s=24,
                          )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')
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
                                  title: Optional[str] = None,
                                  cmap: str = 'viridis',
                                  save: bool = False,
                                 ) -> None:
        '''
        plot both the expected and observed distribution of the given species
        '''
        # get species' niche
        niche = self.spp[sp].niche
        # calculate map of expected distribution
        expec = np.zeros(self.env[0, :, :].shape)
        # calculate presence probability at all cells
        for i in range(self.env[0, :, :].shape[0]):
            for j in range(self.env[0, :, :].shape[1]):
                prob = _calc_pres_prob(env_vals=[e[i, j] for e in self.env],
                                       niche=self.spp[sp].niche,
                                       use_multivar_normal=self._use_multivar_normal_niche,
                                       prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                      )
                expec[i, j] = prob
        # calculate map of all pixels where species is observed
        # (setting pixels without communities to NaNs)
        obser = np.zeros(self.env[0, :, :].shape)
        for pt, survey in zip(self.sites, self.comms):
            if sp in survey:
                obser[int(pt[0]), int(pt[1])] = survey[sp]
        for i in range(self.env[0, :, :,].shape[0]):
            for j in range(self.env[0, :, :].shape[1]):
                if (i+0.5, j+0.5) not in self.sites:
                    obser[i, j] = np.nan
        # plot both
        show_fig = False
        fig = plt.figure(figsize=(14,8))
        if title is None:
            title = f"sp. {sp}"
            if self.spp[sp].prob_detect is not None:
                title = title + " ($P(detect) = %0.2f$)" % self.spp[sp].prob_detect
        fig.suptitle(title)
        gs = fig.add_gridspec(nrows=80, ncols=140)
        axs_env = [fig.add_subplot(gs[:25,
                (i*20)+(i*5):(i+1)*20+(i*5)]) for i in range(self.env.shape[0])]
        ax_expec = fig.add_subplot(gs[25:, :55])
        ax_obser = fig.add_subplot(gs[25:, 85:])
        for i, e in enumerate(self.env):
            ax = axs_env[i]
            img = ax.imshow(e,
                            vmin=self._env_min_vals[i],
                            vmax=self._env_max_vals[i],
                            cmap=self._env_cmaps[i],
                           )
            ax.set_xticks(())
            ax.set_xticks(())
            plt.colorbar(img)
            ax.set_title("$Env_%s$" % i, size=14)
        img = ax_expec.imshow(expec,
                              cmap=cmap,
                              vmin=0,
                              vmax=1,
                             )
        plt.colorbar(img)
        ax_expec.set_title('expected distribution')
        ax_obser.imshow(obser,
                         cmap=cmap,
                         vmin=0,
                         vmax=1,
                        )
        ax_obser.set_title('observed distribution')
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

def _standardize_vec(vec: vectorlike):
    '''
    standardize a numerical vector-like object
    '''
    return (np.array(vec) - np.nanmean(vec))/np.nanstd(vec)


def _rescale_arr(arr: Union[vectorlike, rasterlike],
                 new_scale: Optional[vectorlike] = [0, 1],
                 by_rast_band : bool = False,
                ) -> Union[np.ndarray, xr.core.dataarray.DataArray]:
    '''
    linearly recast an array to a new interval (default: [0,1])
    using min-max scaling, optionally by raster band (i.e., by index on axis 0;
    defaults to rescaling the entire raster's set of values, regardless of axes)
    '''
    assert len(new_scale) == 2
    new_range = new_scale[1] - new_scale[0]
    assert new_range > 0
    out = deepcopy(arr)
    if by_rast_band:
        assert len(out.shape) == 3
        for i in range(out.shape[0]):
            out[i] = (((out[i]-np.nanmin(out[i])) * new_range)/
                       (np.nanmax(out[i])-np.nanmin(out[i]))) + new_scale[0]
    else:
        out = (((out-np.nanmin(out)) * new_range)/
               (np.nanmax(out)-np.nanmin(out))) + new_scale[0]
    return out


def _standardize_arr(arr: Union[vectorlike, rasterlike],
                     by_rast_band : bool = False,
                    ) -> Union[np.ndarray, xr.core.dataarray.DataArray]:
    '''
    recast an array to the standard normal distribution (i.e., ~N(0, 1)),
    optionally by raster band (i.e., along axis 0),
    '''
    out = deepcopy(arr)
    if by_rast_band:
        assert len(out.shape) == 3
        for i in range(out.shape[0]):
            out[i] = (out[i] - np.nanmean(out[i]))/np.nanstd(out[i])
    else:
        out = (out - np.nanmean(out))/np.nanstd(out)
    return out


def _draw_random_survey_sites(dims: Tuple[Union[float, int]],
                             n: int) -> List[Tuple[float]]:
    '''
    draw a set of random survey site points within a raster whose coordinates
    range from 0 to dim-1 in both axes in dims; each point will be in a
    separate pixel, so n must not exceed the number of pixels
    '''
    assert n > 0 and n <= np.prod(dims)
    X, Y = np.meshgrid(range(dims[0]), range(dims[1]))
    xs = X.ravel()
    ys = Y.ravel()
    pts = [*zip(xs, ys)]
    np.random.shuffle(pts)
    idxs = np.random.choice(range(len(pts)), replace=False, size=n)
    # NOTE: add 0.5 to all site points, to place them in cell centers
    rand_pts = [tuple(np.array(pts[idx])+0.5) for idx in idxs]
    return rand_pts



def _calc_pres_prob(env_vals: vectorlike,
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


def _sim_comm(env: rasterlike,
              spp: Dict[int, Species],
              i: float,
              j: float,
              max_prob_pres: float = 1.0,
              prob_pres_thresh_round_to_1: Optional[float] = None,
              use_multivar_normal: bool = False,
              debug: bool = False,
             ) -> Dict[int, int]:
    '''
    use the list of environmental layers provided and the dict of species
    and their niches to simulate complete community composition at grid cell i,j
    '''
    # list to store all species present
    survey = {}
    # get environmental values at point grid cell i,j
    # NOTE: site points sit at cell centers, so int() converts to their cell indices
    env_vals = [e[int(i), int(j)] for e in env]
    for s, sp in spp.items():
        # calculate presence probability
        prob = _calc_pres_prob(env_vals,
                               sp.niche,
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
            survey[s] = np.random.poisson(prob * sp.max_poisson_lambda)
    return survey


def _calc_rarefaction_curve(N: int,
                       comm: Dict[int, int],
                       effort: float,
                      ):
    '''
    calcuates n (sample size) and k (species richness of sample) as functions of
    N (total community count), K (total species richness), N_i (count of each
    species), and sampling effort (a measure constrained to the [0, 1] interval)

        '''
    assert N > 0
    assert 0 <= effort <= 1
    K = len(comm)
    k = K - np.sum([N - comb(Ni, n) for Ni in comm.values()])/(comb(N, n))
    return n, k


def _sim_sample(spp: Dict[int, Species],
                comm: Dict[int, int],
                effort: Optional[float] = None,
                by_rel_abund: bool = True,
                by_detect_prob: bool = False,
               ) -> Dict[int, int]:
    '''
    simulate a sample of species observations from the community provided
    using sampling arguments including effort (default to None, in which case
    only a single 'opportunistic' sample is returned; otherwise constrained
    to [0, 1]), and whether or not relative abundances and/or intrinsic
    detection probabilities should influence species' observation probabilities
    '''
    # get total number of individuals in the whole community
    N = np.sum([*comm.values()])
    # copy the comm, for use as a counter object
    counter = deepcopy(comm)
    # create output object
    sample = {}
    # get vector of probs that a single sighting happens to be of each species
    # (starts as all ones, then gets multiplied by needed values)
    sp_probs = np.ones(len(comm))
    # multiply by abundances, if relative abundance factors into sampling probs
    if by_rel_abund:
        sp_probs *= np.array([*comm.values()])
    # mutliply by species intrinsic detection probabilities, if needed
    if by_detect_prob:
        sp_probs *= np.array([spp[sp].prob_detect for sp in comm.keys()])
    # now normalize to proper probabilities, for use in np.random.choice
    sp_probs = sp_probs/np.sum(sp_probs)
    assert np.allclose(np.sum(sp_probs), 1)
    # use effort and rarefaction to determine size of sample...
    if effort is not None:
        # NOTE: FOR NOW, ASSUMES SIMPLE LINEAR SCALING OF SAMPLE SIZE WITH EFFORT
        n = int(np.round(N*effort, 0))
    # ... or set it to 1, if effort is not provided and this is thus an
    # 'opportunistic' sample
    else:
        n = 1
    # loop over sample size, draw samp, and pop it from counter into sample
    while np.sum([*sample.values()]) < n:
        sp = np.random.choice([*comm.keys()], p=sp_probs)
        if sp in counter:
            counter[sp] -= 1
            if counter[sp] == 0:
                del counter[sp]
            if sp in sample:
                sample[sp] += 1
            else:
                sample[sp] = 1
        else:
            pass
    # check all counts are <= full count in comm
    for sp in sample:
        assert sample[sp] <= comm[sp]
    # check total sample size is correct
    assert np.sum([*sample.values()]) == n
    if effort is None:
        assert np.sum([*sample.values()]) == 1
    return sample



def _prep_GDM_input_data(gamma: int,
                         surveys: List[Dict[int, int]],
                         survey_sites: List[Tuple[float]],
                         env: rasterlike,
                         bio_data_type: str = 'abund',
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
        # NOTE: adding x and y columns (sites are expressed as (i, j) matrix
        #       indices, so flip them express as (x, y) geographic coordinates)
        pt = survey_sites[i]
        site_surv_mat[i, 1] = pt[1]
        site_surv_mat[i, 2] = pt[0]
        for j, abund in survey.items():
            if bio_data_type == 'abund':
                site_surv_mat[i, j+add_cols] = abund
            elif bio_data_type == 'pres_abs':
                site_surv_mat[i, j+add_cols] = 1
    site_surv_df = pd.DataFrame(site_surv_mat)
    site_surv_df.columns = ['site', 'x', 'y'] + [f'spp{i}' for i in range(gamma)]
    site_surv_df.to_csv(site_survey_filename, index=False)
    # create and save environmental raster
    ydim, xdim = env.shape[1], env.shape[2]
    n_bands = env.shape[0]
    dtype = env.dtype
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
            dst.write(env[n], n + 1)
    print("\nGDM INPUTS SAVED TO DISK.\n")




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
MIN_NICHE_SIGMA = 0.01
MAX_NICHE_SIGMA = 1.5
PROB_PRES_THRESH_ROUND_TO_1 = None
MAX_POISSON_LAMBDA_ACROSS_SPP = 1000
GDM_DATA_TYPE = 'abund'

SEED = 2
if SEED is not None:
    np.random.seed(SEED)

# knots and coeffs for f(Env)
knots = ([-1.5, -1, 0, 1, 1.5],
         [-1.3, -0.2, 0.2, 1.1, 1.3],
         [0, 10, 90, 100],
        )
coeffs = ([1, 1.5, 2, 2.5, 3, 0],
          [0.1, 0.2, 0, 0.2, 6, 0],
          [0.1, 0.1, 0.1, 0.1, 0],
         )
FENV = [fEnv(id=i, knots=k, coeffs=c) for i, (k, c) in enumerate(zip(knots,
                                                                     coeffs))]

# landscape params
DIMS = (50, 50)
ENV_H = (0.5, 0.5, 0.5)
ADD_NOISE = True
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
# rescale to a normal centered on 0
ENV = [_rescale_arr(e, new_scale=fenv._x_minmax) for fenv, e in zip(FENV, ENV)]


# params to determine 'inventory' diversities (a la Whittaker)
GAMMA=2000

# species-species lambdas (for ~Pois distributions determining abundance)
MAX_POISSON_LAMBDAS = None

# detection probability vector (or None, to have randomly assigned)
DETECT_PROBS = None

# create the simulator
sim = Sim(env=ENV,
          fenvs=FENV,
          gamma=GAMMA,
          n_survey_sites=None,
          survey_sites=None,
          min_niche_sigma=MIN_NICHE_SIGMA,
          max_niche_sigma=MAX_NICHE_SIGMA,
          use_multivar_normal_niche=USE_MULTIVAR_NORMAL_NICHE,
          prob_pres_thresh_round_to_1=PROB_PRES_THRESH_ROUND_TO_1,
          max_poisson_lambda_across_spp=MAX_POISSON_LAMBDA_ACROSS_SPP,
          max_poisson_lambdas=MAX_POISSON_LAMBDAS,
          detect_probs=DETECT_PROBS,
          verbose=VERBOSE,
          debug=DEBUG,
          timeit=TIMEIT,
         )
assert False
# run GDM on full communities
sim.run_GDM(surveys=None)

# plot and save results
sim.plot(scatter_survey_sites=False,
         plot_fenv_input=False,
         title='before change',
         save=True,
        )
# plot expected vs. observed distribution for random species
sp = [9, 10, 16]
for s in sp:
    sim.plot_expec_vs_obser_distr(sp=s,
                                  title=f"before change: sp {s}",
                                  cmap='viridis',
                                  save=True,
                                 )

# deepcopy sim (just in case)
sim_b4 = deepcopy(sim)

# update the environment to simulate environmental change, then rerun the
# same set of GDM results
increase = nlmpy.mpd(50, 50, 1)*0.8
ENV[1] = ENV[1] + increase
sim.update_env(ENV)
sim.run_GDM(surveys=None)
# plot again
sim.plot(scatter_survey_sites=False,
         plot_fenv_input=False,
         title='after change',
         save=True,
        )

# plot expected vs. observed distribution for random species
sp = [9, 10, 16]
for s in sp:
    sim.plot_expec_vs_obser_distr(sp=s,
                                  title=f"before change: sp {s}",
                                  cmap='viridis',
                                  save=True,
                                 )





## compare GDM results for:
#             # 1. complete community data
#scenarios = [{'scheme': 'perfect'},
#             # 2. abundance- and detectability-weighted surveys of 0.5 effort
#            {'scheme': 'sample',
#             'efforts': np.ones((sim.n_sites))*0.02,
#             'by_rel_abund': True,
#             'by_detect_prob': True,
#            },
#             # 3. abundance- and detectability-weighted surveys of variable effort
#           #  {'scheme': 'sample',
#           #   'efforts': np.random.uniform(0, 1, sim.n_sites),
#           #   'by_rel_abund': True,
#           #   'by_detect_prob': True,
#           #  },
#            ]
#for scenario, plot_title in zip(scenarios, ['complete',
#                                            'even effort',
#                                            'variable effort',
#                                           ]):
#    obs = sim.sim_obs(**scenario)
#    sim.run_GDM(surveys=obs,
#                plot_it=True,
#                plot_fenv_input=False,
#                plot_title=plot_title,
#               )
#
#




