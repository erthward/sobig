import numpy as np
import pandas as pd
from nlmpy import nlmpy
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt

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

# behavioral params
PLOT_IT = True

# landscape params
DIMS = (20, 20)
N_ENV = 2
ENV_H = (1, 1)
ENV = [nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=h) for h in ENV_H]

# splines
env_spline_vals = ([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.0, 0.0, 0.0, 0.05, 0.1, 0.4, 0.75, 0.85, 0.9, 0.95, 1.0]],
                   [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.0, 0.2, 0.8, 0.9, 0.93, 0.95, 0.965, 0.975, 0.985, 0.995, 1.0]],
                  )
# TODO: look into better spline-fitting procedure that matches GDM approach,
#       anchors splines to the interval [0, 1], and so forth
# TODO: then implement asserts checking that min and max y values are 0 and 1,
#       derivatives are always >= 0, ... anything else?
LAM = 0.0002
SPLINES = [make_smoothing_spline(v[0], v[1], lam=LAM) for v in env_spline_vals]
if PLOT_IT:
    fig_splines = plt.figure()
    ax = fig_splines.add_subplot(1, 1, 1)
    x_plot = np.arange(0, 1.01, 0.01)
    for i, spline in enumerate(SPLINES):
        ax.plot(x_plot, spline(x_plot), label=f"env {i+1}")
    lgd = ax.legend()
    plt.show()

# 'inventory' diversities (a la Whittaker)
# TODO: how to constrain min and max alpha values vis-a-vis gamma??
ALPHA = None
if ALPHA is None:
    ALPHA = nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=ENV_H[0])
else:
    assert isinstance(ALPHA, np.ndarray)
GAMMA=1000

def create_species(gamma,
                   env_splines,
                  ):
    '''
    create a dict recording all species' μ and σ values for all environmental
    layers
    '''
    spp = dict()
    for sp in range(gamma):
        niche = dict()
        for e, spline in enumerate(env_splines):
            # TODO: need to ensure coverage across environmental space! how
            #       best to do this? Maybe we use the alpha layer to
            #       get the conditional probabilities of a species having its
            #       niche midpoint within every hyper-subcube of the env
            #       hyperspace, then make binomial draws using those
            #       probabilities?
            μ = np.random.uniform()
            # TODO: get rid of np.abs once we assert that derivative always >=0
            slope = np.abs(spline.derivative()(μ))
            σ = ...
            niche[e] = (μ, σ)
        spp[sp] = niche
    return spp


def main(env=ENV,
         splines=SPLINES,
         alpha=ALPHA,
         gamma=GAMMA,
         survey_points=...
         sample_schemes=[...]
        ):
    # create all species

    # create the full survey at each point

    # create samples for each sampling scheme

    # plot simple viz of results

    # save all

