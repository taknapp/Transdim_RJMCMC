#!/usr/bin/env python3

import sys
sys.path.append("modules")
sys.path.append("modules/TransdimensionalSplineFitting")
from loguru import logger
import numpy as np
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from spline_redshift import createSplineRedshift
import pickle
import cloudpickle
from pathlib import Path

import json


from popstock_tsf_helper import (create_injected_OmegaGW,
                                 get_sigma_from_noise_curves,
                                 create_popOmegaGW)
import transdimensional_spline_fitting as tsf
from tsf_models import RedshiftSampler
import argparse


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 20

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def main(args):
    # variables for generating injected Omega_GW
    freqs = np.arange(10, 200, 0.25)
    Tobs = 86400 * 365.25 * args.Tobs
    Lambda_0 =  {'alpha': 1.9, 'beta': 3.4, 'delta_m': 3, 'lam': 0.04,
                'mmax': 100, 'mmin': 4, 'mpp': 33, 'sigpp':5, 'gamma': 3,
                'kappa': 3.4, 'z_peak': 2, 'rate': 25}
#     num_waveforms = int(1.e4)
#     num_knots_for_redshift_model = 40
#
#     # variables for sampling the redshift distribution that calculates Omega_GW
#     num_mcmc_samples = 10_000

    mass_obj = SinglePeakSmoothedMassDistribution()
    redshift_obj = MadauDickinsonRedshift(z_max=10)

    sigma = get_sigma_from_noise_curves(['CE'], freqs, Tobs)

    # plotting everything
    # plt.loglog(injected_pop.frequency_array, injected_pop.omega_gw, color='blue', linestyle='--', label=r'new $\Omega_{\rm GW}$: $\alpha=4.5$')
    # plt.loglog(spline_pop.frequency_array, spline_pop.omega_gw, color='red', linestyle='-', label='Spline $R(z)$')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel(r'$\Omega_{\rm GW}$')
    # plt.legend()
    # plt.show()

    # plt.loglog(freqs, np.abs(fake_data))
    # plt.loglog(freqs, sigma)
    # plt.show()

    omega_gw_holder = []
    for ii in range(50):
        xvals, spline_pop, injected_pop, amplitudes, configuration, Lambda_start = \
            create_injected_OmegaGW(freqs, Lambda_0, args.num_waveforms,
                                    mass_obj, redshift_obj)
        omega_gw_holder.append(injected_pop.omega_gw)

    plt.savefig('testing.pdf')  

    with open('testing.pkl', "wb") as myf:
        cloudpickle.dump(omega_gw_holder, myf)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--Tobs', type=float,
                    help='observation time in years', default=2/52)
    parser.add_argument('--num-waveforms',
                        help='number of waveforms for resampling', type=int,
                        default=int(1e5))
    parser.add_argument("--num-mcmc-samples", type=int, default=50000,
                        help='number of mcmc samples for rj fitting')
    parser.add_argument("--max-num-knots", type=int, default=10,
                        help='max number of knots for interpolation fitting')
    parser.add_argument("--output-directory", type=str, default='./',
                        help='output directory for results')
    parser.add_argument("--run-name", type=str, default="test_run",
                        help='name of the run, used to create output files')


    args = parser.parse_args()
    main(args)
