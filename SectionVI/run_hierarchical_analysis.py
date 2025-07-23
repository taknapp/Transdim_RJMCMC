# === Standard Library ===
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# === Third-Party Libraries ===
import cloudpickle
import gwpopulation
import interpax
import jax
import jax.numpy as jnp
import lal
import matplotlib.pyplot as plt
import numpy as np
import psutil
from jax import lax
from loguru import logger

# === GWPopulation Setup ===
gwpopulation.set_backend('jax')
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift

# === Local Modules ===
sys.path.append("../modules")
sys.path.append("../modules/TransdimensionalSplineFitting")

import transdimensional_spline_fitting as tsf
import westley
from popstock.constants import H0
from popstock_tsf_helper import (
    create_injected_OmegaGW,
    get_sigma_from_noise_curves,
    create_popOmegaGW,
)
from spline_redshift import createSplineRedshift
from tsf_models import (
    RedshiftSampler,
    RedshiftSamplerJAX,
    WestleyRedshiftSamplerJAX,
)

# === lal Setup ===
lal.swig_redirect_standard_output_error(False)


# === functions for this script ===
def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"[{time.strftime('%H:%M:%S')}] Memory usage {tag}: {mem:.2f} MB")

def propagate_last_selected(x, a, mask):
    """
    Go through and anywhere that mask is set to zero, set that value
    to the last value that was turned on. i.e.

    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]
    mask = [1., 0., 0., 1]

    then:
    
    xout = [1, 1, 1, 4]
    yout = [1, 1, 1, 16]

    We use this to handle the interpolation with JAX because JAX doesn't
    want array sizes to change. So when `xout` and `yout` are passed to the 
    interpolator, the values they now have functionally make it as if we were
    just using
    [1, 4]
    [1, 16]
    as the points we were interpolating between. But we just haven't changed the array size. 
    """

    first_idx = jnp.argmax(mask)
    def step(carry, inputs):
        x_carry, a_carry = carry
        x_i, a_i, m_i = inputs
        new_x = jnp.where(m_i, x_i, x_carry)
        new_a = jnp.where(m_i, a_i, a_carry)
        return (new_x, new_a), (new_x, new_a)
    
    init = (x[first_idx], a[first_idx])
    _, (out_x, out_a) = jax.lax.scan(step, init, xs=(x, a, mask))
    return out_x, out_a

def make_calculate_omgw(spline_pop):

    H0_si = H0.si.value
    samples = spline_pop.proposal_samples
    p_masses = spline_pop.calculate_p_masses(samples, {key: Lambda_0[key] for key in spline_pop.model_args['mass']})
    p_spins = spline_pop.calculate_p_spin_models(samples, Lambda_0)
    
    def calculate_probabilities(Lambda_redshift):
        p_redshift = spline_pop.calculate_p_z(samples, Lambda_redshift)
        return p_masses*p_spins*p_redshift
    
    def calculate_weights(Lambda):
        probabilities = calculate_probabilities(Lambda)
        weights = (probabilities / spline_pop.pdraws)
        weights = jnp.where(probabilities!=0.0, weights, 0)
        return weights
    
    def omegagw(Lambda):
        redshift_model_norm_in_Gpc3 = spline_pop.models['redshift'].normalisation(Lambda)/1.e9
        Rate_norm_in_Gpc3_per_seconds = Lambda['rate']/(60*60*24*365)
        Rate_norm = redshift_model_norm_in_Gpc3 * Rate_norm_in_Gpc3_per_seconds
        weights = calculate_weights(Lambda)
        weighted_energy = jnp.nansum(weights[:, None] * spline_pop.wave_energies, axis=0) / weights.size
        conv = spline_pop.frequency_array_xp**3 * 4. * jnp.pi**2 / (3 * H0_si**2)
        return Rate_norm * conv * weighted_energy
    return omegagw


# === begin script ===
log_memory_usage("start")

jpls = jax.jit(propagate_last_selected)

mass_obj = SinglePeakSmoothedMassDistribution()
redshift_obj = MadauDickinsonRedshift(z_max=10)

# === set variables ===

Lambda_0 =  {'alpha': 2.63, 'beta': 1.26, 'delta_m': 5., 'lam': 0.1,
            'mmax': 86., 'mmin': 4.6, 'mpp': 33., 'sigpp':5.7, 'gamma': 2.7,
            'kappa': 6.1, 'z_peak': 2.4, 'rate': 25}


Tobs = 86400 * 7    # default about of time is set to a week
freqs = np.arange(10, 200, 0.25)
max_knots = 30
num_waveforms = 100_000

# === draw waveforms using popstock ===

xvals, spline_pop, injected_pop, amplitudes, configuration, Lambda_start = \
    create_injected_OmegaGW(freqs, Lambda_0, num_waveforms,
                            mass_obj, redshift_obj, backend='jax', max_knots=max_knots)


with open("tmp_jax_machinery.pkl", "wb") as f:
    cloudpickle.dump([xvals, spline_pop, injected_pop, amplitudes, configuration, Lambda_start], f)


spline_pop.calculate_omega_gw(Lambda_start)
injected_pop.calculate_omega_gw(Lambda_start)
calc_omgw = make_calculate_omgw(spline_pop)

jco = jax.jit(calc_omgw)

for ii in range(max_knots):
    Lambda_start[f'configuration{ii}'] = 1.

# === generate simulated noise ===

sigma = get_sigma_from_noise_curves(['CE'], freqs, Tobs)
fake_data = np.random.randn(freqs.size) * sigma + injected_pop.omega_gw
data_object = tsf.SmoothCurveDataObj(freqs, fake_data, sigma)

# === begin hierarchical sampling ===
try:
    wes_rs_sampler_jax = WestleyRedshiftSamplerJAX(jax.jit(calc_omgw), spline_pop, Lambda_start, data_object, max_knots,
                                [0.0, 10], [0.001, 30],
                                min_knots=0, birth_gauss_scalefac=1, birth_uniform_frac=0.3)

    wes_rs_sampler_jax.state

    wes_rs_sampler_jax.ln_likelihood(wes_rs_sampler_jax.state.configuration,
                                wes_rs_sampler_jax.state.heights,
                                wes_rs_sampler_jax.state.knots)

    import westley.parallel_tempering as wpt
    pt_sampler = wpt.ParallelTempering(wes_rs_sampler_jax, n_temps=10, max_temp=10000.0, adapt_ladder=True, t_0=1000, nu=30)
    
    N_samples = 50_000

    print('Start sampler.')
    results = pt_sampler.sample(N_samples, swap_interval=20)

    with open(f'tmp_jax_results_{N_samples}.pkl', "wb") as myf:
            cloudpickle.dump(results, myf)

    cold_chain = results.get_chain_results(0)

    # plt.plot(freqs, inj_pop.omega_gw)
    import copy
    omegas = []
    newlambda = copy.deepcopy(Lambda_start)
    for idx in np.arange(0, len(cold_chain)):
        if idx% 1000 == 0: print(idx)

        newlambda.update({f'configuration{ii}': cold_chain[idx].state.configuration[ii] for ii in range(max_knots)})
        newlambda.update({f'amplitudes{ii}': cold_chain[idx].state.heights[ii] for ii in range(max_knots)})
        newlambda.update({f'xvals{ii}': cold_chain[idx].state.knots[ii] for ii in range(max_knots)})
        plt.plot(freqs, jco(newlambda), c='C0', alpha=0.05)
        omegas.append(jco(newlambda))
    plt.savefig('test.pdf')

    with open(f'tmp_jax_omegas_{N_samples}.pkl', "wb") as myf:
            cloudpickle.dump(omegas, myf)
            
except: log_memory_usage("after allocating data")