
import numpy as np
import popstock
import bilby
from spline_redshift import createSplineRedshift

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

H0 = 2e-18 # seconds

from pygwb.baseline import Baseline

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from scipy.interpolate import interp1d

import transdimensional_spline_fitting as tsf

from popstock.PopulationOmegaGW import PopulationOmegaGW

import lal
lal.swig_redirect_standard_output_error(False)


def get_sigma_from_noise_curves(detector_names, freqs, obs_T):
    """
    Make noise curves given names of detectors, frequencies and an observation time (in seconds)

    Parameters:
    -----------
    detector_names: list
        list of detector names
    freqs: numpy.ndarray
        frequency array at which to evaluate noise curve
    obs_T: float
        observation time (in seconds)

    Returns:
    --------
    sigma : numpy.ndarray
        1-sigma sensitivity noise curve for Omega_gw(f)
    """
    # make empty detectors
    detectors = []

    if len(detector_names) == 1:
        if 'CE' in detector_names:
            CE = bilby.gw.detector.get_empty_interferometer('CE')
            CE1 = bilby.gw.detector.get_empty_interferometer('H1')
            CE2 = bilby.gw.detector.get_empty_interferometer('L1')
            CE.strain_data.frequency_array = freqs
            CE1.strain_data.frequency_array = freqs
            CE2.strain_data.frequency_array = freqs
            CE1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array = freqs, psd_array=CE.power_spectral_density_array)
            CE2.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array = freqs, psd_array=CE.power_spectral_density_array)
            detectors.append(CE1)
            detectors.append(CE2)
    else:
        detectors.append(bilby.gw.detector.get_empty_interferometer(detector_names[0]))
        detectors.append(bilby.gw.detector.get_empty_interferometer(detector_names[1]))
        for det in detectors: 
            det.strain_data.frequency_array = freqs

    # make baseline & calculate ORF
    duration = 1/np.abs(freqs[1] - freqs[0])
    BL = Baseline('BL', detectors[0], detectors[1], frequencies=freqs, duration=duration)
    BL.orf_polarization = 'tensor'

    # calculate sigma^2
    df = np.abs(freqs[1] - freqs[0])

    S0 = 3 * H0**2 / (10 * np.pi**2 * freqs**3)
    sigma_2 = (detectors[0].amplitude_spectral_density_array)**2 * (detectors[1].amplitude_spectral_density_array)**2 / (2 * obs_T * df * BL.overlap_reduction_function **2 * S0**2)

    return np.sqrt(sigma_2)

def generate_data(signal_name, freqs, sigma, omega_break, f_break_turnover=60, fref=10, alpha_pre=2, alpha_post=-2):
    """
    Generate a broken power law signal
    """
    if signal_name == 'BPL':
        signal = np.zeros(freqs.size)
        N = np.argmin(np.abs(freqs - f_break_turnover))

        signal[:N] = omega_break * (freqs[:N] / fref)**(alpha_pre)
        signal[N:] = omega_break * (freqs[N] / fref)**(alpha_post) * (freqs[N:] / freqs[N])**(alpha_post)

    elif signal_name == 'squiggly':
        signal = 1e-7 * np.sin(freqs / fref) + 2e-7

    elif signal_name == 'Sachdev':
        file_path = 'Plotdata.csv'
        data = pd.read_csv(file_path)
        x = data['x'] + 15
        y = data[' y'] * 100
        interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
        signal = interp_func(freqs)
    
    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = tsf.SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj


def sample_Omega(freqs, N_samples, data_obj, N_possible_knots=30):
    # # fitting Omega directly
    fit_omega = FitOmega(data_obj, N_possible_knots, (freqs[0], freqs[-1]), (-13, -3), log_output=True, log_space_xvals=False, min_knots=0)
    fit_results_omega = fit_omega.sample(N_samples, proposal_weights=(1, 1, 1, 1, 1), prior_test=False)
    return fit_omega, fit_results_omega

######

def return_lls(fit_omega, fit_results_omega, freqs, signal, N_samples, offset=0, N_possible_knots=30):
    lls = []
    for ii in np.arange(offset,N_samples):
        signal_vals = np.interp(fit_results_omega.knots[ii], freqs, signal)
        lls.append(fit_omega.ln_likelihood(np.ones(N_possible_knots).astype(bool),np.log10(signal_vals), fit_results_omega.knots[ii]))
    return lls

def return_knot_info(fit_results_omega, offset=0):
    knot_configs = fit_results_omega.configurations[offset:, :]
    num_knots = knot_configs.sum(axis=1)
    return knot_configs, num_knots

def return_knot_placements(fit_results_omega, offset=0):
    all_weights = []
    all_bins = []
    for ii in range(fit_results_omega.knots.shape[1]):
        if np.sum(fit_results_omega.configurations.astype(bool)[offset:, ii]) > 0:    
            weights, bins, x = plt.hist(fit_results_omega.knots[fit_results_omega.configurations.astype(bool)[:, ii], ii])
            all_weights.append(weights)
            all_bins.append(bins[:-1])
    #plt.show()
    return all_bins, all_weights

def return_knot_heights(fit_results_omega, offset=0, toggle=False):
    if toggle:
        knot_heights = []
        for ii in range(fit_results_omega.knots.shape[1]):
            knot_heights.append(10**(fit_results_omega.heights[fit_results_omega.configurations.astype(bool)[:, ii], ii]))
    else: 
        knot_heights = 10**fit_results_omega.heights[offset:, :]
    return knot_heights

def return_knot_frequencies(fit_results_omega, offset=0, toggle=False):
    temp = []
    for ii in range(fit_results_omega.knots.shape[1]):
        if toggle:
            temp.append(fit_results_omega.knots[fit_results_omega.configurations.astype(bool)[:, ii], ii])
        else:
            temp.append(fit_results_omega.knots[:, ii])
    return temp

######

def plot_knot_placements(freqs, signal, fit_results_omega, offset=0, toggle=False):
    # toggle config to plot all knots or just knots turned on/off
    # config = True -> only knots toggled on are plotted 
    xs = return_knot_frequencies(fit_results_omega, offset=offset, toggle=toggle)
    ys = return_knot_heights(fit_results_omega, offset=offset, toggle=toggle)
    if toggle: 
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i])
    else: 
        plt.scatter(xs, ys)
    
    plt.loglog(freqs, signal)
    #plt.xlim(min(freqs), max(freqs))
    plt.ylim(1e-9, 1e-5)
    plt.yscale('log')
    plt.xlabel('freqs [Hz]')
    plt.ylabel('$\Omega_{GW}$')
    plt.show()

def plot_posterior_fits(fit_omega, fit_results_omega, freqs, N_samples, offset=0, num_posteriors=1000):
    choices = N_samples - offset
    for ii in range(num_posteriors):
        idx = np.random.choice(np.arange(choices))
        plt.plot(freqs, 10**fit_omega.evaluate_interp_model(np.log10(fit_omega.data.data_xvals), fit_results_omega.heights[idx+offset], fit_results_omega.configurations[idx+offset].astype(bool), np.log10(fit_results_omega.knots[idx+offset])), alpha=0.01, c='k')
    return

def create_popOmegaGW(freqs, mass_obj, redshift_obj, backend='numpy'):
    """
    Create population OmegaGW model from a gwpopulations mass object and
    redshift object."""
    # choose models for calculating Omega_GW
    models = {'mass_model' : mass_obj,'redshift_model' : redshift_obj,}
    # Populations object requires mass and redshift distributions and frequencies
    pop_obj = PopulationOmegaGW(models=models, frequency_array=freqs, backend=backend)
    return pop_obj

def create_injected_OmegaGW(freqs, Lambda_0, N_proposal_samples, mass_obj, redshift_obj, max_knots=10, backend='numpy'):
    """
    Create an injected Omega_GW
    """
    # create population object
    injection_pop = create_popOmegaGW(freqs, mass_obj, redshift_obj, backend=backend)

    # We need to define Λ_0 hyperparameters for population (must match formalism in the redshift and mass models) and number of desired samples
    injection_pop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)

    # Calculates Ω_GW
    injection_pop.calculate_omega_gw(Lambda=Lambda_0, multiprocess=False)

    # create a new set of Lambdas, with a different value of alpha
    injection_pop.calculate_omega_gw(Lambda=Lambda_0)

    # additional arguments needed to pass into sampling to use the spline redshift model
    args = np.argsort(injection_pop.proposal_samples['redshift'])

    xvals = np.linspace(min(injection_pop.proposal_samples['redshift']), max(injection_pop.proposal_samples['redshift']), num=max_knots)
    # these can be our starting amplitudes
    amplitudes = np.interp(xvals, injection_pop.proposal_samples['redshift'][args], injection_pop.models['redshift'].psi_of_z(injection_pop.proposal_samples['redshift'], **Lambda_0)[args])
    # we'll start with all knots turned on
    configuration = np.ones(amplitudes.size, dtype=bool)

    # Keep the mass model the same, redshift model becomes the spline redshift
    splredshift = createSplineRedshift(amplitudes.size)(z_max=10, backend=injection_pop.backend)
    models = {'mass_model' : mass_obj,'redshift_model' : splredshift}

    # instantiate population object using the spline redshift model
    pop_for_sampling = PopulationOmegaGW(models=models, frequency_array=freqs, backend=backend)

    # define the hyperparameters (need additional ones to Λ to match additional parameters needed for the spline redshift model)
    params_start = {**{f'amplitudes{ii}': amplitudes[ii] for ii in range(amplitudes.size)},
                    **{f'xvals{ii}': xvals[ii] for ii in range(xvals.size)},
                    **{f'configuration{ii}': configuration[ii] for ii in range(configuration.size)}}
    # now add our spline initial knots, amplitudes, configurations
    # to the parameters dictionary.
    Lambda_start = {**params_start, **Lambda_0}

    # number of waveforms to use for resampling
    # N_proposal_samples = int(4.e4)

    # sample and calulate Ω_GW for spline redshift model
    # we're kinda cheating here by using a "fiducial" or "starting"
    # model that is the same as the "true" model.
    pop_for_sampling.draw_and_set_proposal_samples(Lambda_start, N_proposal_samples=N_proposal_samples)
    pop_for_sampling.calculate_omega_gw(Lambda=Lambda_start, multiprocess=False)

    return xvals, pop_for_sampling, injection_pop, amplitudes, configuration, Lambda_start
