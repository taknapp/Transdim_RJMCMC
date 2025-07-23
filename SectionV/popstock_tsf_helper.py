
import numpy as np
import popstock
import bilby
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 15

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from scipy.interpolate import interp1d
import lal
lal.swig_redirect_standard_output_error(False)
import pandas as pd

import transdimensional_spline_fitting as tsf

from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from popstock.PopulationOmegaGW import PopulationOmegaGW

from pygwb.baseline import Baseline
import astropy.cosmology

R0 = 31.4
H0 = astropy.cosmology.Planck18.H0.to(astropy.units.s**-1).value


class SmoothCurveDataObj(object):
    """
    A data class that can be used with our spline model
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitRedshift(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    def ln_likelihood(self, config, heights):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        redshift_model = 10**self.evaluate_interp_model(np.log10(bbh_pickle.ref_zs), heights, config, log_xvals=True)
        
        model = bbh_pickle.eval(R0, redshift_model, self.data.data_xvals)
        
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))
    
class FitOmega(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    
    def ln_likelihood(self, config, heights, knots):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        omega_model = 10**self.evaluate_interp_model(np.log10(self.data.data_xvals), heights, config, np.log10(knots))

        return np.sum(norm.logpdf(omega_model - self.data.data_yvals, scale=self.data.data_errors))

def get_sigma_from_noise_curves(detector_names, freqs, obs_T):
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
        elif 'ET' in detector_names:
            ET = bilby.gw.detector.TriangularInterferometer('ET')
            ET.frequency_array = freqs
            detectors.append(ET)
    
    else:
        if 'ET' in detector_names:
            detectors.append(bilby.gw.detector.TriangularInterferometer('ET'))
            detectors.append(bilby.gw.detector.get_empty_interferometer(detector_names[1]))
            detectors[0].frequency_array = freqs
            detectors[1].strain_data.frequency_array = freqs
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

def generate_data(signal_name, freqs, sigma, BPL_turnover=60):
    if signal_name == 'BPL':
        signal = np.zeros(freqs.size)
        
        N = int(list(freqs).index(BPL_turnover))
        
        signal[:N] = 2e-10 * (freqs[:N] / 10)**(2) 
        signal[N:] = 2e-10 * (freqs[N] / 10)**(2) * (freqs[N:] / freqs[N])**(-2) 

    elif signal_name == 'FOPT':
        Omega_star = 1.8e-10
        f_star = 30 # [Hz], where knee break occurs
        alpha1 = 3
        alpha2 = -4
        delta = 2
        signal = Omega_star * (freqs / f_star)**alpha1 * (1 + (freqs / f_star)**delta )**((alpha2 - alpha1)/delta)

    elif signal_name == 'classic':
        dfile = np.load('omegagw_0_BPL_1000000_samples.npz', allow_pickle=True)
        x = dfile['omega_gw']
        y = dfile['freqs']
        interp_func = interp1d(y, x, kind='linear', fill_value='extrapolate')
        print('freqs : ', y)
        print('omegas: ', x)
        signal = interp_func(freqs)

    elif signal_name == 'Sachdev':
        file_path = 'Sachdev_curve.csv'
        data = pd.read_csv(file_path)
        print(data.columns)
        x = data['x']
        y = data[' y']
        interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
        signal = interp_func(freqs)
    
    data = signal + sigma * np.random.randn(freqs.size)
    data_obj = tsf.SmoothCurveDataObj(freqs, data, sigma)

    return signal, data, data_obj


def sample_Omega(freqs, N_samples, data_obj, N_possible_knots=30, interp_type='linear'):
    # # fitting Omega directly
    # original prior range was (-13,-3)
    fit_omega = FitOmega(data_obj, N_possible_knots, (freqs[0], freqs[-1]), (-13, -3), log_output=True, log_space_xvals=True, min_knots=0, interp_type=interp_type)
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
    return np.array(knot_configs).astype('int'), np.array(num_knots).astype('int')

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
       #print(len(fit_results_omega.knots[:, ii]))
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
    plt.ylim(1e-14, 1e-3)
    plt.yscale('log')
    plt.xlabel('freqs [Hz]')
    plt.ylabel('$\Omega_{GW}$')
    plt.show()

def BPL_helper(x, n, a):
    signal = np.zeros(x.size)
    N = int(n)
    signal[:N] = 2e-10 * (x[:N] / 10)**(a) 
    signal[N:] = 2e-10 * (x[N] / 10)**(a) * (x[N:] / x[N])**(-a) 
    return signal

def plot_posterior_fits(fit_omega, fit_results_omega, freqs, N_samples, offset=0, num_posteriors=1000, label_str='', alpha=0.75, color='red'):
    choices = N_samples - offset
    fit_funcs2 = []

    for ii in range(num_posteriors):
        idx = np.random.choice(np.arange(choices))
        fit_func2 = 10**fit_omega.evaluate_interp_model(np.log10(fit_omega.data.data_xvals), fit_results_omega.heights[int(idx+offset)], fit_results_omega.configurations[int(idx+offset)].astype(bool), np.log10(fit_results_omega.knots[int(idx+offset)]))
        fit_funcs2.append(fit_func2)

    fit_funcs = []
    for func in fit_funcs2: 
        if np.any(np.isnan(func)):
            continue
        else: 
            if all(i < 1e-3 and i > 1e-20 for i in func):
                fit_funcs.append(func)
    
    #plt.plot(freqs, fit_funcs, alpha=0.01, c='tab:blue')
    bounds = np.percentile(np.array(fit_funcs), [5,95], axis=0)
    plt.fill_between(freqs, *np.percentile(np.array(fit_funcs), [5,95], axis=0), alpha= alpha, label = label_str, color=color)
    plt.plot(freqs, bounds[0], color=color)
    plt.plot(freqs, bounds[1], color=color)

    return

def calc_Bayes(num_knots, knot_cutoff=2):
    numer = 0
    denom = 0
    try:
        if knot_cutoff == 0:
            return len([i for i in num_knots if i > 0]) / len([i for i in num_knots if i == 0])
        else:
            return len([i for i in num_knots if i > knot_cutoff]) / len([i for i in num_knots if i > 0 and i <= knot_cutoff])
    except: return np.inf