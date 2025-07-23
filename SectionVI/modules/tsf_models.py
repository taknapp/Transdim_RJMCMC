#!/usr/bin/env python3

import sys
sys.path.append("Transdimensional_Spline_Fitting")
import numpy as np
from copy import deepcopy
from scipy.stats import norm
import random
from gwpopulation.utils import xp

import transdimensional_spline_fitting as tsf
import westley.fitter
from spline_redshift import propagate_last_selected
from interpax import interp1d as interpax_interp1d

import jax
jplsn = jax.jit(propagate_last_selected)

class RedshiftSampler(tsf.BaseSplineModel):
    """
    Redshift Sampling Model:

    Be sure to run `set_base_population_information(pop_object,lambda_0)`
    before sampling. `pop_object` will be used to calculated omega_gw
    and lambda_0 are the stock set of parameters for the population model,
    many of which we need for the mass models, which we assume for now to be known.
    """
    def set_base_population_information(self, pop_object, lambda_0):
        self.pop_object = pop_object
        self.lambda_0 = lambda_0

    def ln_likelihood(self, config, heights, knots):
        # construct what needs to go into calculating omega_gw
        params = self.lambda_0
        params.update({**{f'amplitudes{ii}': heights[ii] for ii in range(heights.size)},
                  **{f'configuration{ii}': config[ii] for ii in range(config.size)},
                  **{f'xvals{ii}': knots[ii] for ii in range(self.available_knots.size)},
                        })
        self.pop_object.calculate_omega_gw(params, multiprocess=False)
        return np.sum(-0.5 * (self.data.data_yvals - self.pop_object.omega_gw)**2 / (2 * self.data.data_errors**2))

class RedshiftSamplerJAX(tsf.BaseSplineModel):
    def set_base_population_information(self, omgw_func, pop_object, lambda_0):
        self.pop_object = pop_object
        self.lambda_0 = lambda_0
        self.omgw_func = omgw_func

    def ln_likelihood(self, config, heights, knots):
        params = deepcopy(self.lambda_0)
        params.update({**{f'amplitudes{ii}': heights[ii] for ii in range(heights.size)},
                  **{f'configuration{ii}': config[ii] for ii in range(config.size)},
                  **{f'xvals{ii}': knots[ii] for ii in range(self.available_knots.size)},
                        })
        omgw = self.omgw_func(params)
        return np.nansum(-0.5 * (self.data.data_yvals - omgw)**2 / (2 * self.data.data_errors**2) - 0.5 * np.log(2 * np.pi * self.data.data_errors**2))

    def evaluate_interp_model(self, redshift, heights, configuration, knots):

        if self.pop_object.backend=='numpy':
            if np.sum(configuration)==0:
                tmp = 0 * xp.zeros(redshift.size)
            elif np.sum(configuration)==1:
                tmp = xp.ones(redshift.size) * amplitudes[configuration]
            else:
                tmp = interp1d(xvals[configuration],
                               amplitudes[configuration],
                               fill_value="extrapolate")(redshift)

        elif self.pop_object.backend=='jax':
            xvals_new, amplitudes_new = jplsn(knots, heights, configuration)
            tmp = interpax_interp1d(redshift, xvals_new,
                           amplitudes_new,
                           extrap=True, method='linear')
        return tmp
        
    def propose_death_move(self, specific_idx=None):
        """
        propose to "turn off" one of the current knots that are turned on.
        This is the same as the death move in the base model, except this one
        does not allow you to turn off the endpoints.
        """
        if np.sum(self.configuration) == self.min_knots:
            return (-np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots)
        else:
            # pick one to turn off
            idx_to_remove = np.random.choice(np.where(self.configuration[1:-1])[0]) + 1
            new_heights = deepcopy(self.current_heights)
            new_config = deepcopy(self.configuration)

            # turn it off
            if specific_idx is None:
                new_config[idx_to_remove] = False
            else:
                idx_to_remove = specific_idx
                new_config[idx_to_remove] = False
                
    
            # Find mean of the Gaussian we would have proposed from
            height_from_model = self.evaluate_interp_model(self.available_knots[idx_to_remove],
                                                           self.current_heights, new_config, self.available_knots)

            log_qx = np.log(self.birth_uniform_frac / self.yrange + \
                              (1 - self.birth_uniform_frac) * norm.pdf(self.current_heights[idx_to_remove],
                                                                          loc=height_from_model,
                                                                          scale=self.birth_gauss_scalefac))
            log_qy = 0
            
            log_px = self.get_height_log_prior(self.current_heights[idx_to_remove]) # + self.get_width_log_prior(self.available_knots[idx_to_remove], idx_to_remove)
            
            log_py = 0

            new_ll = self.ln_likelihood(new_config, self.current_heights, self.available_knots)
            
            return new_ll, (log_py - log_px) + (log_qx - log_qy), new_config, new_heights, self.available_knots

class WestleyRedshiftSamplerJAX(westley.fitter.BaseSplineModel):
    def __init__(self, omgw_func, pop_object, lambda_0, *args, **kwargs):
        self.omgw_func = omgw_func
        self.pop_object = pop_object
        self.lambda_0 = lambda_0
        super(WestleyRedshiftSamplerJAX, self).__init__(*args, **kwargs)
        self.init_args = (self.omgw_func, self.pop_object, self.lambda_0) + self.init_args
        self.state.knots[0] = 0.
        self.state.knots[-1] = self.xhigh
        self.state.heights[0] = 1.

    # def set_base_population_information(self, omgw_func, pop_object, lambda_0):
        # self.pop_object = pop_object
        # self.lambda_0 = lambda_0
        # self.omgw_func = omgw_func

    def ln_likelihood(self, config, heights, knots):
        params = self.lambda_0
        params.update({**{f'amplitudes{ii}': heights[ii] for ii in range(heights.size)},
                  **{f'configuration{ii}': config[ii] for ii in range(config.size)},
                  **{f'xvals{ii}': knots[ii] for ii in range(self.available_knots.size)},
                        })
        omgw = self.omgw_func(params)
        return np.sum(-0.5 * (self.data.data_yvals - omgw)**2 / (2 * self.data.data_errors**2) - 0.5 * np.log(2 * np.pi * self.data.data_errors**2))

    def copy(self):
        # Recreate a new instance with the same arguments, avoiding closure traps
        return type(self)(*self.init_args, **self.init_kwargs)

    def evaluate_interp_model(self, redshift, heights, configuration, knots):

        if self.pop_object.backend=='numpy':
            if np.sum(configuration)==0:
                tmp = 0 * xp.zeros(redshift.size)
            elif np.sum(configuration)==1:
                tmp = xp.ones(redshift.size) * amplitudes[configuration]
            else:
                tmp = interp1d(xvals[configuration],
                               amplitudes[configuration],
                               fill_value="extrapolate")(redshift)

        elif self.pop_object.backend=='jax':
            xvals_new, amplitudes_new = jplsn(knots, heights, configuration)
            tmp = interpax_interp1d(redshift, xvals_new,
                           amplitudes_new,
                           extrap=True, method='linear')
        return tmp

    @westley.fitter.proposal(name='change_knot_location', weight=1)
    def change_knot_location(self):
        """Change the location of an existing knot."""
        if self.state.configuration.sum() == 2:
            return None
        active_idx = np.where(self.state.configuration[1:-1])[0] + 1
        idx_to_change = random.choices(active_idx, k=1)[0]
        
        new_knots = self.state.knots.copy()
        new_knots[idx_to_change] = (np.random.rand() * 
            (self.xhighs[idx_to_change] - self.xlows[idx_to_change]) + 
            self.xlows[idx_to_change])
        
        new_ll = self.ln_likelihood(
            self.state.configuration, self.state.heights, new_knots)
        
        return westley.fitter.ProposalResult(
            new_ll, 0.0, self.state.configuration.copy(),
            self.state.heights.copy(), new_knots
        )

    @westley.fitter.proposal(name='death', weight=1)
    def death(self):
        """Death proposal: Remove an existing knot."""
        active_idx = np.where(self.state.configuration[1:-1])[0] + 1 # don't kill the first point.
        if len(active_idx) <= self.min_knots:
            return None

        idx_to_remove = random.choices(active_idx, k=1)[0]
        new_config = self.state.configuration.copy()
        new_config[idx_to_remove] = False

        log_ratio = self._calculate_death_ratio(
            idx_to_remove, 
            self.state.heights,
            self.state.knots
        )

        new_ll = self.ln_likelihood(new_config, self.state.heights, self.state.knots)
        return westley.fitter.ProposalResult(
            new_ll, log_ratio, new_config, 
            self.state.heights.copy(), self.state.knots.copy()
        )

    @westley.fitter.proposal(name='change_amplitude_prior_draw', weight=1)
    def change_amplitude_prior_draw(self):
        """Change amplitude by drawing from the prior."""
        if self.state.configuration.sum() == 1:
            return None
        active_idx = np.where(self.state.configuration[1:])[0] + 1
        idx_to_change = random.choices(active_idx, k=1)[0]

        new_heights = self.state.heights.copy()
        new_heights[idx_to_change] = (np.random.rand() * 
            (self.yhigh - self.ylow) + self.ylow)

        log_py_before = self.get_height_log_prior(self.state.heights[idx_to_change])
        log_py_after = self.get_height_log_prior(new_heights[idx_to_change])
        log_ratio = log_py_after - log_py_before

        new_ll = self.ln_likelihood(
            self.state.configuration, new_heights, self.state.knots)

        return westley.fitter.ProposalResult(
            new_ll, log_ratio, self.state.configuration.copy(),
            new_heights, self.state.knots.copy()
        )

    @westley.fitter.proposal(name='change_amplitude_gaussian', weight=1)
    def change_amplitude_gaussian(self):
        """Change amplitude using a Gaussian proposal."""
        if self.state.configuration.sum() == 1:
            return None
        active_idx = np.where(self.state.configuration[1:])[0] + 1
        idx_to_change = random.choices(active_idx, k=1)[0]

        new_heights = self.state.heights.copy()
        new_heights[idx_to_change] += norm.rvs(scale=self.birth_gauss_scalefac)

        log_py_before = self.get_height_log_prior(self.state.heights[idx_to_change])
        log_py_after = self.get_height_log_prior(new_heights[idx_to_change])
        log_ratio = log_py_after - log_py_before

        new_ll = self.ln_likelihood(
            self.state.configuration, new_heights, self.state.knots)

        return westley.fitter.ProposalResult(
            new_ll, log_ratio, self.state.configuration.copy(),
            new_heights, self.state.knots.copy()
        )

    def get_height_log_prior(self, height):
        """Log-uniform prior Calculate the log prior for a given height."""
        
        if height < self.ylow or height > self.yhigh:
            return -np.inf
        prior = 1/ (np.log(self.yhigh / (self.ylow + 1e-10)) * height) # log uniform prior
        return np.log(prior)