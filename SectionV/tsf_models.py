#!/usr/bin/env python3

import sys
sys.path.append("Transdimensional_Spline_Fitting")
import numpy as np

import transdimensional_spline_fitting as tsf

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
