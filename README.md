# Transdim_RJMCMC
Files required to produce samples for arXiv:2507.08095

The organization of both folders is for ease in downloading and running the associated scripts. If there are script redundancies, it is such that each folder may be downloaded and run independently. 

## Section V

Files in this folder pertain to the injections and recoveries showcased in Sec. V of the relevant manuscript. 

## Section VI

Files in this folder pertain to the hierarchical injections and recoveries showcased in Sec. VI of the relevant manuscript.

Install with `conda env create -f env.yml` This will install all the latest dependencies, from the correct place, so that `popstock` will work with JAX (from the `meyers-academic` fork and `jax` branch), and the newest version of `westley` is installed for parallel tempering.

To run the hierarchical sampler, please run the `run_hierarchical_analysis.py` script in the `westley_fitter` environment created above. The `run_hierarchical_analysis.py` saves three files as outputs: 
1. `tmp_jax_machinery.pkl` : Includes the injected signals and parameters as well as starting knot locations in the frequency space.
2. `tmp_jax_results_{N_samples}.pkl` : Includes the results object from the sampling. This encapsulates the posterior information in the redshift space.
3. `tmp_jax_omegas_{N_samples}.pkl` : Includes the computed Omega(f) draws computed from each of the redshift space posteriors. 


For questions about this repository, please contact Taylor Knapp (tknapp@caltech.edu). Thank you!
