# Transdim_RJMCMC
Files required to produce samples for arXiv:2507.08095

The organization of both folders is for ease in downloading and running the associated scripts. If there are script redundancies, it is such that each folder may be downloaded and run independently. 

## Section V

Files in this folder pertain to the injections and recoveries showcased in Sec. V of the relevant manuscript. 

## Section VI

Files in this folder pertain to the hierarchical injections and recoveries showcased in Sec. VI of the relevant manuscript.

Install with `conda env create -f env.yml` This will install all the latest dependencies, from the correct place, so that `popstock` will work with JAX (from the `meyers-academic` fork and `jax` branch), and the newest version of `westley` is installed for parallel tempering.

To run the hierarchical sampler, please run the `run_hierarchical_analysis.py` script in the `westley_fitter` environment created above. 

For questions about this repository, please contact Taylor Knapp (tknapp@caltech.edu). Thank you!
