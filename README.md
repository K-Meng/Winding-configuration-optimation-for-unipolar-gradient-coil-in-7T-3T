# Winding-configuration-optimation-for-unipolar-gradient-coil-in-7T-3T

The parameters file contains the parameters required for optimization. main_checkthisone is the optimization program to be used for 7T, while draw_checkthisone includes some additional drawing functions.  

Files starting with T_ are programs designed for 3T scanners. Compared to 7T, since 3T scanners have more space, magnetic field homogeneity was introduced as an objective function. However, the optimization results were suboptimal.  

Files ending with _bys are programs that use Bayesian search to find the optimal parameter combinations, but their effectiveness is average. Files containing sa_ga in their names are programs that combine simulated annealing and genetic algorithms, with similarly average performance.
