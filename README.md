# Neutron-attenuation
Monte-carlo simulation of neutrons passing through materials
#
Ben Shortland
University of Manchester
Computational Physics
March 2018

Functions:
 - Chi: Find goodness of fit between x and y
 - randssp: Tests randomness of randssp
   o randssp_plot: plot randssp data
 - Number_Generator_3D: Generates a 3D array of random numbers, using random.uniform()
   o Plotter_3D: plots random numbers
 - Random_check: shows how uniform random.randint() is
 - Exponential_Distribution: Finds an exponential distibtution of values for a given Lanbda
 - Sphere_Surface: Finds n random positions on the surface of a sphere with radius r
 - Sphere_Distribution: Finds n random positions with a random r wich follows the exponential distribution.
 - Isotropic_Steps: Function to find the number of transmitted, reflected and absorbed neutrons, for a given material
 - Uncertainty_with_neutrons: Function which calls Isotropic_Steps to return array of data for n simulations
 - Neutrons_Through_Fixed_Tickness: Finds number of neutrons that have passed through a material of a given thickness
   o Neutrons_Through_varying_Thickness_analysis: analyses data output by previsous function
   
Main:
  Defines materials and calls each function. Uses Matplotlib to plot graphs
