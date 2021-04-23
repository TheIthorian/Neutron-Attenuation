# Neutron-attenuation
Monte-carlo simulation of neutrons passing through materials
#
University of Manchester
Computational Physics
March 2018

Functions:
 - Chi: Find goodness of fit between x and y
 - randssp: Tests randomness of randssp
  - randssp_plot: plot randssp data
 - Number_Generator_3D: Generates a 3D array of random numbers, using random.uniform()
  - Plotter_3D: plots random numbers
 - Random_check: shows how uniform random.randint() is
 - Exponential_Distribution: Finds an exponential distibtution of values for a given Lanbda
 - Sphere_Surface: Finds n random positions on the surface of a sphere with radius r
 - Sphere_Distribution: Finds n random positions with a random r wich follows the exponential distribution.
 - Isotropic_Steps: Function to find the number of transmitted, reflected and absorbed neutrons, for a given material
 - Uncertainty_with_neutrons: Function which calls Isotropic_Steps to return array of data for n simulations
 - Neutrons_Through_Fixed_Tickness: Finds number of neutrons that have passed through a material of a given thickness
  - Neutrons_Through_varying_Thickness_analysis: analyses data output by previsous function
   
Main:
  Defines materials and calls each function. Uses Matplotlib to plot graphs


In a nuclear reactor, uranium undergoes induced fission when its nucleus captures
neutrons to gain enough energy to split and release more neutrons which induce
more fission reactions. The attenuation length of neutrons in different materials is
important to know when constructing a nuclear reactor to ensure safety outside the
reactor and to keep the chain reaction sub-critical.

This python script constains functions for simulating the scattering and absorption of
neutrons through different materials that are commonly used in a reactor to find the
attenuation length of the neutrons though the materials. This is done using a Markov
chain Monte Carlo technique of isometric steps as the neutron passes through the
material
