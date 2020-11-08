## Thermal Neutron Simulation
# Project_3_script.py
#
# --------------------------------------------------------------
# TheIthorian
# University of Manchester
# Computational Physics
# March 2018
# --------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import math
import matplotlib.patches as mpatches
#------------------------------
# Functions
#------------------------------


# Finds the goodness of fit between x and y.
def Chi(x,y,sigma_y,yfit):
    chi = 0
    for i in range(len(x)):
        chi = chi + ((yfit[i] - y[i])/(sigma_y[i]))**2
    chi = chi / (len(x) - 2)
    return chi

# Trying randsp:
def randssp(p,q):
    global m, a, c, randssp_x
    m = pow(2, 31)
    a = pow(2, 16) + 3
    c = 0
    
    x = randssp_x
    try: p
    except NameError:
        p = 1
    try: q
    except NameError:
        q = p
    
    r = np.zeros([p,q])
    for l in range (0, q):
        for k in range (0, p):
            x = np.mod(a*x + c, m)
            r[k, l] = x/m

    return r

#------------------------------

def randssp_plot():
    global randssp_x 
    randssp_x = 123456789
    u = randssp(10000, 1)
        
    fig = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    x, y, z, title, x_label, y_label, z_label = u[0:-3], u[1:-2], u[2:-1], 'Consecutive RANDSSP random numbers', 'x', 'y', 'z'
        
    ax.text2D(0.25, 1, title, transform = ax.transAxes, size = 18)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # Plot the data points.
    ax.scatter(x, y, z, s = 1, c = 'g')
    plt.show()


#------------------------------

# Random Number Generator:
# Creates a File of 3 Dimetntions.
def Number_Generator_3D (Number_Range, n):
    File = open ("3D_Numbers.txt", "w")
    
    # Write 3 randoms intergers between 0 and Number_Range, n times.
    for i in range(n):
        # 3 Components of each line (x,y,z)
        x = random.uniform(0,Number_Range)
        y = random.uniform(0,Number_Range)
        z = random.uniform(0,Number_Range)
        
        # Write to file
        File.write (str(x) + "\t" + str(y) + "\t" + str(z) + "\n")
    File.close()

#------------------------------
# Generate 3n Random Numbers to plot in 3D. (n for each dimention)
def Plotter_3D (Filename):
    # Parameters for 3D Random Number Generator:
    x, y, z = [], [], []
    
    # Create Arrays of Numbers
    File = open(Filename, "r")
    for line in File:
        SplitUp = line.split("\t")
        x.append(float(SplitUp[0]))
        y.append(float(SplitUp[1]))
        z.append(float(SplitUp[2]))
    File.close()
    
    fig = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

#------------------------------
    
# Check that random.randint is random:
def Random_Check ():
    # Show uniform-ness
    x = []
    for i in range (100000):
        xs = random.randint(0, 20)
        x.append(xs)
    fig = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    plt.hist(x, bins = 21, density = True)
    plt.xlabel("Ui")
    plt.ylabel("Frequancy (Normalised)")
    plt.show()
    
    # Show a gaussin distribution
    Sums = []
    # Sum 50,000 times
    for i in range (50000):
        Sum = 0
        # Sum over 50 random variables
        for i in range (50):
            x = random.randint(0, 100)
            Sum = Sum + x
        Sums.append(Sum)
    fig = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    Num_Bins = math.ceil(len(Sums)**0.5)
    # Create histogram
    plt.hist(Sums, bins = Num_Bins)
    plt.title("Distribution of sums")
    plt.xlabel("Sum of random variable")
    plt.ylabel("Frequency")
    plt.show()
    
#------------------------------

# Finds an exponential distibtution of values for a given Lanbda
def Exponential_Distribution (Lambda, Number_of_Particles):
    # x is the variable exponentially distributed
    x = []
    # Appending the random variable
    for i in range (Number_of_Particles):
        x.append(- Lambda * np.log ((1 - random.uniform(0, 0.99))))

    # Creating a histogram of the variabe.
    # Number of bins is root of number of variables
    root_k = math.ceil(len(x) ** (1/2))
    no_of_bins = root_k    
    print("Number of bins = " + str(no_of_bins))
    frequency, bins = np.histogram(x, bins = no_of_bins, normed = False)   

    # Plotting the histogram
    fig1 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    plt.hist(x, bins = no_of_bins)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Frequency")
    fig1.show()
    plt.show()

    Mid_Bin = []
    # Finds the mid point of the bins
    for i in range(len(bins)):
        if i != len(bins) - 1:
            # Finds the mid point of the bins
            temp = ( bins[i] + bins[i+1] ) / 2
            Mid_Bin.append(temp)

    # Plots the histogram as points       
    fig2 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    plt.plot(Mid_Bin, frequency, marker = "x", linestyle = " ")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # Finds the nautral log of the frequency and finds the fit between this and the mid bin points

    # Removes the points with no frequency
    index = 0
    while index < len(frequency):
        if frequency[index] == 0:
            frequency = np.delete(frequency, index)
            Mid_Bin = np.delete(Mid_Bin, index)
        else:
            index = index + 1

    # Finds the log of the frequency and the fit        
    Log_frequency = np.log(frequency)
    coef, covr = np.polyfit (Mid_Bin, Log_frequency, 1, cov = True)
    y_fit = np.polyval(coef, Mid_Bin)

    # Plots line of best fit
    fig3 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
    plt.plot(Mid_Bin, Log_frequency, marker = "x", linestyle = " ")
    plt.plot(Mid_Bin, y_fit, color = "peru", linestyle = "-")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Log (Frequency)")
    plt.show()

    #Shows user results
    print ("Gradient: " + str(coef[0]) + " +/- " + str(round(covr[0][0] ** 0.5, 5)))
    print ("Intercept: " + str(coef[1]) + " +/- " + str(round(covr[1][1] ** 0.5, 5)))
    print ("Calculated Lambda: " + str(round(-1 / coef[0], 5)) + " +/- " + str(round((covr[0][0] ** 0.5 ) / (coef[0] ** 2), 5)))

#------------------------------
   
# Finds n random positions on the surface of a sphere with radius r.
def Sphere_Surface (r, n):
    x, y, z, = [],[],[]
    
    x, y, z, = [],[],[]
    
    # Finds the coordinates and writes to a file
    File = open ("Sphere_Surface_Plot.txt", "w")
    
    for i in range (n):        
        theta = np.arccos(1 - (2 * np.random.uniform(0, 1)))
        phi = 2 * np.pi * np.random.uniform(0, 1)

        # Convert from spherical to cartesian coordinates.
        z = r * np.cos(theta)
        y = r * np.sin(theta) * np.sin(phi)
        x = r * np.sin(theta) * np.cos(phi)

        File.write (str(x) + "\t" + str(y) + "\t" + str(z) + "\n")
    File.close()  
    # Call function to plot the coordinates
    Plotter_3D ("Sphere_Surface_Plot.txt")    

#------------------------------
    
# Finds n random positions with a random r wich follows the exponential distribution.
def Sphere_Distribution (Lambda, n):
    x, y, z, = [],[],[]
    
    File = open ("Spherical_plot.txt", "w")
    for i in range (n):
        r = - Lambda * np.log ((1 - random.uniform(0, 1)))
        theta = np.arccos(1 - (2 * np.random.uniform(0, 1)))
        phi = 2 * np.pi * np.random.uniform(0, 1)

        # Convert from spherical to cartesian coordinates.
        z = r * np.cos(theta)
        y = r * np.sin(theta) * np.sin(phi)
        x = r * np.sin(theta) * np.cos(phi)
        
        # Write results to a file
        File.write (str(x) + "\t" + str(y) + "\t" + str(z) + "\n")
    File.close()    
    Plotter_3D ("Spherical_plot.txt")   

#------------------------------

# Function to find the number of transmitted, reflected and absorbed neutrons.   
def Isotropic_Steps (Material, Sigma_a, Sigma_s, Density, Mass, Thickness, Plot, Number_of_Neutrons, Print, Test_Repeats):
    # Prints the material for the user, if specified
    if Print == True:
        print(Material + ": ")
       
    # Arrays used later
    No_of_Reflections_array = []
    No_of_Transmitions_array = []
    No_of_Absorptions_array = []
    
    # Repeats the experiment over each test
    for j in range(Test_Repeats):
        
        # Initial conditions
        No_of_Transmitions = 0
        No_of_Absorptions = 0
        No_of_Reflections = 0
        
        # Finds what happens to each neutron when it passes the material
        for i in range (Number_of_Neutrons):
            
            # Used to sum the total path distance taken
            total_path = 0
            
            # Calcultaing the mean free paths
            Lambda_s = Mass * (10 ** 24) / (Density * 6.02 * (10 ** 23) * Sigma_s)
            Lambda_a = Mass * (10 ** 24) / (Density * 6.02 * (10 ** 23) * Sigma_a)

            # Initial Conditions:
            is_Absorbed, is_Reflected, is_Transmitted, Will_Be_Absorbed = 0, 0, 0, 0
            x, y, z = [0], [0], [0]

            # 1st Step:
            # Distance is random with exponential distribution
            dx_scatter = - Lambda_s * np.log ((1 - random.uniform(0, 1)))
            dx_absorbed = - Lambda_a * np.log ((1 - random.uniform(0, 1)))
            
            # Decides what to happen to the neutron in the first step, depending on the distance of each outcome
            # Shorter distances imply they occur first
            if dx_scatter < dx_absorbed:
                dx = dx_scatter
            # If not scattered, it is absorbed and so 1 absorption is added to the count
            else:
                dx = dx_absorbed
                is_Absorbed = 1
                if Plot == True:
                    print ("Neutron is Absorbed")
                No_of_Absorptions += 1
                
            # If the neutron travlled all the way through:     
            if dx > Thickness:
                is_Transmitted = 1
                No_of_Transmitions += 1
                
            # No change to the vertical or horizontal dircations after 1st step
            dy, dz = 0, 0
            total_path += dx
            index = 0
            x.append(x[index] + dx), y.append(y[index] + dy), z.append(z[index] + dz)

            # Rest of the steps:
            index = 1
            
            # Loops over all steps. Loop breaks when absorbed, transmitted or reflected
            while is_Absorbed == 0 and is_Transmitted == 0:
                # Path length for the neutron if it is scattered or absorbed
                Scatter_path = - Lambda_s * np.log ((1 - random.uniform(0, 1)))
                Absorption_path = - Lambda_a * np.log ((1 - random.uniform(0, 1)))
                
                # As before, the path that is the shortest occurs first.
                if Scatter_path < Absorption_path:
                    dr = Scatter_path
                elif Scatter_path > Absorption_path:
                    dr = Absorption_path
                    Will_Be_Absorbed = 1
                    
                 # Determining how the neutron will scatter
                theta = np.arccos(1 - (2 * np.random.uniform(0, 1)))
                phi = 2 * np.pi * np.random.uniform(0, 1)

                # Convert from spherical to cartesian coordinates.
                dz = dr * np.cos(theta)
                dy = dr * np.sin(theta) * np.sin(phi)
                dx = dr * np.sin(theta) * np.cos(phi)
                
                # Appending the scatter point to plot later
                x.append(x[index] + dx)
                y.append(y[index] + dy)
                z.append(z[index] + dz)
                
                # Adding this distance travelled to the total path
                total_path += dr

                index = index + 1
                
                # Determining if reflected or transmitted
                if x[index] < 0:
                    if Plot == True:
                        print ("Neutron is reflected")
                    No_of_Reflections += 1
                    is_Reflected = 1
                    break

                if x[index] > Thickness:
                    if Plot == True:
                        print ("Neutron is Transmitted")
                    No_of_Transmitions += 1
                    is_Transmitted = 1
                    break

                if Will_Be_Absorbed == 1:
                    if Plot == True:
                        print ("Neutron is Absorbed")
                    No_of_Absorptions += 1
                    is_Absorbed = 1
                    break
                    
            # Plots the results
            if Plot == True:
                print ("Distance travelled = " + str ( round(total_path, 5)) + "cm")
                fig4 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
                ax = fig4.add_subplot(111, projection='3d')
                ax.plot(x, y, z)
                plt.show()
                
        # Appends how many reflections and transmitions and absorptions occur
        No_of_Reflections_array.append(No_of_Reflections)
        No_of_Transmitions_array.append(No_of_Transmitions)
        No_of_Absorptions_array.append(No_of_Absorptions)
    
    # How many decimal places to show
    D_P = 8
    
    # Finding the mean and error for each number of reflections, tansmitions and absorptions
    Mean_Reflections = round(np.mean(No_of_Reflections_array), D_P)
    Error_Reflections = round( np.std(No_of_Reflections_array, ddof = 1) / (len(No_of_Reflections_array) ** 0.5 ), D_P)
    
    Mean_Transmitions = round(np.mean(No_of_Transmitions_array), D_P)
    Error_Transmitions = round( np.std(No_of_Transmitions_array, ddof = 1) / (len(No_of_Transmitions_array) ** 0.5 ), D_P)
    
    Mean_Absorptions = round(np.mean(No_of_Absorptions_array), D_P)
    Error_Absorptions = round( np.std(No_of_Absorptions_array, ddof = 1) / (len(No_of_Absorptions_array) ** 0.5 ), D_P)
    
    # Prints the results
    if Plot == False and Print == True:
        print ("Number of repeats: " + str(Test_Repeats))
        print ("Number of Neutrons: " + str(Number_of_Neutrons) )           
        print ("Reflected: (" + str(100*Mean_Reflections / Number_of_Neutrons) + " \u00B1 " + str(100*Error_Reflections / Number_of_Neutrons) + ") %" )
        print ("Transmitted: (" + str(100*Mean_Transmitions/Number_of_Neutrons) + " \u00B1 " + str(100*Error_Transmitions / Number_of_Neutrons) + ") %" )
        print ("Absorbed: (" + str(100*Mean_Absorptions/Number_of_Neutrons) + " \u00B1 " + str(100*Error_Absorptions/Number_of_Neutrons) + ") %" ) 
        print ("-----------------------------")
        print ()
    return Number_of_Neutrons, No_of_Reflections, No_of_Transmitions, No_of_Absorptions, Error_Transmitions, Error_Absorptions, Error_Reflections

#------------------------------

# How does error in fraction change with neutron count
def Uncertainty_with_neutrons (repeats, material, Neutron_range):
    # Lists that are returned later
    no_of_neutrons_array = []
    Err_T = []
    Err_A = []
    Err_R = []
    # How many repeates are done
    Test_Repeats = repeats
    
    # Repeats for 8000 divided by Neutron_range times
    for k in range(1, Neutron_range):
        Number_of_Neutrons = math.ceil( k * (8000 / Neutron_range))
        no_of_neutrons_array.append(Number_of_Neutrons)
        m = material
        
        N, No_of_Reflections, No_of_Transmitions, No_of_Absorptions, Error_Transmitions, Error_Absorptions, Error_Reflections = Isotropic_Steps (m.name, m.Sigma_a, m.Sigma_s, m.Density, m.Mass, Thickness, Plot, Number_of_Neutrons, False, Test_Repeats)
        
        # Error in each fraction
        Err_T.append(100 * Error_Transmitions / Number_of_Neutrons)
        Err_A.append(100 * Error_Absorptions / Number_of_Neutrons)
        Err_R.append(100 * Error_Reflections / Number_of_Neutrons)
        
        #print ("Transmition Fraction: " + str(round(Average_Transmition_Fraction, 5)) + " +/- " + str(round(Uncertainty_Transmition_Fraction, 5)))
    
    return [[no_of_neutrons_array, Err_T, Err_A, Err_R], [No_of_Transmitions, No_of_Absorptions, No_of_Reflections]]
    
#------------------------------
    
# Neutrons through a sample of a certain thickness and material.
def Neutrons_Through_Fixed_Tickness (Thickness, Plot, Number_of_Neutrons, Material, Test_Repeats, Print):
    
    # So we do not plot too many graphs
    if Plot == True:
        Number_of_Neutrons = 3
    print ("Calculating for Thickness = " + str(Thickness) + " cm...")
    print()

    # Simulates n neutrons for each material and outputs how many were absorbed, reflected, and transmitted.
    i = Material
    N, No_of_Reflections, No_of_Transmitions, No_of_Absorptions, Error_Transmitions, Error_Absorptions, Error_Reflections = Isotropic_Steps (i.name, i.Sigma_a, i.Sigma_s, i.Density, i.Mass, Thickness, Plot, Number_of_Neutrons, Print, Test_Repeats)

    return  No_of_Transmitions, Error_Transmitions

#------------------------------

def Neutrons_Through_varying_Thickness (Number_of_Neutrons, Test_Repeats, Plot, Thickness_Range, Print, Material):    
    thicknesses = []
    transmitions = []
    err_transmitions = []
    
    # Finds the transmission fraction for a range of thicknesses
    print("Calculating transmission probabilites with varying thicknesses for " + str(Material.name))
    for i in range(1,Thickness_Range):
        Thickness = 5 * i
        No_of_Transmitions, Error_Transmitions = Neutrons_Through_Fixed_Tickness (Thickness, Plot, Number_of_Neutrons, Material, Test_Repeats, Print)
        thicknesses.append(Thickness)
        transmitions.append(No_of_Transmitions)
        err_transmitions.append(Error_Transmitions)
    
    transmitions, err_transmitions = np.array(transmitions), np.array(err_transmitions)
    transmitions_prob = np.array(transmitions / Number_of_Neutrons)
    err_transmitions_prob = np.array(err_transmitions / Number_of_Neutrons)
    
    fig4 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')    
    plt.errorbar(thicknesses, transmitions_prob, err_transmitions_prob)
    plt.xlabel("Thickness (cm)")
    plt.ylabel("Transmission Probability")
    plt.show()
    
    return thicknesses, transmitions, err_transmitions

#-----------------------------------

def Neutrons_Through_varying_Thickness_analysis(Number_of_Neutrons, Test_Repeats, Plot, Thickness_Range, Print, Materials):
    for Material in Materials:
        thicknesses, transmitions, err_transmitions = Neutrons_Through_varying_Thickness (Number_of_Neutrons, Test_Repeats, Plot, Thickness_Range, Print, Material)
        
        # Turning lists into numpy arrays:
        thicknesses, transmitions, err_transmitions = np.array(thicknesses), np.array(transmitions), np.array(err_transmitions)
        
        # Deletes points where there is no frequency
        index = 0
        for i in transmitions:
            if transmitions[index] == 0:
                transmitions = np.delete(transmitions, index)
                thicknesses = np.delete(thicknesses, index)
                err_transmitions = np.delete(err_transmitions, index)
            else:
                index += 1
        
        # Finds the log of the frequencies and associated error
        Log_transmitions = np.log(transmitions/Number_of_Neutrons)
        y_err = np.array(err_transmitions)   
        Log_tran_err = np.array (y_err / transmitions)

        weight = np.array (1 / Log_tran_err)
        
        print()
        print("thicknesses")
        for i in thicknesses:
            print(str(i))
            
        print()
        print("Log transmitions")   
        for i in Log_transmitions:
            print(str(i))
        
        print()
        print("Error in log transmitions")
        for i in Log_tran_err:
            print(str(i))

        coef, covr = np.polyfit(thicknesses, Log_transmitions, 1, cov=True, w = weight)
        y_fit = np.polyval(coef, thicknesses)

        fig5 = plt.figure(figsize=(16,12), dpi=60, facecolor='w', edgecolor='k')
        plt.errorbar(thicknesses, Log_transmitions, Log_tran_err, linestyle = " ", marker = "x")
        
        plt.plot(thicknesses, y_fit)
        plt.xlabel("Thickness (cm)")
        plt.ylabel("Probability of Transmission")
        plt.show()

        #Shows user results
        print ("Results for " + str(Material.name))
        print ("Gradient: " + str(coef[0]) + " +/- " + str(round(covr[0][0] ** 0.5, 5)))
        print ("Intercept: " + str(coef[1]) + " +/- " + str(round(covr[1][1] ** 0.5, 5)))
        print ("Calculated Lambda: " + str(round(-1 / coef[0], 5)) + " +/- " + str(round((covr[0][0] ** 0.5 ) / (coef[0] ** 2), 5)))
        print ("Reduced Chi Squared: " + str(round(Chi(thicknesses,Log_transmitions,Log_tran_err,y_fit), 5)))
        print()  
        
#------------------------------
# Main Program
#------------------------------

%matplotlib notebook

# Figure font formatting
plt.rcParams.update({'font.size': 22})

# Properties of the materials:
#----------------------------------------------
class Water:
    name = "Water"    # Name of the material
    Sigma_a = 0.6652  # Absorption corss section
    Sigma_s = 103.0   # Scatter cross section
    Density = 1       # Desnity g/cm
    Mass = 18.01528   # Molar mass
    color = "b"
    
class Lead:
    name = "Lead"
    Sigma_a = 0.158
    Sigma_s = 11.221
    Density = 11.35
    Mass = 207.2
    color = "r"
    
class Graphite:
    name = "Graphite"
    Sigma_a = 0.0045
    Sigma_s = 4.74
    Density = 1.67
    Mass = 12.01
    color = "g"
#----------------------------------------------
Materials = [Water, Lead, Graphite]      #The materials being tested


# Observe randssp scatter.
randssp_plot ()

# Check that random is uniform:
Random_Check ()

# Create a random 3D scatter
Filename = "3D_Numbers.txt"
n = 100
Number_Range = (10)
Number_Generator_3D (Number_Range, n) 
Plotter_3D (Filename) 

# Plot exponential
Lambda = 45 # cm
Number_of_Particles = 1000000
Exponential_Distribution (Lambda, Number_of_Particles)

# Plot scatter on the surface of a sphere
r = 10
n = 2000
Sphere_Surface (r, n)

# Plot a scatter with random direction but distance from the origin determined by exponential distribution
n = 500
Sphere_Distribution (Lambda, n)

#------------------------
# Single thickness Tests:

# Probabilities of transmission etc for each material
Materials = [Water, Lead, Graphite]
Thickness = 10    # Thickness of material
Plot = False      # Should the path be plotted?
Number_of_Neutrons = 5000    # Number of neutrons being simulated
Test_Repeats = 10       # How many repeats are being measured
Print = True     # Should the results be printed (Used Flase in when calling from other functions)
if Plot == True:
    Number_of_Neutrons = 3
for Material in Materials:
    Neutrons_Through_Fixed_Tickness (Thickness, Plot, Number_of_Neutrons, Material, Test_Repeats, Print)


# How the uncertianty in tranmission changes with number of neutrons
Neutron_range = 20
repeats = 10
# Repeat over all Materials
index = 0
Data = []
for Material in Materials:
    Data.append(Uncertainty_with_neutrons (repeats, Material, Neutron_range))
    
    no_of_neutrons_array = np.array(Data[index][0][0])
    
    Err_T = Data[index][0][1]
    Err_A = Data[index][0][2]
    Err_R = Data[index][0][3]
    
    if Material.name == "Water":
        T = 0.06
        A = 25
        R = 75
    elif Material.name == "Lead":
        T = 24.7
        A = 15.7
        R = 59.6
    elif Material.name == "Graphite":
        T = 29.4
        A = 13.9
        R = 69.2
        
    fig7 = plt.figure(figsize=(16,12), dpi=60, facecolor='w')
    plt.plot(no_of_neutrons_array, Err_T, color = "r", marker = "o", linestyle = "-")
    plt.plot(no_of_neutrons_array, Err_A, color = "b", marker = "o", linestyle = "-")
    plt.plot(no_of_neutrons_array, Err_R, color = "g", marker = "o", linestyle = "-")
    
    # Legends format:
    Label_1 = "Transmission "
    Label_2 = "Absorptions "
    Label_3 = "Reflections "
    
    E1 = mpatches.Patch(color='r', label=Label_1)
    E2 = mpatches.Patch(color='b', label=Label_2)
    E3 = mpatches.Patch(color='g', label=Label_3)
    plt.legend(handles=[E1, E2, E3])
    
    plt.xlabel("Number of Neutrons")
    plt.ylabel("Uncertainty in Probability (%)")
    plt.show()
    index += 1
    
#----------------------------
# Neutrons through varying thickness
Materials = [Lead, Graphite]
Thickness_Range = 20
Number_of_Neutrons = 5000
Test_Repeats = 5
Plot = False
Print = False

Neutrons_Through_varying_Thickness_analysis(Number_of_Neutrons, Test_Repeats, Plot, Thickness_Range, Print, Materials)
