<h1 align="center">Neural Networks for Polar Codes </h1>

<h1 align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCuZ8pVmHRM0ebEVWpbJvhTimaJwjmJYxnYA&s" alt="Nokia Logo" width="300">
</h1>


*The project was developed with Vasiliki Batsari, Kiarash Rezaei and Francesco Ruan in collaboration with [**Micron**](https://www.micron.com/)*.

## Overview

The problem formulation and the assistance for the project was provided by **Micron** within the initiative *Adopt a course* between the companies and Politecnico di Milano.


The goal of the project is to predict the efficiency of [`Polar Codes`](https://en.wikipedia.org/wiki/Polar_code_(coding_theory)#:~:text=In%20information%20theory%2C%20polar%20codes,channel%20into%20virtual%20outer%20channels.) using `Deep Neural Networks`. Based on the trained model, the objective is to leverage its parameters in order to produce a configuration of the polar codes with high efficiency. The evaluation the the efficiency is carried out by analyzing the performance of the canstalation for a different set of `Signal to Noise Ration (SNR)`. For further details on polar codes refer to [this paper](Materials/PolarCodes.pdf).

The results are simulated for a polar codes with `64 information bytes`. We also provide the description of data simulation for other configurations using [`Aff3ct Toolbox`](Materials/Toolbox.pdf). Moreover, we provide the search of the optimal configuration of bits using `Random Search` and `Projected Gradient Descent`.


## Approach
### 1. Dataset Generation
To generate the initial dataset, we utilized [`Aff3ct`](Materials/Toolbox.pdf). This simulator provides a `Frame Error Rate (FER)` and `Bit Error Rate (BER)` for a sequence of frozen and information bits' positions and SNR using `Monte Carlo (MC)` simulation. 

We started from a frozen set of bits for `Gaussian Approximation (GA)` we applied random permutations to the bits to get the required number of samples (`~15K samples` of 1024 bits/64 bytes of payload). We automated the data acquisition process using `bash` scripting. 


### 2. Data Preprocessing
First, our approach was to solve the problem directly using the `optimization` techniques provided in [this paper](materials/Designing_Operating_and_Reoptimizing_Elastic_Optical_Networks.pdf) and adapted to include the cost of regenerations. The solution was feasible for a network with 10 links and 100 demands in a considerable amount of time, which was infeasible for a larger network/traffic due to the computation constraints. After the unsuccessful attempt, we decided to use `heuristic methods`, as they are, generally, faster and more scalable. 

#### 2.1 Mixed Integer Linear Problem (MILP)

We start by formulating the problem for `soft spectrum partitioning` as a `MILP`. Our goal is to minimize the sum of the total blocked traffic plus the cost of the regenerations. In order to solve the optimization problem, we used [`Gurobipy`](https://support.gurobi.com/hc/en-us). By setting up the constraints and a set of variables, we have not been able to solve the problem due to its large scale even without the constraints on the hard partitioning.  

#### 2.2 First Fit Approach
The first heuristic we applied was the `First Fit method`. The basic idea behind it is to compute *k* shortest paths between a pair of nodes. Then for each demand between the nodes, we try to route it using the first available shortest path. This approach is suboptimal, particularly for a small number of *k*, but it is extremely fast in comparison to the other algorithms.

#### 2.3 Genetic Algorithm
The metaheuristic algorithm is inspired by the process of natural selection that belongs to the larger class of **evolutionary algorithms (EA)**. One candidate solution in `Genetic Algorithm` is called **Chromosome**. One value of the Chromosome is called **Gene**. All the calculated Chromosomes are 
called Population. Taking the best candidate populations as **Parents**, then taking the best offsprings and make them **Parents**, and so one until the best solution is found.

In our case, we use [`jMetalPy`](https://github.com/jMetal/jMetalPy) library due to its simplicity and support of a large variety of metaheuristic algorithms. Our goal is to optimize thee total spectrum allocation while using as a binary variable if a specific demand was routed successfully or not while satisfying the constraints.


### 3. Simulation 
We perform the simulation over each topology for both algorithms using both partitioning. To analyze the `hard spectrum partitioning` we define a hyperparameter, which controls the percentage of spectrum allocated to 75GHz channels. We continuously increase the number of demands by 50 until the blocking probability exceeds 1%. Since the traffic for each number of demands is random, we perform multiple `Monte Carlo simulations` and record the mean value of the cost and blocking probability with confidence intervals. 

### 4. Results
The final results are presented [this file](materials/Final%20Presentation%20Project%208.pdf). The presentation was held at the office of Nokia and received the maximum grade for the project. 


## Miscellaneous
The folder `miscellaneous` includes the homework/lab assignments during the course `Communication Network Design` done using [`Net2Plan`](https://www.net2plan.com/) software *with Esat Ince*.
