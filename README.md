<h1 align="center">Neural Networks for Polar Codes </h1>

<h1 align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcToBLl2TfiS111dcSbabRW8EwohsJDhMvQoMg&s" alt="Nokia Logo" width="300">
</h1>


*The project was developed with Vasiliki Batsari, Kiarash Rezaei and Francesco Ruan in collaboration with [**Micron**](https://www.micron.com/)*.

## Overview

The problem formulation and the assistance for the project was provided by **Micron** within the initiative *Adopt a course* between the companies and Politecnico di Milano.

The goal of the project is to predict the efficiency of [`Polar Codes`](https://en.wikipedia.org/wiki/Polar_code_(coding_theory)#:~:text=In%20information%20theory%2C%20polar%20codes,channel%20into%20virtual%20outer%20channels.) using `Deep Neural Networks`. Based on the trained model, the objective is to leverage its parameters in order to produce a configuration of the polar codes with high efficiency. The evaluation the the efficiency is carried out by analyzing the performance of the canstalation for a different set of `Signal to Noise Ration (SNR)`. For further details on polar codes refer to [this paper](Materials/PolarCodes.pdf).

The results are simulated for a polar codes with `64 information bytes`. We also provide the description of data simulation for other configurations using [`Aff3ct Toolbox`](Materials/Toolbox.pdf). Moreover, we provide the search of the optimal configuration of bits using `Random Search` and `Projected Gradient Descent`.


## Approach
### 1. Dataset Generation
To generate the initial dataset, we utilized [`Aff3ct`](Materials/Toolbox.pdf). This simulator provides a `Frame Error Rate (FER)` and `Bit Error Rate (BER)` for a sequence of frozen and information bits' positions and SNR using `Monte Carlo (MC)` simulation. 

We started from a frozen set of bits for `Gaussian Approximation (GA)` we applied random permutations to the bits to get the required number of samples (`~15K samples` of 1024 bits/64 bytes of payload). We automated the data acquisition process using `bash` scripting. The results are saved in two `txt` files, where **fb** represents the code structure and **fer** - corresponding metric. 

When the task is to generate the code configuretion, we save the proposed results into `txt` files with **rnd** and **pgd** being the identifiers of the generating algorithm.


### 2. Data Preprocessing
First, we discard the bits with zero variance along the sample dimensions, as these features an uninformative. We further standardize the bits by converting them to +-1, while the **FER** if converted to linear units. The converter also capable of converting the data backwards by including the redundant bits and leveraging stored means per bit, and taking a proper logarithm from the labels. Later the data is converted to a `torch` dataset and batched using `DataLoader`.

### 3. Simulations
We simulate the multi layer perceptron (`MLP`)-based architecture. The controllable parameters are the following:
- Number of Training Epochs
- Hidden Dimension Size
- Depth of The Network
- Frequency of Skip-connections

The model is evaluated in terms of `Inflation of Error (IOE)`, which indicates the percentage of how inaccurate the output is with respect to the true value.

We provide a broad analysis of how the parameters impact the performance of the network on the final output. The figures and be found in the [report](Materials/Presentation.pdf).

### 4 Optimal Input Generation
In order to predict an optimal frozen bit set two following methods were used: `Projected Gradient Descent` and `Random Search` algorithms. The goal for both of them is to find the input such that the output of the model is maximized.

**Random Search (RS)** at each time step inputs `1000` random string of bits (maintaining R = ½) and produces corresponding `FERs`. In case the model predicts the `FER` smaller then the best sample in the dataset, the input iis stored in the file. This process is repeated for a fixed amount of time.

`Projected Gradient Descent (PGD)` with frozen network parameters attempts to optimize the input by minimizing the output using `Stochastic Gradient Descent (SGD)` and projecting the input on the discrete. The procedure is repeated for a number of iterations, after which the input configurations better then the best in the dataset (if any) are stored.


### 4. Results
The final results are presented [this file](Materials/Presentation.pdf). The results were presented online in front of **Micron** engineers and received the maximum grade for the project. 


## References
Léonardon, Mathieu & Gripon, Vincent. (2021). Using Deep Neural Networks to Predict and Improve the Performance of Polar Codes. 1-5. 10.

Nachmani, Eliya & Marciano, Elad & Lugosch, Loren & Gross, Warren & Burshtein, David & Be'ery, Yair. (2017). Deep Learning Methods for Improved Decoding of Linear Codes. IEEE Journal of Selected Topics in Signal Processing. 12. 10.

Huang, Lingchen & Zhang, Huazi & Li, Rong & Ge, Yiqun & Wang, Jun. (2019). AI Coding: Learning to Construct Error Correction Codes. 