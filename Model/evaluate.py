import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def stats(model, epochs, train_loader, val_loader, processor, max_time, num_runs=1):
    mean_results, worst_results = [], []
    run = 0
    start_time = time.time()
    while True:
        if int(time.time() - start_time) >- max_time or run == num_runs: 
            break     
        run += 1
        print("Run: ", run) 
        model.fit(train_loader, epochs)
        results = model.test(val_loader)
        mean, worst = _IOE(*results, processor) # outputs mean and worst IOE
        mean_results.append(mean)
        worst_results.append(worst)
    print("DONE")
    return [np.mean(mean_results), np.mean(worst_results), _confidence(mean_results), _confidence(worst_results), run] 


def visualize(parameter, values, results):
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Plot the mean performance
    plt.subplot(2, 1, 1) 
    plt.plot(values, results[:, 0]) # mean IOA
    plt.fill_between(values, results[:, 0] - results[:, 2], 
                     results[:, 0] + results[:, 2], color='b', alpha=.1)
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Mean validation IOE")

    # Plot the worst performance
    plt.subplot(2, 1, 2)
    plt.plot(values, results[:, 1]) # worst IOA
    plt.fill_between(values, results[:, 1] - results[:, 3], 
                     results[:, 1] + results[:, 3], color='r', alpha=.1)
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Worst validation IOE")

    plt.title(f"Validation Loss for different {parameter}")
    plt.tight_layout()
    plt.show()


def _IOE(y_true, y_hat, processor):
    """Compute the inflation of error (IOE)"""
    y_hat = torch.cat(y_hat, dim = 0).cpu() 
    y_hat = processor(y_hat) # to linear
    
    mean_ratio, worst_ratio = 0, 0
    for i in range(y_hat.shape[0]):
        ratio = max(y_hat[i] / y_true[i], y_true[i] / y_hat[i]) - 1 
        mean_ratio += ratio
        worst_ratio = max(worst_ratio, ratio)
    mean_ratio /= y_hat.shape[0]
    return mean_ratio, worst_ratio


def _confidence(scores): 
    return np.std(scores) * 1.96 / math.sqrt(len(scores)) 
