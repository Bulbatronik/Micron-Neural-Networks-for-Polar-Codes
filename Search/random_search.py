import time
import torch
from .utils import write_txt


def RandomSearch(model, x, preprocessor, max_time, best_fer):
    print("Best FER (init):", best_fer)
    total_tested = 0
    step = 1000
    start_time = time.time()

    while True:
        # Create a batch of inputs, all initialized to 1
        inp = torch.ones(step, x.shape[1]).to(model.device)
        
        # Randomly set half of each input's elements to -1
        for i in range(inp.shape[0]):
            inp[i, torch.randperm(inp.shape[1])[:inp.shape[1] // 2]] = -1
        
        # Run the model on the batch of inputs
        with torch.no_grad():
            output = model(inp) 
        
        # Find the input with the best (lowest) FER in this batch
        index = torch.argmin(output)
        
        # If we've found a new best FER, update and save it
        if output[index] < best_fer:
            best_fer = output[index]
            data_, output_ = preprocessor.inverse_transform(inp[index].reshape(1, -1), output[index])
            write_txt(data_, output_, "rnd")
            print(f"New best FER: {float(output[index])} ({float(output_):E}), achieved at iteration {total_tested}")
        
        # Check if we've exceeded the maximum time
        if time.time() - start_time > max_time:
            break
        
        total_tested += step

    return best_fer  # This line is implied but not explicitly in the original code