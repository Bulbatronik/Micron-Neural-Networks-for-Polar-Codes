import torch
from .utils import write_txt


def ProjectedGradientDescent(model, x, preprocessor, iterations, best_fer):
    print("Best FER (init):", float(best_fer))
    device = model.parameters().__next__().device

    # Disable gradient computation for the model
    model.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False
    
    # Initialize data with ones, shape based on input x
    data = torch.ones(1, x.shape[1]).to(device)
    
    # Randomly set half of the elements to -1
    for i in range(data.shape[0]):
        data[i, torch.randperm(data.shape[1])[:data.shape[1] // 2]] = -1 
    
    # Create a copy of data for optimization
    data_approx = data.clone().to(device)
    
    # Enable gradient computation for data and data_approx
    data.requires_grad = True
    data_approx.requires_grad = True
    
    # Set up optimizer
    optimizer = torch.optim.SGD([data, data_approx], lr=0.01)
    
    # Main optimization loop
    for it in range(iterations):
        optimizer.zero_grad()
        
        # Quantize data_approx to -1 or 1 based on median
        with torch.no_grad():
            quatntzer = torch.median(data_approx, dim=1)[0]
            for i in range(data.shape[0]): 
                data[i][torch.where(data_approx[i] <= quatntzer[i])] = -1
                data[i][torch.where(data_approx[i] > quatntzer[i])] = 1
        
        # Forward pass
        output = model(data)  # predicts FER (Frame Error Rate?)
        
        # Check if we've found a new best FER
        with torch.no_grad():
            obtained_fer = output.item()
            if obtained_fer < best_fer:                
                best_fer = obtained_fer
                data_, output_ = preprocessor.inverse_transform(data, output)
                write_txt(data_, output_, "pgd")
                print(f"New best FER: {float(obtained_fer)} ({float(output_):E}), achieved at iteration {it}")
        
        # Backward pass
        output.backward()
        
        # Apply gradients to data_approx
        data_approx.grad = data.grad
        
        # Update parameters
        optimizer.step()