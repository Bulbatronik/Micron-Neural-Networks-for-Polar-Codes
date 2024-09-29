from pathlib import Path
import torch

def load_dataset(file_path, fb_name, fer_name):
    """
    Load data from file_path/fb_name and file_path/fer_name
    """
    # Load data corresponnding to the bit vectors
    with open(Path(file_path)/fb_name) as f: 
        content = f.readlines()
    dim = len(content[0].split())
    n = len(content)
    data = torch.zeros(n, dim)
    
    for i in range(n):
        vals = content[i].split()
        for j in range(dim):
            data[i,j] = float(vals[j])

    # Load data corresponding FERs        
    with open(Path(file_path)/fer_name) as f:
        content = f.readlines()
    assert(len(content) == n)
    target = torch.zeros(n)
    for i in range(n):
        target[i] = float(content[i])
    return data, target