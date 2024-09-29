import os

def write_txt(inputs, outputs, method):
    os.makedirs("/results/", exist_ok=True)
    with open(f"results/proposed_fb_{method}.txt", "w") as f:
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                f.write(str(int(inputs[i,j].item())) + " ")
            f.write("\n")
    with open(f"results/proposed_fer_{method}.txt", "w") as f:
        for i in range(outputs.shape[0]):
            f.write(str(outputs[i].item()) + "\n")