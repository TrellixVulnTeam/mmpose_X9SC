import numpy as np
import torch
from tools import read_pkl
from ADModel import ADModel
if __name__=="__main__":
    data = read_pkl("../train_v4.pkl")
    input = torch.tensor(data["keypoints"][200:210])

    input = input.cuda()
    label = data["labels"][200:210]
    model = torch.load("model_latest_test_acc_0.9888682745825603.pth")
    out = model(input)
    print(out,label)


