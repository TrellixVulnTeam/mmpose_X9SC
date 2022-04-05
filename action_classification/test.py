import numpy as np
import torch
from tools import read_pkl
from ADModel import ADModel
import pandas as pd
if __name__=="__main__":
    # data = read_pkl("../train_v4.pkl")
    # input = torch.tensor(data["keypoints"][200:210])
    #
    # input = input.cuda()
    # label = data["labels"][200:210]
    # model = torch.load("model_latest_test_acc_0.9888682745825603.pth")
    # out = model(input)
    # out = out.cpu().detach().numpy().argmax(1)
    # print(out)

    # label_list = {}
    # df = pd.read_csv("label_list.txt",sep=" ")
    # for i in range(len(df)):
    #     label_list.update({df.loc[i]["num"]:df.loc[i]["class"]})

    out = np.array([[0.3,0.4,0.5],[0.2,0.3,0.8],[0.1,0.4,0.6]])
    index = np.array(out).argmax(1)
    thr = [res[res.argmax()] for res in out]
    print(thr)


