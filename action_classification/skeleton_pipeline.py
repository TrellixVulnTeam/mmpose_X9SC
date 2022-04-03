from pose_estimination import parser_top_down_args,init_top_down_model,top_down_img
import cv2
from tools import process_img_skeleton_data,get_action_class,vis_action_label
import torch
from ADModel import ADModel
if __name__=="__main__":
    #get skeleton data
    det_model, pose_model = init_top_down_model()
    img = cv2.imread("123.jpg")
    img,pose_results = top_down_img(img,det_model,pose_model)
    # caculate distance
    dis_data = process_img_skeleton_data(pose_results)
    #inpuut model
    dis_data = torch.tensor(dis_data).reshape((-1,1,11,11))
    model = torch.load("model_latest_test_acc_0.9888682745825603.pth")
    out = model(dis_data)
    out = out.cpu().detach().numpy().argmax(1)
    #visualize skeleton and bbox
    action_class = get_action_class()
    labels = [action_class[i] for i in out]
    vis_action_label(img,pose_results["bbox"],labels)
    cv2.imshow(img)