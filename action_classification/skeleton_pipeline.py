from pose_estimination import parser_top_down_args,init_top_down_model,top_down_img
import cv2
from tools import process_img_skeleton_data, get_action_class, vis_action_label, model_process
import torch
from ADModel import ADModel

if __name__=="__main__":
    #get skeleton data
    img = cv2.imread("123.jpg")
    det_model, pose_model = init_top_down_model()
    img,pose_results = top_down_img(img,det_model,pose_model)
    model = torch.load("model_latest_test_acc_0.9888682745825603.pth")
    #inpuut model
    labels = model_process(pose_results,model)
    # visualize skeleton and bbox
    vis_action_label(img,labels)
    cv2.imshow(img)