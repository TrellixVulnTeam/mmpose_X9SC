from pose_estimination import parser_top_down_args,init_top_down_model,top_down_img
import abc
import cv2

if __name__=="__main__":
    det_model, pose_model = init_top_down_model()
    img = cv2.imread("123.jpg")
    img,pose_results = top_down_img(img,det_model,pose_model)
    cv2.imshow("img",img)


