import math
import os.path as osp
import os
import cv2
import pandas as pd
import pickle
import numpy as np
#1.获取文件名列表
def get_video_name(folder_dir):
    video_names = os.listdir(folder_dir)
    video_names = [osp.join(folder_dir, vid_n) for vid_n in video_names]
    return video_names

def process_img_skeleton_data(img_data):  #return everyone skeleton distance on image
    peoples_dis_matri = []
    for peopel_data in img_data:
        kp = peopel_data["keypoints"][:11]
        dis_matri = []
        for kp1 in kp:
            dis = []
            for kp2 in kp:
                x1, y1, x2, y2 = kp1[0], kp1[1], kp2[0], kp2[1]
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dis.append(d)
            dis_matri.append(dis)
        peoples_dis_matri.append(dis_matri)
    return peoples_dis_matri

def process_raw_train_skeleton():
    with open(r"C:\Users\radishi\Desktop\data\process_data\train.pkl","rb") as fo:
        skeleton_data = pickle.load(fo,encoding="bytes")

    for i,video_data in enumerate(skeleton_data):
        video_data = [ frame_data[:11 ] for frame_data in video_data["keypoints"]]
        #calculate skeleton distance
        video_dis_matri = []
        for frame_data in video_data:
            dis_matri = []
            for kp1 in frame_data:
                dis = []
                for kp2 in frame_data:
                    x1,y1,x2,y2 = kp1[0],kp1[1],kp2[0],kp2[1]
                    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    dis.append(d)
                dis_matri.append(dis)
            video_dis_matri.append(dis_matri)
        skeleton_data[i]["keypoints"] = video_dis_matri
    print(skeleton_data[0])
    with open(r"train_v3.pkl","wb") as fo:
        pickle.dump(skeleton_data,fo)

def data_match_lable():
    dis_data = read_pkl("train_v3.pkl")
    keypoints = []
    labels = []
    for video_data in dis_data:
        vid_label = int(video_data["label"])
        keypoint_num = len(video_data["keypoints"])
        #labels_ = np.zeros((keypoint_num, 4),dtype=int)  #generate one hot encoding
        # for label in labels_:
        #     label[vid_label-1] = 1
        #     labels.append(label)
        for kp in video_data["keypoints"]:
            labels.append(vid_label-1)
            keypoints.append([kp])
    #save keypoints labels
    save_pkl("train_v4.pkl",dict(labels=labels,keypoints=keypoints))
    return keypoints,labels

def get_action_class():
    label_name_list = {}
    df = pd.read_csv("label_list.txt", sep=" ")
    for i in range(len(df)):
        label_name_list.update({df.loc[i]["num"]: df.loc[i]["class"]})
    return label_name_list

def vis_action_label(img,bbox,labels):
    for box,label in zip(bbox,labels):
        x1,y1 = box[0],box[1]
        cv2.putText(img,label,(x1+20,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)


def read_pkl(file_dir):
    with open(file_dir,"rb") as fo:
        data = pickle.load(fo,encoding="bytes")
    return data

def save_pkl(file_name,data):
    with open(file_name,"wb") as fo:
        pickle.dump(data,fo)

if __name__=="__main__":
    data_match_lable()
    data = read_pkl("train_v4.pkl")
    print(np.array(data["labels"]).shape)







