import pandas as pd
import numpy as np
import os
import torch
from api_simulation import *
from taskdataset import *
from tqdm import tqdm
dataset = torch.load("data/ModelStealingPub.pt")

df = pd.DataFrame()

ids = []
imgs = []
labels = []

for i in range(len(dataset)):
    ids.append(dataset[i][0])
    imgs.append(dataset[i][1])
    labels.append(dataset[i][2])

df["id"] = ids
df["img"] = imgs
df["label"] = labels


def load_img_from_ids(df, img_id):
    return df[df['id'] == img_id]['img'].values[0]

def ret_embeding_from_id(df, img_id):

    PNG_TEMPORARY_FILE_NAME = "temp.png"

    load_img_from_ids(df, img_id).save(PNG_TEMPORARY_FILE_NAME, "PNG")

    # encoded_emb = model_stealing(PNG_TEMPORARY_FILE_NAME)

    os.remove(PNG_TEMPORARY_FILE_NAME)

    return [1]
#     return encoded_emb

control_id = 73838
ID_ORDER_FILE_PATH = 'data/kolejnosc_id_v1.csv'
ids = pd.read_csv(ID_ORDER_FILE_PATH).drop(columns = ['Unnamed: 0']).squeeze().values
control_embeding = ret_embeding_from_id(df,control_id)

target_embedings = []
checkpoint_embedings = [(control_id,control_embeding)]
iter = 0
for id in tqdm(ids,total=len(ids)):
    if id != control_id:
        #embedings = []
        iter +=1
        target_embedings.append((id,ret_embeding_from_id(df,id)))
        checkpoint_embedings.append(ret_embeding_from_id(df,control_id))

        # sdev = np.std(modified_control_embeding - control_embeding)
        # number_of_pic = 10
        # for _ in range(number_of_pic):
        #     embedings.append((id,ret_embeding_from_id(df,id)))
        # final_embeding = pd.DataFrame(embedings).mean()
        # final_embedings.append((id,final_embeding))

        #all_embedings = all_embedings + embedings



        if iter % 250 == 0:
            pd.DataFrame(checkpoint_embedings).to_csv(f'./data/chekpoint_embeding_{iter}.csv', index=False)
            pd.DataFrame(target_embedings).to_csv(f'./data/target_embeding_{iter}.csv', index=False)

pd.DataFrame(checkpoint_embedings).to_csv('./data/final_checkpoint_embedings.csv', index=False)
pd.DataFrame(target_embedings).to_csv('./data/all_embeding.csv', index=False)