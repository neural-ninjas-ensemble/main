import pandas as pd
import numpy as np
import os


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
ID_ORDER_FILE_PATH = '/content/drive/MyDrive/kolejnosc_id_v1.csv'
ids = pd.read_csv(ID_ORDER_FILE_PATH).drop(columns = ['Unnamed: 0']).squeeze().values
control_embeding = ret_embeding_from_id(df,control_id)

all_embedings = []
final_embedings = []
for id in ids:
    if id == control_id:
        pass
    else:
        embedings = []
        embedings.append((id,ret_embeding_from_id(df,id)))
        modified_control_embeding = ret_embeding_from_id(df,control_id)
        sdev = np.std(modified_control_embeding - control_embeding )
        number_of_pic = 10
        for _ in number_of_pic:
            embedings.append((id,ret_embeding_from_id(df,id)))
        final_embeding = pd.DataFrame(embedings).mean()
        final_embedings.append((id,final_embeding))
        all_embedings = all_embedings + embedings
   
pd.DataFrame(final_embedings).to_csv('./final_embedings.csv')
pd.DataFrame(all_embedings).to_csv('./all_embeding.csv')