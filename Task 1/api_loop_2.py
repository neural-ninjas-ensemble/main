import pandas as pd
import numpy as np
import os
import torch
from api_simulation import *
from taskdataset import *
from tqdm import tqdm
from datetime import datetime
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

START = 250

def load_img_from_label(label):
    return df[df['label'] == label]['img'].values[0]

def ret_embeding_from_label(df, label):

    PNG_TEMPORARY_FILE_NAME = "temp2.png"

    load_img_from_label(df, label).save(PNG_TEMPORARY_FILE_NAME, "PNG")

    encoded_emb = model_stealing(PNG_TEMPORARY_FILE_NAME)

    os.remove(PNG_TEMPORARY_FILE_NAME)

    # return [1]
    return encoded_emb

def load_img_from_ids(df, img_id):
    return df[df['id'] == img_id]['img'].values[0]

def ret_embeding_from_id(df, img_id):

    PNG_TEMPORARY_FILE_NAME = "temp2.png"

    load_img_from_ids(df, img_id).save(PNG_TEMPORARY_FILE_NAME, "PNG")

    encoded_emb = model_stealing(PNG_TEMPORARY_FILE_NAME)
    os.remove(PNG_TEMPORARY_FILE_NAME)

    # return [1]
    return encoded_emb

def run_old(output_dir):
  control_id = 73838
  ID_ORDER_FILE_PATH = 'data/kolejnosc_id_v2.csv'
  ids = pd.read_csv(ID_ORDER_FILE_PATH).drop(columns = ['Unnamed: 0']).squeeze().values
  control_embeding = ret_embeding_from_id(df,control_id)

  target_embedings = []
  checkpoint_embedings = [control_embeding]
  iter = 1000

  ids = ids[START:]

  for id in tqdm(ids[1000:],total=len(ids)):
      if id != control_id:
          #embedings = []
          iter +=1

          if iter % 100 == 0:
              try:
                  checkpoint_embedings.append(ret_embeding_from_id(df,control_id))
              except Exception:
                  print(f"Error on id {control_id}")
                  continue

          try:

            target_embedings.append((id,ret_embeding_from_id(df,id)))

          except Exception:
              print(f"Error on id {id}")
              continue

          # sdev = np.std(modified_control_embeding - control_embeding)
          # number_of_pic = 10
          # for _ in range(number_of_pic):
          #     embedings.append((id,ret_embeding_from_id(df,id)))
          # final_embeding = pd.DataFrame(embedings).mean()
          # final_embedings.append((id,final_embeding))

          #all_embedings = all_embedings + embedings



          if iter % 250 == 0:
              pd.DataFrame(checkpoint_embedings).to_csv(f"./{output_dir}/chekpoint_embeding_{iter}_START_{START}.csv", index=False)
              pd.DataFrame(target_embedings).to_csv(f"./{output_dir}/target_embeding_{iter}_START_{START}.csv", index=False)

  pd.DataFrame(checkpoint_embedings).to_csv(f"./{output_dir}/final_checkpoint_embedings_START_{START}.csv", index=False)
  pd.DataFrame(target_embedings).to_csv(f"./{output_dir}/all_embeding_START_{START}.csv", index=False)

if __name__ == '__main__':
    now = datetime.now()

    output_dir = f"output_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_START_{START}"
    os.mkdir(output_dir)
    run_old(output_dir)