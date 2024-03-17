'''
This script sends queries to get image embeddings for different users and saves it to a CSV file.
'''
import time
import json
from copy import deepcopy

import torch
import pandas as pd
from typing import List, Tuple

from taskdataset import TaskDataset
from request import sybil, sybil_submit, sybil_reset


def make_user_queries(path_to_data: str, query_limit: int=2000, calib_len: int=1000) -> List[Tuple[List[int], List[int]]]:
    '''
    Given a data file and query limit per endpoint prepare lists of image ids
    for users.

    :param path_to_data: path to a pt file with a torch dataset
    :param query_limit: limit of queries per user
    :return: lists of lists of ids per user. Each user has an identical first half of their
    ids because it serves as a calibration for transformation to home.
    '''
    assert calib_len <= query_limit
    new_len = query_limit - calib_len

    dataset = torch.load(path_to_data)

    ids = dataset.ids

    calibration, per_user = ids[:calib_len], ids[calib_len:]

    ids_users = []

    for i in range(len(per_user) // new_len):
        this_user_ids = (deepcopy(calibration), deepcopy(per_user[i * new_len : (i+1) * new_len]))
        ids_users.append(this_user_ids)

    return ids_users


def query_sybil_full_dataset_binary(data_path: str) -> pd.DataFrame:
    sybil_reset('binary', 'home')
    sybil_reset('binary', 'defense')

    # list of 19 lists each of 2000 ids
    ids_users = make_user_queries(data_path, calib_len=200)

    # result dict
    result = {}

    # user 0 home
    user0_calib, user0_new = ids_users[0]
    user0_calib_reps = sybil(user0_calib, 'home', 'binary')
    user0_new_reps = sybil(user0_new, 'home', 'binary')
    result[0] = {
        'calib': {
            img_id: rep for img_id, rep in zip(user0_calib, user0_calib_reps)
        },
        'new': {
            img_id: rep for img_id, rep in zip(user0_new, user0_new_reps)
        }
    }
    
    # quer defense with users 1-18
    i = 1
    user_num = len(ids_users)
    while i < user_num:
        sybil_reset('binary', 'defense')
        print(i)
        try:
            this_calib, this_new = ids_users[i]
            this_calib_reps = sybil(this_calib, 'defense', 'binary')
            this_new_reps = sybil(this_new, 'defense', 'binary')
            
            result[i] = {
                'calib': {
                    img_id: rep for img_id, rep in zip(this_calib, this_calib_reps)
                },
                'new': {
                    img_id: rep for img_id, rep in zip(this_new, this_new_reps)
                }
            }

            i = i + 1
        except Exception:
            time.sleep(1)
            print('sleep')

    return result


def query_sybil_full_dataset_binary(data_path: str) -> pd.DataFrame:
    sybil_reset('binary', 'home')
    sybil_reset('binary', 'defense')

    # list of 19 lists each of 2000 ids
    ids_users = make_user_queries(data_path, calib_len=200)

    # result dict
    result = {}

    # user 0 home
    user0_calib, user0_new = ids_users[0]
    user0_calib_reps = sybil(user0_calib, 'home', 'binary')
    user0_new_reps = sybil(user0_new, 'home', 'binary')
    result[0] = {
        'calib': {
            img_id: rep for img_id, rep in zip(user0_calib, user0_calib_reps)
        },
        'new': {
            img_id: rep for img_id, rep in zip(user0_new, user0_new_reps)
        }
    }
    
    # quer defense with users 1-18
    i = 1
    user_num = len(ids_users)
    while i < user_num:
        sybil_reset('binary', 'defense')
        print(i)
        try:
            this_calib, this_new = ids_users[i]
            this_calib_reps = sybil(this_calib, 'defense', 'binary')
            this_new_reps = sybil(this_new, 'defense', 'binary')
            
            result[i] = {
                'calib': {
                    img_id: rep for img_id, rep in zip(this_calib, this_calib_reps)
                },
                'new': {
                    img_id: rep for img_id, rep in zip(this_new, this_new_reps)
                }
            }

            i = i + 1
        except Exception:
            time.sleep(1)
            print('sleep')

    return result


if __name__ == '__main__':
    # result = query_sybil_full_dataset_binary('/home/hack33/task2/SybilAttack.pt')
    result = query_sybil_full_dataset_binary('/home/hack33/task2/SybilAttack.pt')
    with open('sybil_data_binary.json', 'w') as f:
        json.dump(result, f, indent=4)
