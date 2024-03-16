import torch
from typing import List, Tuple
from copy import deepcopy
from taskdataset import TaskDataset

from request import sybil, sybil_submit, sybil_reset


def make_user_queries(path_to_data: str):
    '''
    TODO
    '''
    dataset = torch.load(path_to_data)

    ids = dataset.ids

    calibration, per_user = ids[:1000], ids[1000:]

    ids_users = []

    for i in range(19):
        this_user_ids = deepcopy(calibration) + deepcopy(per_user[i * 1000 : (i+1) * 1000])
        ids_users.append(this_user_ids)

    print(len(ids_users))
    for usr in ids_users:
        print(len(usr))

    return ids_users


def query_sybil_full_dataset_affine(data_path: str):
    sybil_reset('affine', 'home')
    sybil_reset('affine', 'defense')

    # list of 19 lists each of 2000 ids
    ids_users = make_user_queries('/home/hack33/task2/SybilAttack.pt')

    # query home iwth user 0
    home_representations = sybil(ids_users[0], 'home', 'affine')
    defense_representations = []

    # quer defense with users 1-18
    for user_ids in ids_users[1:]:
        defense_representations.append(
            sybil(user_ids, 'defense', 'affine')
        )
        sybil_reset('affine', 'defense')
    
    print(len(defense_representations))


if __name__ == '__main__':
    # make_user_queries('/home/hack33/task2/SybilAttack.pt')
    query_sybil_full_dataset_affine('/home/hack33/task2/SybilAttack.pt')