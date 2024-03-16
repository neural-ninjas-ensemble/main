import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import scipy
from sklearn.linear_model import Ridge
import numpy as np

from taskdataset import TaskDataset


def revert_user_ridge(calib_dirty, calib_clean, x_dirty):
    clf = Ridge(alpha=1.0)
    clf.fit(calib_dirty, calib_clean)
    return clf.predict(x_dirty).tolist()


# def revert_user_scipy(calib_dirty, calib_clean, x_dirty):
#     calib_dirty, calib_clean, x_dirty = calib_dirty.numpy(), calib_clean.numpy(), x_dirty.numpy()
#     x_inv = scipy.sparse.linalg.pinv(calib_clean)
#     A = np.outer(calib_dirty, x_inv)
#     return scipy.linalg.inv(A)


def revert_user(calib_dirty, calib_clean, x_dirty):
    '''
    calib_dirty and calib_clean must be the same order 
    returns x_clean
    '''
    # Create an instance of the model
    model = nn.Linear(384, 384)

    # Convert data to PyTorch tensors
    inputs = torch.tensor(calib_dirty)
    targets = torch.tensor(calib_clean)
    to_translate = torch.tensor(x_dirty)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4)

    # Training loop
    num_epochs = 5000
    for epoch in range(num_epochs):
        # TODO add validation part
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    x_clean_tensor = model(to_translate)

    return x_clean_tensor.tolist()


def revert_transforms(data: dict) -> dict:
    # the assumption is that user 0 is home and rest is defense
    data_clean = {} # image_id: embedding

    data_clean.update(data['0']['calib'])
    data_clean.update(data['0']['new'])

    calib_order = sorted(data['0']['calib'].keys())
    calib_clean = [
        data['0']['calib'][i]
        for i in calib_order
    ]

    for user_id, user_data in data.items():
        if user_id == '0':
            continue
        
        calib_dirty = [
            user_data['calib'][i]
            for i in calib_order
        ]

        new_order = sorted(user_data['new'].keys())
        new_dirty = [
            user_data['new'][i] for i in new_order
        ]

        new_clean = revert_user(calib_dirty, calib_clean, new_dirty)

        data_clean.update(
            {
                image_id: clean_emb for image_id, clean_emb in zip(new_order, new_clean)
            }
        )

    print(len(data_clean))

    return(data_clean)




if __name__ == '__main__':
    with open('sybil_data.json', 'r') as f:
        data = json.load(f)

    reverted_data = revert_transforms(data)

    dataset = torch.load('/home/hack33/task2/SybilAttack.pt')
    id_order = dataset.ids
    reps_in_order = [reverted_data[str(i)] for i in id_order]

    np.savez(
        "submission.npz",
        ids=id_order,
        representations=reps_in_order,
    )

    with open('sybil_decoded.json', 'w') as f:
        json.dump(reverted_data, f, indent=4)
