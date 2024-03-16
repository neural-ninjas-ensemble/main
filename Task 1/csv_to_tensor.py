import torch
import pandas as pd

num = 2000

vector_list=[]
for vector in pd.read_csv(f'./data/target_embeding_{num}.csv')['1'].values:
    vector_list.append([float(i) for i in vector.strip('][').split(', ')])
df = pd.DataFrame(vector_list)
tensor = torch.tensor(df.values)
torch.save(tensor, f'./data/TargetEmbeddings{num}.pt')
ids = pd.read_csv(f'./data/target_embeding_{num}.csv').rename(columns={'0': 'id'})['id']

ids.to_csv(f'./data/ids{num}.csv', index=False)
