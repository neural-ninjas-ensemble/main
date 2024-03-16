import torch
import pandas as pd

num = 500

vector_list=[]
for vector in pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['1'].values:
    vector_list.append(vector.strip('][').split(', '))
df = pd.DataFrame(vector_list)
tensor = torch.tensor(df.values)
torch.save(tensor, f'./data/target_embeding_tensor{num}.pt')
pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['0'].to_csv('id_tagret_{num}.csv')