import torch
import pandas as pd

num = 500

vector_list=[]
for vector in pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['1'].values:
    vector_list.append([ float( i) for i in vector.strip('][').split(', ')])
df1 = pd.DataFrame(vector_list)
# tensor = torch.tensor(df.values)
# torch.save(tensor, f'./data/target_embeding_tensor{num}.pt')
# pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['0'].to_csv('id_tagret_{num}.csv')
_1 = pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['0']

num2 = 2000
_2 = pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['0']


vector_list=[]
for vector in pd.read_csv(f'./data/data_dla_wojtka/target_embeding_{num}.csv')['1'].values:
    vector_list.append([ float( i) for i in vector.strip('][').split(', ')])
df2 = pd.DataFrame(vector_list)
df3 = pd.concat([df1,df2])
_3 = pd.concat([_1,_2])
tensor = torch.tensor(df3.values)
torch.save(tensor, f'./data/target_embeding_tensor2500.pt')
_3.to_csv('./data/id_tagret_2500.csv')