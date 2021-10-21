# %%
import json
import matplotlib.pyplot as plt

with open('./data.json', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

#plt.plot(data['time'])
# %%
data.keys()
# %%
len(data['observation'][1])
# %%
len(data['observation'][-1])
# %%
plt.plot(data['time'])
# %%
