# %%
import json
import matplotlib.pyplot as plt


with open('./latest.json', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)


def get_vec_by_id(data, key='observation', idx=0):
    loc_data = data[key]
    vector = [datarow[idx] for datarow in loc_data]
    return vector


# plt.plot(data['time'])
# %%
data.keys()
# %%
len(data['observation']['motors'][0]['power'])
# %%
# %%
plt.plot(data['time'])
# %%
plt.plot(data['observation']['motors'][0]['power'])
# %%
# %%
plt.plot(data['observation']['analogs'][0]['force'])
# %%
plt.plot(data['observation']['analogs'][0]['dforce'])
# %%
plt.plot(data['observation']['force_torques'][0]['mx'])
# %%
