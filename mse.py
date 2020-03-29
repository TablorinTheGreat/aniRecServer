from sklearn.preprocessing import MinMaxScaler
import numpy as np
import socket
import math
import os.path
import scipy.stats
import spice_api
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import time
def ends():
    #define stream chunk
    import pyaudio
    import wave
    chunk = 1024

    #open a wav format music
    f = wave.open('TPT3TONE.wav',"rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()
anime=pd.read_csv('anime.csv')
anime["rating"] = anime["rating"].astype(float)
anime["members"] = anime["members"].astype(float)
def pre():
    global distances,indices,anime_features,mean_distance,rt
    anime_features = pd.concat([anime["genre"].str.get_dummies(sep=","),pd.get_dummies(anime[["type"]]),anime[["rating"]],anime[["members"]],anime["episodes"]],axis=1)
    min_max_scaler = MinMaxScaler()
    anime_features = min_max_scaler.fit_transform(anime_features)
    np.round(anime_features,2)
    myratingz = pd.read_csv("myrating.csv")
    watched_names = []
    myr = []
    for index, row in myratingz.iterrows():
        watched_names = np.append(watched_names, anime.loc[anime['anime_id'] == int(row.anime_id), 'name'].values)
        myr += [[0, int(row.anime_id), int(row.rating)]]
    my=pd.DataFrame(myr, columns=['user_id', 'anime_id', 'rating'])
    rt = my.append(pd.read_csv('rating2.csv'))
    name_data = watched_names
    anime_voters_list={}
    mean_distance={}
    if os.path.isfile("top50.csv"):
        top = pd.read_csv("top50.csv")
    else:
        top = top50()
        top.to_csv("top50.csv")
    mymean = float(my['rating'].values.mean())
    user_sort_table = []
    sum=0
    for i in name_data:
        print i
        sigma = 0
        id = int(anime.loc[anime['name'] == i, 'anime_id'].values)
        users = rt.loc[(rt['anime_id'] == id) & (rt['user_id'].isin(top['user_id'].values)), 'user_id'].values
        if len(users) >= 5:
            weightsum = top.loc[top['user_id'].isin(users), 'comp'].values.sum()
            for j in users:
                usrt = rt.loc[(rt['user_id'] == j) & (rt['anime_id'] == id), 'rating'].values
                usmean = rt.loc[rt.user_id == j, 'rating'].values.mean()
                sigma += (float(usrt) - float(usmean)) * float(top.loc[top['user_id'] == j, 'comp'])
            predrate = sigma/ weightsum
            predrate += mymean
            myrate = myratingz.loc[myratingz["anime_id"] == get_id_from_name(i), "rating"].values[0]
            sum+=(predrate-myrate)**2
    sum/=len(name_data)
    print(sum)
def get_index_from_name(name):
    return anime[anime["name"]==name].index.tolist()[0]
def get_similar_animes(id=None):
    global distances, indices
    return anime.ix[indices[id][1:]]["name"],distances[id][1:]
def get_id_from_name(name):
    return anime.loc[anime["name"] == name, "anime_id"].values[0]
def get_score_from_name(name):
    return anime.loc[anime["name"] == name, "rating"].values[0]
def key_sort(item):
    return item[3]
def save(s,n):
    with open(n, 'w') as f:
        for item in s:
            f.write("%s\n" % item)
def ins(l, obj):
    oid, ocomp = obj
    for i in range(len(l) - 1, -1, -1):
        id, comp = l[i]
        if not math.isnan(ocomp):
            if ocomp < comp:
                l = np.insert(l, i + 1, obj, axis=0)
                break
            elif i == 0 & (ocomp > comp):
                l = np.insert(l, i, obj, axis=0)
                break
    if (len(l) > 50):
        l = np.delete(l, 50, axis=0)
    return l
def top50():
    rt["user_id"] = rt["user_id"].astype(int)
    rt["rating"] = rt["rating"].astype(float)
    q = 1
    a = 0
    b = 1000
    g = [[-1.0, -1.0]]
    while q != 0:
        piv = rt[((rt.user_id <= b) & (rt.user_id > a)) | (rt.user_id == 0)]
        a += 1000
        b += 1000
        q = np.unique(piv['user_id'].values).argmax()
        piv = piv.pivot_table(index=['user_id'], columns=['anime_id'], values='rating')
        l = piv.iloc[0].values
        v = ~np.isnan(l)
        s = len(l[v])
        s = float(s)
        for i in range(len(piv.index.values)):
            if i != 0:
                print piv.index[i]
                y = piv.iloc[i].values
                mask = ~np.isnan(l) & ~np.isnan(y)
                x = l[mask]
                m = y[mask]
                r, p = scipy.stats.pearsonr(x, m)
                k = (float(len(x)) / s)*100
                if k>=25:
                  g = ins(g, [piv.index[i], r])
    return pd.DataFrame(g, columns=['user_id', 'comp'])
def voters2str(l):
    s=""
    for a in l:
        s+=a
        s+=" ,"
    return s[:len(s)-2]
def mattostr(p):
    g=""
    for k in p:
        for i in k:
            if isinstance(i,list):
                g+=voters2str(i)
            else:
                g+=str(i)+" ,"
        g+='\n'
    return g

u,i=pre()

