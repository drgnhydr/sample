import glob
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

def isSame(x,y,eps=1e-4):
    return (np.abs(x-y) < eps)
        

path = './2024-*.csv'
flist = glob.glob(path)
# df_result = pd.DataFrame(index=[], columns=["r", "K", "ratio"])
# number_r = 29
# r_list = np.linspace(-0.07, 0.07, number_r)
r = 0.02
number_K = 101
K_list = np.linspace(0.03, 0.01, number_K)
array_result = np.zeros(number_K)
L = 1

for file in flist:
    print(file)
    df = pd.read_csv(file)
    r = df.iloc[1]["r"]
    # if r < 0: continue
    L = df.iloc[1]["L"]
    N = L*L
    df = df[["K", "iterations", "steps", "ratio"]]
    
    iterations = df["iterations"].max()

    for id_K in range(len(K_list)):
        if not isSame(K_list[id_K], df["K"][0]): continue
        df_rK = df[["iterations", "steps", "ratio"]]
        mean_iter_list = np.zeros(iterations)
        for iteration in range(iterations):
            df_rK_iter = df_rK[df_rK["iterations"]==iteration][["steps", "ratio"]]
            mean_iter = 0.
            if df_rK_iter["ratio"].min() < 0.5 / N:
                mean_iter = 0.
            elif df_rK_iter["ratio"].max() > 1- 0.5 / N:
                mean_iter = 1.
            else:
                total_step = df_rK_iter["steps"].max()
                mean_iter = df_rK_iter[df_rK_iter["steps"] > 0.5 * total_step]["ratio"].mean()
            mean_iter_list[iteration] = mean_iter
        ratio = mean_iter_list.mean()
        # record = pd.Series([K, r, ratio], index=df_result.columns)
        # df_result = df_result.append(record, ignore_index=True)
        print(ratio)
        array_result[id_K] = ratio

#np.set_printoptions(precision=1)
index_list=["{:.5f}".format(i) for i in K_list]
# colums_list=["{:.4f}".format(i) for i in r_list]
# df_result = pd.DataFrame(data=array_result, index=index_list, columns=["ratio"])
# s1 = pd.Series(array_result, index=index_list, name="ratio")
# s2 = pd.Series([L]*number_K, index=index_list, name="L")
# df_result = pd.concat([s1, s2], axis=1)
df_result = pd.DataFrame(data={"L":[L]*number_K, "ratio":array_result, "K":K_list})
df_result.to_csv("./dataframe_L" + str(L)+".csv")
# for index in index_list:
#     if index[-1] != "0":
#         index = ""
# for colum in colums_list:
#     if not colums_list[-1] in ["0", "5"]:
#         colum = ""
# sns.heatmap(df_result,xticklabels=index_list, yticklabels=colums_list)
# plt.xlabel("Cost-to-Benefit Ratio")
# plt.ylabel("温度係数")
# plt.show()
print(0)
print(1)