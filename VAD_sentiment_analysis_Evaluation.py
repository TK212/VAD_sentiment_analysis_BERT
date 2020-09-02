# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:00:47 2018

@author: Ryutaro Takanami
"""


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np

df_V = pd.read_table('DissertationData\VADprediction\V_score_bySemEval2018.txt', delimiter=',', header=None)
df_A = pd.read_table('DissertationData\VADprediction\A_score_bySemEval2018.txt', delimiter=',', header=None)
df_D = pd.read_table('DissertationData\VADprediction\D_score_bySemEval2018.txt', delimiter=',', header=None)

predicted_df = pd.concat([df_V, df_A, df_D], axis=1)
predicted_df.columns = ["ind1","V","ind2","A","ind3","D"]

df_ang = pd.read_table('DissertationData\WASSA2017anger.txt', delimiter='\t', header=None)
df_fear = pd.read_table('DissertationData\WASSA2017fear.txt', delimiter='\t', header=None)
df_joy = pd.read_table('DissertationData\WASSA2017joy.txt', delimiter='\t', header=None)
df_sadness = pd.read_table('DissertationData\WASSA2017sadness.txt', delimiter='\t', header=None)

df_polarity_true = pd.read_table('DissertationData\polarity\SemEval2017-task4-test.subtask-A.english.txt', delimiter='\t', header=None)
df_polarity_predicted = pd.read_table('DissertationData\polarity\V_score_Polarity_SemEval2018.txt', delimiter=',')

df_polarity = df_polarity_predicted
df_polarity['polarity'] = df_polarity_true[1]
df_polarity['text'] = df_polarity_true[2]
df_polarity.columns = ["id","V","polarity","text"]







#Polarity value count
#print(df_polarity['polarity'].value_counts())





#Polarity Plot
x = df_polarity['V']
y = df_polarity['id']

plt.xlim(0, 1000)
plt.ylim(0, 1000)

plt.xlabel("V score", fontsize=20)
plt.ylabel("ID", fontsize=20)


color = df_polarity
color.loc[color['polarity'] == 'positive', 'polarity'] = 'r'
color.loc[color['polarity'] == 'neutral', 'polarity'] = 'y'
color.loc[color['polarity'] == 'negative', 'polarity'] = 'b'


plt.scatter(x, y, c=color['polarity'])
plt.show()


#Polarity Accuracy
negative_num = 0
neutral_num = 0
positive_num = 0

negative_miss = 0
neutral_miss = 0
positive_miss = 0


for index1, line in df_polarity.iterrows():
    if line['V'] < 300:
        negative_num += 1
        if line['polarity'] != 'b':
            negative_miss += 1
    elif line['V'] < 700:
        neutral_num+= 1
        if line['polarity'] != 'y':
            neutral_miss += 1
    else:
        positive_num += 1
        if line['polarity'] != 'r':
            positive_miss += 1

print("Negative: Num, ", negative_num, "Miss, ", negative_miss, "Accuracy, ", 1-negative_miss/negative_num)
#print("Neutral: Num, ", neutral_num, "Miss, ", neutral_miss, "Accuracy, ", 1-neutral_miss/neutral_num)
print("Positive: Num, ", positive_num, "Miss, ", positive_miss, "Accuracy, ", 1-positive_miss/positive_num)





df_VAD = pd.DataFrame([
    ['anger', 167, 865, 657], 
    ['fear', 73, 840, 293],
    ['joy', 980, 824, 794],
    ['sadness', 52, 288, 164],
    ])
    





# 3D散布図でプロットするデータを生成する為にnumpyを使用
"""
X_mark = [0.167, 0.073, 0.98, 0.052]
Y_mark = [0.865, 0.84, 0.824, 0.288]
Z_mark = [0.657, 0.293, 0.794, 0.164]
"""

X_mark = [167, 73, 980, 52]
Y_mark = [865, 840, 824, 288]
Z_mark = [657, 293, 794, 164]


"""
X = df_V[1]
Y = df_A[1]
Z = df_D[1]
"""


"""
X = predicted_df['V']
Y = predicted_df['A']
Z = predicted_df['D']

# グラフの枠を作成
fig = plt.figure()
ax = Axes3D(fig)

# X,Y,Z軸にラベルを設定
ax.set_xlabel("V")
ax.set_ylabel("A")
ax.set_zlabel("D")

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 1000)

#fig, ax = plt.subplots()

# .plotで描画
ax.scatter(X,Y,Z,marker="o",linestyle='None')
ax.scatter(X_mark, Y_mark, Z_mark, s=50, marker="*", color='r')


# 最後に.show()を書いてグラフ表示
plt.show()
"""



"""
#最短距離を計算するため
#predicted_df = predicted_df["V","A","D"]

#predicted_df = predicted_df[1:10]

min_distances =[]
predicted_labels = []
for index1, line in predicted_df.iterrows():
    distance_list = []
    for index2, emotion in df_VAD.iterrows():
        distance = math.sqrt((emotion[1] - line['V']) ** 2 + (emotion[2] - line['A']) ** 2 + (emotion[3] - line['D']) ** 2)
        distance_list.append(distance)
    predicted_labels.append(df_VAD[0][np.argmin(distance_list)])
     



#それぞれの予測感情ごとのaccuracyを調べる
df_WASSA = pd.concat([df_ang, df_fear, df_joy, df_sadness], ignore_index = True)
df_WASSA.columns = ["id","text","label","intensity"]
true_labels = df_WASSA[["label"]]

anger_accuracy = 0
fear_accuracy = 0
joy_accuracy = 0
sadness_accuracy = 0

anger_num = 0
fear_num = 0
joy_num = 0
sadness_num = 0

for index1, true_label in true_labels.iterrows():
    if(true_label[0] == 'anger'):
        anger_num += 1
        if(true_label[0] == predicted_labels[index1]):
            anger_accuracy += 1
            
    if(true_label[0] == 'fear'):
        fear_num += 1
        if(true_label[0] == predicted_labels[index1]):
            fear_accuracy += 1
            
    if(true_label[0] == 'joy'):
        joy_num += 1
        if(true_label[0] == predicted_labels[index1]):
            joy_accuracy += 1
            
    if(true_label[0] == 'sadness'):
        sadness_num += 1
        if(true_label[0] == predicted_labels[index1]):
            sadness_accuracy += 1

print("Anger: Num, ", anger_num, "Correct_Num, ", anger_accuracy, "Accuracy, ", anger_accuracy/anger_num)
print("Fear: Num, ", fear_num, "Correct_Num, ", fear_accuracy, "Accuracy, ", fear_accuracy/fear_num)
print("Joy: Num, ", joy_num, "Correct_Num, ", joy_accuracy, "Accuracy, ", joy_accuracy/joy_num)
print("Sadness: Num, ", sadness_num, "Correct_Num, ", sadness_accuracy, "Accuracy, ", sadness_accuracy/sadness_num)

"""





"""
#3カテゴリに緩めたバージョンの正答率

df_WASSA = pd.concat([df_ang, df_fear, df_joy, df_sadness], ignore_index = True)
df_WASSA.columns = ["id","text","label","intensity"]
true_labels = df_WASSA[["label"]]


accuracy = 0
for index1, true_label in true_labels.iterrows():
    if(true_label[0] == predicted_labels[index1]):
        accuracy += 1
    elif((predicted_labels[index1]=='anger') & (true_label[0]=='fear')):
        accuracy += 1
    elif((predicted_labels[index1]=='fear') & (true_label[0]=='anger')):
        accuracy += 1
    #if((true_label[0] == ''))
accuracy = accuracy / len(true_labels)
print('Accuracy : ', accuracy)

"""









