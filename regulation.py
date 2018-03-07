import pandas as pd
threshold_4_5 = 4.75
def cata(x):
    if (x < 4)and(x>=1):
        return x
    elif (x < threshold_4_5)and (x>=4):
        return 4
    elif x<=1:
        return 1
    elif x>=threshold_4_5:
        return 5

data = pd.read_csv('pre.csv',names=['id','score'])
data_1 = pd.read_csv('cnn-baseline2.csv',names=['id','score'])
data_2 = pd.read_csv('tfidf_cleardata2_no_threshold.csv',names=['id','score'])
data['score'] = data.score.apply(lambda x:cata(x))*0.51824/(0.51223+0.51146+0.51824)
data_1['score'] = data_1.score.apply(lambda x:cata(x))*0.51146/(0.51223+0.51146+0.51824)
data_2['score'] = data_2.score.apply(lambda x:cata(x))*0.51223/(0.51223+0.51146+0.51824)
data = pd.merge(data,data_1,how='left',on='id')
data = pd.merge(data,data_2,how='left',on='id')
print(data.head())
data['score'] = data['score_x']+data['score_y']+data['score']
data = data.drop(['score_x','score_y'],axis=1)
data.to_csv('sub.csv',index=False,header=False)