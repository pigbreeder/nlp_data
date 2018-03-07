import pandas as pd
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import scipy
from sklearn.model_selection import KFold
# from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, hstack
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import xgboost as xgb
def open_dict(Dict = 'hahah', path=r''):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'

deny_word = open_dict(Dict = '否定词', path= r'')
posdict = open_dict(Dict = 'positive', path= r'')
negdict = open_dict(Dict = 'negative', path= r'')
degree_word = open_dict(Dict = '程度级别词语', path= r'')
mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]

def jieba_split(word):
    word_1 = pseg.cut(word)
    n = 0
    v = 0
    su = 0
    each_word = len(word)
    pos=0
    neg=0
    for char in word_1:
        if 'n' in char.flag:
            n += 1
        if 'v' in char.flag:
            v += 1


        su+=1

    segtmp = jieba.lcut(word, cut_all=False)
    i = 0
    a = 0
    poscount = 0
    poscount2 = 0
    poscount3 = 0
    negcount = 0
    negcount2 = 0
    negcount3 = 0
    for word in segtmp:
        if word in posdict:
            pos+=1
            poscount += 1
            c = 0
            for w in segtmp[a:i]:
                if w in mostdict:
                    poscount *= 4.0
                elif w in verydict:
                    poscount *= 3.0
                elif w in moredict:
                    poscount *= 2.0
                elif w in ishdict:
                    poscount *= 0.5
                elif w in deny_word:
                    c += 1
            if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                poscount *= -1.0
                poscount2 += poscount
                poscount = 0
                poscount3 = poscount + poscount2 + poscount3
                poscount2 = 0
            else:
                poscount3 = poscount + poscount2 + poscount3
                poscount = 0
            a = i + 1  # 情感词的位置变化

        elif word in negdict:  # 消极情感的分析，与上面一致
            negcount += 1
            neg+=1
            d = 0
            for w in segtmp[a:i]:
                if w in mostdict:
                    negcount *= 4.0
                elif w in verydict:
                    negcount *= 3.0
                elif w in moredict:
                    negcount *= 2.0
                elif w in ishdict:
                    negcount *= 0.5
                elif w in degree_word:
                    d += 1
            if judgeodd(d) == 'odd':
                negcount *= -1.0
                negcount2 += negcount
                negcount = 0
                negcount3 = negcount + negcount2 + negcount3
                negcount2 = 0
            else:
                negcount3 = negcount + negcount2 + negcount3
                negcount = 0
            a = i + 1
        elif word == '！' or word == '!':  ##判断句子是否有感叹号
            for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                if w2 in posdict or negdict:
                    poscount3 += 2
                    negcount3 += 2
                    break
        i += 1  # 扫描词位置前移

        # 以下是防止出现负数的情况
        pos_count = 0
        neg_count = 0
        if poscount3 < 0 and negcount3 > 0:
            neg_count += negcount3 - poscount3
            pos_count = 0
        elif negcount3 < 0 and poscount3 > 0:
            pos_count = poscount3 - negcount3
            neg_count = 0
        elif poscount3 < 0 and negcount3 < 0:
            neg_count = -poscount3
            pos_count = -negcount3
        else:
            pos_count = poscount3
            neg_count = negcount3




    if (neg_count==0) and (pos_count==0):
        para=0
    else:
        para = pos_count/(neg_count+pos_count)

    return su,n/su,v/su,each_word/su,pos/su,neg/su,pos_count,neg_count,para
def split_word(data):
    data['new'] = data['Discuss'].apply(lambda x:jieba_split(x))
    data_temp = data['new'].apply(pd.Series)
    data = pd.concat((data,data_temp),axis=1)
    data = data.drop(['Discuss','new'],axis=1)
    return data

def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res':list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true,y_pre['res'].values)**0.5)
#

train_result = pd.read_csv('embedding_cnn_train_result2.csv')
train_orig = pd.read_csv('clean_train2.csv')
train_orig = train_orig['Discuss']
train = pd.concat((train_result,train_orig),axis=1)

test_result = pd.read_csv('embedding_add_filter128_shrink_pool2.csv')
test_orig = pd.read_csv('predict_first.csv')
test_orig = test_orig['Discuss']
test = pd.concat((test_result,test_orig),axis=1)
train = split_word(train)
train.to_csv('train_all.csv',index=False)
y = train.pop('label')
train.pop('id')

print(train)
test = split_word(test)
test.to_csv('test_all.csv',index=False)

random_seed = range(1000,2000,10)



gamma = [i/1000.0 for i in range(0,300,3)]

max_depth = [2,3,5]

lambd = range(0,300,2)

subsample = [i/1000.0 for i in range(500,700,2)]

colsample_bytree = [i/1000.0 for i in range(250,350,1)]

min_child_weight = [i/1000.0 for i in range(250,550,3)]



random.shuffle(list(random_seed))





random.shuffle(list(gamma))

random.shuffle(list(max_depth))

random.shuffle(list(lambd))

random.shuffle(list(subsample))

random.shuffle(list(colsample_bytree))

random.shuffle(list(min_child_weight))
cv_pred=[]
for i in range(16):
    train_x,test_x,train_y,test_y = train_test_split(train,y,test_size=0.2,random_state=i)


    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    dtest_1 = xgb.DMatrix(test)
    param = {




        'gamma': gamma[i],

        'max_depth': max_depth[i%3],

        'lambda': lambd[i],

        'subsample': subsample[i],

        'colsample_bytree': colsample_bytree[i],

        'min_child_weight': min_child_weight[i],

        'eta': 0.08,

        'seed': random_seed[i],
        'silent':1,

        'nthread': 8

    }
    evallist  = [(dtrain,'train'), (dtest,'test')]
    num_round = 300
    bst = xgb.train(param, dtrain, num_round, evallist)
    xgb.plot_importance(bst)
    plt.show()
    preds = bst.predict(dtest_1)
    pre = bst.predict(dtest)
    cv_pred.append(preds)
    print(xx_mse_s(pre, test_y))

s = 0
for i in cv_pred:
    s = s + i

s = s/16
res = pd.DataFrame()
res['Id'] = list(test_id)
res['pre'] = list(s)


res.to_csv('pre.csv',index=False)
