import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb

ATTRS=['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']

M=[12,31,99,12,31,99]
W=[1,1,100,1,1,100]

# ---------------------------
# LOAD
# ---------------------------

def parse_X(path):

    df=pd.read_csv(path)

    ids=df.iloc[:,0].values

    seqs=[]

    for row in df.iloc[:,1:].values:

        seq=[int(x) for x in row if not pd.isna(x)]

        seqs.append(seq)

    return ids,seqs


train_ids,train_seqs=parse_X("dataset/X_train.csv")
val_ids,val_seqs=parse_X("dataset/X_val.csv")
test_ids,test_seqs=parse_X("dataset/X_test.csv")

Y_train=pd.read_csv("dataset/Y_train.csv")
Y_val=pd.read_csv("dataset/Y_val.csv")

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

def build_features(seqs):

    feats=[]

    for seq in seqs:

        n=len(seq)

        arr=np.array(seq)

        c=Counter(seq)

        entropy=-sum((v/n)*np.log(v/n+1e-9) for v in c.values())

        repeat=1-len(c)/n

        early=np.mean(seq[:max(1,n//4)])

        late=np.mean(seq[max(0,3*n//4):])

        f=[

        seq[0] if n>0 else -1,
        seq[1] if n>1 else -1,
        seq[2] if n>2 else -1,
        seq[4] if n>4 else -1,
        seq[7] if n>7 else -1,
        seq[9] if n>9 else -1,
        seq[-1] if n>0 else -1,

        n,
        entropy,
        repeat,
        arr.mean(),
        arr.std(),
        early,
        late,
        late-early

        ]

        feats.append(f)

    return np.array(feats)


X_tr=build_features(train_seqs)
X_va=build_features(val_seqs)
X_te=build_features(test_seqs)

# ---------------------------
# TRAIN
# ---------------------------

models=[]

for i,a in enumerate(ATTRS):

    y_tr=Y_train[a].values
    y_va=Y_val[a].values

    model=lgb.LGBMRegressor(

        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8

    )

    model.fit(

        X_tr,y_tr,
        eval_set=[(X_va,y_va)],
        eval_metric="l2",
        verbose=False

    )

    models.append(model)

# ---------------------------
# VALIDATION METRIC
# ---------------------------

preds=[]

for i,m in enumerate(models):

    preds.append(m.predict(X_va))

preds=np.stack(preds,axis=1)

score=0

for j in range(6):

    diff=(preds[:,j]-Y_val[ATTRS[j]].values)/M[j]

    score+=W[j]*np.mean(diff**2)

print("VAL SCORE:",score)

# ---------------------------
# TEST
# ---------------------------

preds=[]

for m in models:

    preds.append(m.predict(X_te))

preds=np.stack(preds,axis=1)

# ---------------------------
# SUBMISSION
# ---------------------------

sub=pd.DataFrame({"id":test_ids})

for j,a in enumerate(ATTRS):

    sub[a]=np.clip(preds[:,j],1,M[j]).round().astype(np.uint16)

sub.to_csv("submission_lgbm.csv",index=False)

print("saved submission_lgbm.csv")