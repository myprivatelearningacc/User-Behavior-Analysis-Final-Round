import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

ATTRS=['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']

M=[12,31,99,12,31,99]
W=[1,1,100,1,1,100]

DATA_DIR="dataset/"

# ============================================================
# LOAD SEQUENCES
# ============================================================

def parse_X(path):

    df=pd.read_csv(path)

    ids=df.iloc[:,0].astype(str).values

    seqs=[]

    for row in df.iloc[:,1:].values:

        seq=[int(x) for x in row if not pd.isna(x)]

        seqs.append(seq)

    return ids,seqs


train_ids,train_seqs=parse_X(DATA_DIR+"X_train.csv")
val_ids,val_seqs=parse_X(DATA_DIR+"X_val.csv")
test_ids,test_seqs=parse_X(DATA_DIR+"X_test.csv")

Y_train=pd.read_csv(DATA_DIR+"Y_train.csv")
Y_val=pd.read_csv(DATA_DIR+"Y_val.csv")

Y_train=Y_train.set_index(Y_train.columns[0]).loc[train_ids].reset_index()
Y_val=Y_val.set_index(Y_val.columns[0]).loc[val_ids].reset_index()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(seqs):

    feats=[]

    for seq in seqs:

        n=len(seq)

        if n==0:
            feats.append([0]*20)
            continue

        arr=np.array(seq)

        c=Counter(seq)

        entropy=-sum((v/n)*np.log(v/n+1e-9) for v in c.values())

        repeat=1-len(c)/n

        early=np.mean(seq[:max(1,n//4)])
        late=np.mean(seq[max(0,3*n//4):])

        f=[

        seq[0] % 12 if n>0 else -1,
        seq[0] % 31 if n>0 else -1,

        seq[1] % 31 if n>1 else -1,
        seq[2] % 99 if n>2 else -1,

        seq[4] % 99 if n>4 else -1,
        seq[-1] % 99 if n>0 else -1,

        seq[7] % 12 if n>7 else -1,
        seq[9] % 31 if n>9 else -1,

        n,
        entropy,
        repeat,

        arr.mean(),
        arr.std(),

        early,
        late,
        late-early,

        len(c),
        max(c.values()),
        min(c.values()),

        arr.max(),
        arr.min()

        ]

        feats.append(f)

    return np.array(feats)


print("Building features...")

X_tr=build_features(train_seqs)
X_va=build_features(val_seqs)
X_te=build_features(test_seqs)

# ============================================================
# BUILD LOOKUP TABLES
# ============================================================

print("Building lookup tables...")

lookup_first={}
lookup_second={}
lookup_last={}

for seq,row in zip(train_seqs,Y_train.itertuples()):

    if len(seq)>0:
        lookup_first[seq[0]]=row.attr_1

    if len(seq)>1:
        lookup_second[seq[1]]=row.attr_2

    if len(seq)>0:
        lookup_last[seq[-1]]=row.attr_6

# ============================================================
# TRAIN LIGHTGBM
# ============================================================

models=[]

for i,a in enumerate(ATTRS):

    y_tr=Y_train[a].values
    y_va=Y_val[a].values

    model=lgb.LGBMRegressor(

        n_estimators=2000,
        learning_rate=0.02,

        max_depth=10,
        num_leaves=256,

        min_child_samples=20,

        subsample=0.9,
        colsample_bytree=0.9,

        reg_alpha=0.1,
        reg_lambda=0.1,

        verbosity=-1

    )

    model.fit(X_tr,y_tr)

    models.append(model)

# ============================================================
# VALIDATION METRIC
# ============================================================

preds=[]

for m in models:

    preds.append(m.predict(X_va))

preds=np.stack(preds,axis=1)

score=0

for j in range(6):

    diff=(preds[:,j]-Y_val[ATTRS[j]].values)/M[j]

    score+=W[j]*np.mean(diff**2)

print("VAL SCORE:",score)

# ============================================================
# TEST PREDICTION
# ============================================================

preds=[]

for m in models:

    preds.append(m.predict(X_te))

preds=np.stack(preds,axis=1)

# ============================================================
# RULE OVERRIDE
# ============================================================

print("Applying rules...")

for i,seq in enumerate(test_seqs):

    if len(seq)>0 and seq[0] in lookup_first:

        preds[i,0]=lookup_first[seq[0]]

    if len(seq)>1 and seq[1] in lookup_second:

        preds[i,1]=lookup_second[seq[1]]

    if len(seq)>0 and seq[-1] in lookup_last:

        preds[i,5]=lookup_last[seq[-1]]

# ============================================================
# SUBMISSION
# ============================================================

sub=pd.DataFrame({"id":test_ids})

for j,a in enumerate(ATTRS):

    sub[a]=np.clip(preds[:,j],1,M[j]).round().astype(np.uint16)

sub.to_csv("submission_rule_lgbm.csv",index=False)

print("Saved submission_rule_lgbm.csv")
print(sub.head())