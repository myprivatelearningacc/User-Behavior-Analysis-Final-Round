import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
DATE_ATTRS = ['attr_1','attr_2','attr_4','attr_5']
REG_ATTRS = ['attr_3','attr_6']

M = {'attr_1':12,'attr_2':31,'attr_3':99,'attr_4':12,'attr_5':31,'attr_6':99}
W = {'attr_1':1,'attr_2':1,'attr_3':100,'attr_4':1,'attr_5':1,'attr_6':100}

DATA_DIR="dataset/"

BATCH=1024
LR=3e-4
EPOCHS=80
PATIENCE=12

EMB=128
TCN_DIM=256
POS_EMB_DIM=16

# ============================================================
# LOAD DATA
# ============================================================

def parse_X(path):

    df=pd.read_csv(path)

    ids=df.iloc[:,0].astype(str).values

    seqs={}

    for i,row in enumerate(df.iloc[:,1:].values):

        seq=[int(x) for x in row if not pd.isna(x)]

        seqs[ids[i]]=seq

    return seqs,ids


train_seqs,train_ids=parse_X(DATA_DIR+"X_train.csv")
val_seqs,val_ids=parse_X(DATA_DIR+"X_val.csv")
test_seqs,test_ids=parse_X(DATA_DIR+"X_test.csv")

Y_train=pd.read_csv(DATA_DIR+"Y_train.csv")
Y_val=pd.read_csv(DATA_DIR+"Y_val.csv")

Y_train=Y_train.set_index(Y_train.columns[0]).loc[train_ids].reset_index()
Y_val=Y_val.set_index(Y_val.columns[0]).loc[val_ids].reset_index()

# ============================================================
# VOCAB
# ============================================================

tokens=set()

for d in [train_seqs,val_seqs,test_seqs]:
    for seq in d.values():
        tokens.update(seq)

action2idx={a:i+3 for i,a in enumerate(sorted(tokens))}
action2idx[0]=0
action2idx['UNK']=1
action2idx['MASK']=2

VOCAB=len(action2idx)+1

MAX_LEN=int(np.percentile([len(s) for s in train_seqs.values()],95))

# ============================================================
# ENCODE
# ============================================================

def encode(seqs,ids):

    X=np.zeros((len(ids),MAX_LEN))
    L=np.zeros(len(ids))

    for i,uid in enumerate(ids):

        seq=seqs[uid]

        l=min(len(seq),MAX_LEN)

        for j in range(l):
            X[i,j]=action2idx.get(seq[j],1)

        L[i]=max(l,1)

    return torch.LongTensor(X),torch.LongTensor(L)


X_tr,L_tr=encode(train_seqs,train_ids)
X_va,L_va=encode(val_seqs,val_ids)
X_te,L_te=encode(test_seqs,test_ids)

# ============================================================
# POSITION TOKENS
# ============================================================

def extract_pos(seqs,ids):

    pos=np.full((len(ids),4),action2idx['MASK'])

    for i,uid in enumerate(ids):

        seq=seqs[uid]
        n=len(seq)

        if n>0: pos[i,0]=action2idx.get(seq[0],1)
        if n>1: pos[i,1]=action2idx.get(seq[1],1)
        if n>2: pos[i,2]=action2idx.get(seq[2],1)
        if n>0: pos[i,3]=action2idx.get(seq[-1],1)

    return torch.LongTensor(pos)


POS_tr=extract_pos(train_seqs,train_ids)
POS_va=extract_pos(val_seqs,val_ids)
POS_te=extract_pos(test_seqs,test_ids)

# ============================================================
# FEATURES
# ============================================================

def features(seqs,ids):

    feats=[]

    for uid in ids:

        seq=seqs[uid]
        n=len(seq)

        arr=np.array(seq)

        c=Counter(seq)

        unique=len(c)

        entropy=-sum((v/n)*np.log(v/n+1e-9) for v in c.values())

        repeat=1-unique/n

        maxfreq=max(c.values())

        bigrams=list(zip(seq[:-1],seq[1:]))

        bigram_div=len(set(bigrams))/len(bigrams) if bigrams else 0

        early=np.mean(seq[:max(1,n//4)])
        late=np.mean(seq[max(0,3*n//4):])

        feats.append([

            n,
            unique,
            repeat,
            entropy,
            maxfreq,
            bigram_div,
            arr.mean(),
            arr.std(),
            early,
            late,
            late-early

        ])

    return np.array(feats)


F_tr=features(train_seqs,train_ids)
F_va=features(val_seqs,val_ids)
F_te=features(test_seqs,test_ids)

scaler=StandardScaler()

F_tr=torch.FloatTensor(scaler.fit_transform(F_tr))
F_va=torch.FloatTensor(scaler.transform(F_va))
F_te=torch.FloatTensor(scaler.transform(F_te))

AUX=F_tr.shape[1]

# ============================================================
# PERSONA (quantile)
# ============================================================

def persona(raw):

    tm=raw[:,6]

    q=np.percentile(tm,[25,50,75])

    p=[]

    for v in tm:

        if v<q[0]: p.append(0)
        elif v<q[1]: p.append(1)
        elif v<q[2]: p.append(2)
        else: p.append(3)

    return torch.LongTensor(p)


P_tr=persona(F_tr.numpy())
P_va=persona(F_va.numpy())
P_te=persona(F_te.numpy())

# ============================================================
# TARGET
# ============================================================

def targets(df):

    y_d=torch.LongTensor(
        np.stack([df[a].values-1 for a in DATE_ATTRS],axis=1)
    )

    y_r=torch.FloatTensor(
        np.stack([df[a].values/99 for a in REG_ATTRS],axis=1)
    )

    return y_d,y_r


y_tr_d,y_tr_r=targets(Y_train)
y_va_d,y_va_r=targets(Y_val)

# ============================================================
# DATASET
# ============================================================

class DS(Dataset):

    def __init__(self,X,L,F,P,POS,y_d=None,y_r=None):

        self.X=X;self.L=L;self.F=F;self.P=P;self.POS=POS
        self.y_d=y_d;self.y_r=y_r

    def __len__(self):
        return len(self.X)

    def __getitem__(self,i):

        base=(self.X[i],self.L[i],self.F[i],self.P[i],self.POS[i])

        if self.y_d is None:
            return base

        return base+(self.y_d[i],self.y_r[i])


train_dl=DataLoader(
    DS(X_tr,L_tr,F_tr,P_tr,POS_tr,y_tr_d,y_tr_r),
    batch_size=BATCH,
    shuffle=True
)

val_dl=DataLoader(
    DS(X_va,L_va,F_va,P_va,POS_va,y_va_d,y_va_r),
    batch_size=BATCH
)

# ============================================================
# TCN MODEL
# ============================================================

class Block(nn.Module):

    def __init__(self,in_c,out_c,d):

        super().__init__()

        self.conv=nn.Conv1d(in_c,out_c,3,padding=d,dilation=d)
        self.act=nn.GELU()
        self.skip=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else nn.Identity()

    def forward(self,x):

        h=self.act(self.conv(x))

        return h+self.skip(x)


class TCN(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(VOCAB,EMB,padding_idx=0)

        self.tcn=nn.Sequential(

            Block(EMB,TCN_DIM,1),
            Block(TCN_DIM,TCN_DIM,2),
            Block(TCN_DIM,TCN_DIM,4)

        )

        self.pool=nn.AdaptiveAvgPool1d(1)

        self.pos_emb=nn.Embedding(VOCAB,POS_EMB_DIM)

        self.persona_emb=nn.Embedding(4,16)

        self.aux=nn.Linear(AUX,64)

        self.fc=nn.Sequential(

            nn.Linear(TCN_DIM+POS_EMB_DIM*4+64+16,256),
            nn.GELU(),
            nn.Dropout(0.3)

        )

        self.cls_heads=nn.ModuleDict({

            'attr_1':nn.Linear(256,12),
            'attr_2':nn.Linear(256,31),
            'attr_4':nn.Linear(256,12),
            'attr_5':nn.Linear(256,31)

        })

        self.reg_heads=nn.ModuleDict({

            'attr_3':nn.Linear(256,1),
            'attr_6':nn.Linear(256,1)

        })


    def forward(self,x,l,f,p,pos):

        emb=self.emb(x).transpose(1,2)

        t=self.tcn(emb)

        t=self.pool(t).squeeze(-1)

        pos=self.pos_emb(pos).flatten(1)

        p=self.persona_emb(p)

        f=self.aux(f)

        z=torch.cat([t,pos,f,p],dim=1)

        z=self.fc(z)

        out_cls={a:self.cls_heads[a](z) for a in DATE_ATTRS}

        out_reg={a:torch.sigmoid(self.reg_heads[a](z)).squeeze(-1) for a in REG_ATTRS}

        return out_cls,out_reg


model=TCN().to(DEVICE)

# ============================================================
# LOSS
# ============================================================

ce=nn.CrossEntropyLoss()

huber=nn.HuberLoss()

W_LOSS={'attr_3':20,'attr_6':20}

# ============================================================
# TRAIN
# ============================================================

opt=torch.optim.AdamW(model.parameters(),lr=LR)

best=999
best_state=None
pat=0

for epoch in range(EPOCHS):

    model.train()

    for batch in train_dl:

        seq,l,f,p,pos,y_d,y_r=[t.to(DEVICE) for t in batch]

        opt.zero_grad()

        o_d,o_r=model(seq,l,f,p,pos)

        loss=0

        for i,a in enumerate(DATE_ATTRS):
            loss+=ce(o_d[a],y_d[:,i])

        for i,a in enumerate(REG_ATTRS):
            loss+=W_LOSS[a]*huber(o_r[a],y_r[:,i])

        loss.backward()

        opt.step()

    # validation metric

    model.eval()

    preds={a:[] for a in ATTRS}

    with torch.no_grad():

        for batch in val_dl:

            seq,l,f,p,pos,y_d,y_r=[t.to(DEVICE) for t in batch]

            o_d,o_r=model(seq,l,f,p,pos)

            for a in DATE_ATTRS:
                preds[a].append(o_d[a].argmax(1).cpu().numpy())

            for a in REG_ATTRS:
                preds[a].append((o_r[a]*99).cpu().numpy())

    for a in ATTRS:
        preds[a]=np.concatenate(preds[a])

    score=0

    for i,a in enumerate(DATE_ATTRS):

        diff=(preds[a]+1-Y_val[a].values)/M[a]

        score+=W[a]*np.mean(diff**2)

    for i,a in enumerate(REG_ATTRS):

        diff=(preds[a]-Y_val[a].values)/M[a]

        score+=W[a]*np.mean(diff**2)

    print(epoch,score)

    if score<best:

        best=score
        best_state=model.state_dict()
        pat=0

    else:
        pat+=1

    if pat>=PATIENCE:
        break


print("BEST",best)

# ============================================================
# TEST PREDICT
# ============================================================

model.load_state_dict(best_state)

model.eval()

test_dl=DataLoader(
    DS(X_te,L_te,F_te,P_te,POS_te),
    batch_size=BATCH
)

preds={a:[] for a in ATTRS}

with torch.no_grad():

    for seq,l,f,p,pos in test_dl:

        seq,l,f,p,pos=[t.to(DEVICE) for t in [seq,l,f,p,pos]]

        o_d,o_r=model(seq,l,f,p,pos)

        for a in DATE_ATTRS:
            preds[a].append(o_d[a].argmax(1).cpu().numpy()+1)

        for a in REG_ATTRS:
            preds[a].append((o_r[a]*99).cpu().numpy())


for a in ATTRS:
    preds[a]=np.concatenate(preds[a])

# ============================================================
# LOOKUP OVERRIDE
# ============================================================

for i,uid in enumerate(test_ids):

    seq=test_seqs[uid]

    if len(seq)>=3:

        preds['attr_1'][i]=seq[0]%12+1
        preds['attr_2'][i]=seq[1]%31+1

# ============================================================
# SUBMISSION
# ============================================================

sub=pd.DataFrame({"id":test_ids})

for a in ATTRS:

    sub[a]=np.clip(preds[a],1,M[a]).round().astype(np.uint16)

sub.to_csv("submission_tcn_v2.csv",index=False)

print("Saved submission_tcn_v2.csv")
print(sub.head())