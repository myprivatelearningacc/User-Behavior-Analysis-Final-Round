import os
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =============================
# CONFIG
# =============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']

M = torch.tensor([12,31,99,12,31,99],dtype=torch.float32).to(DEVICE)
W = torch.tensor([1,1,100,1,1,100],dtype=torch.float32).to(DEVICE)

BATCH_SIZE = 1024
EPOCHS = 60
LR = 3e-4

# =============================
# DATA LOADER
# =============================

def parse_sequences(path):

    df = pd.read_csv(path)

    ids = df.iloc[:,0].astype(str).values

    seqs = []
    for row in df.iloc[:,1:].values:

        seq = [int(x) for x in row if not pd.isna(x)]
        seqs.append(seq)

    return ids, seqs


train_ids, train_seqs = parse_sequences("dataset/X_train.csv")
val_ids, val_seqs = parse_sequences("dataset/X_val.csv")
test_ids, test_seqs = parse_sequences("dataset/X_test.csv")

Y_train = pd.read_csv("dataset/Y_train.csv")
Y_val = pd.read_csv("dataset/Y_val.csv")

# =============================
# VOCAB
# =============================

all_actions = set()

for seq in train_seqs + val_seqs + test_seqs:
    all_actions.update(seq)

action2idx = {a:i+2 for i,a in enumerate(sorted(all_actions))}
action2idx[0]=0
action2idx['UNK']=1

VOCAB_SIZE=len(action2idx)+1

MAX_LEN = int(np.percentile([len(s) for s in train_seqs],95))

# =============================
# ENCODE
# =============================

def encode(seqs):

    X=np.zeros((len(seqs),MAX_LEN),dtype=np.int64)
    L=np.zeros(len(seqs),dtype=np.int64)

    for i,seq in enumerate(seqs):

        l=min(len(seq),MAX_LEN)

        for j in range(l):
            X[i,j]=action2idx.get(seq[j],1)

        L[i]=max(l,1)

    return torch.LongTensor(X), torch.LongTensor(L)

X_tr,L_tr=encode(train_seqs)
X_va,L_va=encode(val_seqs)
X_te,L_te=encode(test_seqs)

# =============================
# BEHAVIOR FEATURES
# =============================

def behavior_features(seqs):

    feats=[]

    for seq in seqs:

        n=len(seq)

        c=Counter(seq)

        uniq=len(c)

        entropy=-sum((v/n)*np.log(v/n) for v in c.values())

        repeat_ratio=1-uniq/n

        max_freq=max(c.values())

        feats.append([
            n,
            uniq,
            uniq/n,
            repeat_ratio,
            entropy,
            max_freq
        ])

    return np.array(feats)

F_tr=behavior_features(train_seqs)
F_va=behavior_features(val_seqs)
F_te=behavior_features(test_seqs)

scaler=StandardScaler()

F_tr=scaler.fit_transform(F_tr)
F_va=scaler.transform(F_va)
F_te=scaler.transform(F_te)

F_tr=torch.FloatTensor(F_tr)
F_va=torch.FloatTensor(F_va)
F_te=torch.FloatTensor(F_te)

# =============================
# TARGET
# =============================

y_tr=torch.FloatTensor(Y_train[ATTRS].values)
y_va=torch.FloatTensor(Y_val[ATTRS].values)

# =============================
# DATASET
# =============================

class SeqDataset(Dataset):

    def __init__(self,X,L,F,y=None):

        self.X=X
        self.L=L
        self.F=F
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,i):

        if self.y is None:
            return self.X[i],self.L[i],self.F[i]

        return self.X[i],self.L[i],self.F[i],self.y[i]


train_dl=DataLoader(SeqDataset(X_tr,L_tr,F_tr,y_tr),batch_size=BATCH_SIZE,shuffle=True)
val_dl=DataLoader(SeqDataset(X_va,L_va,F_va,y_va),batch_size=BATCH_SIZE)
test_dl=DataLoader(SeqDataset(X_te,L_te,F_te),batch_size=BATCH_SIZE)

# =============================
# MODEL
# =============================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding=nn.Embedding(VOCAB_SIZE,128,padding_idx=0)

        self.transformer=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.gru=nn.GRU(
            128,
            256,
            batch_first=True,
            bidirectional=True
        )

        self.fc_aux=nn.Sequential(
            nn.Linear(6,64),
            nn.GELU()
        )

        self.head=nn.Sequential(
            nn.Linear(256*2+128+64,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.GELU(),
            nn.Linear(128,6)
        )

    def forward(self,x,l,aux):

        emb=self.embedding(x)

        t=self.transformer(emb)

        packed=nn.utils.rnn.pack_padded_sequence(
            t,l.cpu(),batch_first=True,enforce_sorted=False
        )

        g,_=self.gru(packed)

        g,_=nn.utils.rnn.pad_packed_sequence(g,batch_first=True)

        g=g.mean(1)

        first=emb[:,0,:]

        aux=self.fc_aux(aux)

        z=torch.cat([g,first,aux],dim=1)

        out=self.head(z)

        return out


model=HybridModel().to(DEVICE)

# =============================
# LOSS
# =============================

def weighted_l2(pred,target):

    diff=(target-pred)/M

    loss=W*diff**2

    return loss.mean()

optimizer=torch.optim.AdamW(model.parameters(),lr=LR)

# =============================
# TRAIN
# =============================

best=999

for epoch in range(EPOCHS):

    model.train()

    for seq,l,aux,y in train_dl:

        seq=seq.to(DEVICE)
        l=l.to(DEVICE)
        aux=aux.to(DEVICE)
        y=y.to(DEVICE)

        optimizer.zero_grad()

        pred=model(seq,l,aux)

        loss=weighted_l2(pred,y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optimizer.step()

    # validation
    model.eval()

    preds=[]
    trues=[]

    with torch.no_grad():

        for seq,l,aux,y in val_dl:

            seq=seq.to(DEVICE)
            l=l.to(DEVICE)
            aux=aux.to(DEVICE)

            p=model(seq,l,aux)

            preds.append(p.cpu())
            trues.append(y)

    preds=torch.cat(preds).numpy()
    trues=torch.cat(trues).numpy()

    score=np.mean(W.cpu().numpy()*((trues-preds)/M.cpu().numpy())**2)

    print("epoch",epoch,"val score",score)

    if score<best:
        best=score
        torch.save(model.state_dict(),"best_model.pt")

print("BEST",best)

# =============================
# TEST PREDICT
# =============================

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

preds=[]

with torch.no_grad():

    for seq,l,aux in test_dl:

        seq=seq.to(DEVICE)
        l=l.to(DEVICE)
        aux=aux.to(DEVICE)

        p=model(seq,l,aux)

        preds.append(p.cpu())

preds=torch.cat(preds).numpy()

ranges=np.array([
[1,12],
[1,31],
[1,99],
[1,12],
[1,31],
[1,99]
])

for i in range(6):
    preds[:,i]=np.clip(preds[:,i],ranges[i,0],ranges[i,1])

preds=np.round(preds).astype(np.uint16)

submission=pd.DataFrame({"id":test_ids})

for i,a in enumerate(ATTRS):
    submission[a]=preds[:,i]

submission.to_csv("submission.csv",index=False)

print("submission saved")