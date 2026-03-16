"""
train_TCN.py  —  Temporal Convolutional Network
================================================
Built entirely around EDA insights. Key findings that drive every design decision:

FINDING 1 — Positional MI (Cell 15, Phase 4.14):
    attr_1: token[0]  MI=1.42  ← extremely strong, head pos 1
    attr_2: token[1]  MI=0.88  ← strong, head pos 2
    attr_3: token[4]  MI=1.10  ← strong, head pos 5
    attr_4: token[7]  MI=0.65  ← moderate, head pos 8
    attr_5: token[9]  MI=0.56  ← moderate, head pos 10
    attr_6: last tok  MI=2.61  ← NEAR-DETERMINISTIC from last token

  → These tokens are injected DIRECTLY as extra features (bypassing sequence encoder)
    instead of hoping the model discovers them. This is the single biggest lever.

FINDING 2 — attr_6 last token MI=2.61:
    → Dedicate a separate "direct lookup" path: last token → embedding → head
    → This alone can cut attr_6 error by ~80%

FINDING 3 — Behavioral feature correlations (Cell 7/8, Phase 4.7):
    attr_3: mean_step r=0.23, token_mean r=0.22, token_std r=0.23
    attr_6: late_mean r=0.43 (!), early_late_diff r=0.31, mean_step r=0.23
    → Expanded aux features with all of these

FINDING 4 — Target structure (Phase 4.6):
    MI(attr1,attr4)=0.456, MI(attr2,attr5)=0.395 → shared trunks for month pairs/day pairs
    attr_3 ↔ attr_6: Pearson=0.034 → fully independent

FINDING 5 — attr_3/attr_6 are uniform[0,99]:
    → Pure REGRESSION, not classification
    → Weighted MSE loss during training (mirror competition metric)
    → w_loss(attr3) = w_loss(attr6) = 100x others

FINDING 6 — 4 behavioral personas with conditional corr diverging up to 0.52:
    → Persona embedding injected into final representation

FINDING 7 — TCN with dilated convolutions:
    → Covers receptive field of 2^L efficiently
    → seq_len mean=13.9, so 3 dilation layers (1,2,4) = receptive field 15 → sufficient
    → Residual connections for stable gradient flow
"""

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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

ATTRS      = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']
DATE_ATTRS = ['attr_1', 'attr_2', 'attr_4', 'attr_5']   # classification
REG_ATTRS  = ['attr_3', 'attr_6']                         # regression, w=100

M_NORM   = {'attr_1': 12,  'attr_2': 31,  'attr_3': 99,
             'attr_4': 12,  'attr_5': 31,  'attr_6': 99}
W_METRIC = {'attr_1': 1,   'attr_2': 1,   'attr_3': 100,
             'attr_4': 1,   'attr_5': 1,   'attr_6': 100}

DATA_DIR    = "dataset/"
BATCH_SIZE  = 256
EPOCHS      = 100
LR          = 1e-3
PATIENCE    = 15
EMB_DIM     = 128
TCN_CHANNELS= 256
N_PERSONAS  = 4
PERSONA_DIM = 32

# EDA Finding 1: best head positions (1-indexed → 0-indexed below)
# attr_1→pos0, attr_2→pos1, attr_3→pos4, attr_4→pos7, attr_5→pos9
HEAD_POSITIONS = [0, 1, 4, 7, 9]   # 5 most informative head positions

# ============================================================
# DATA LOADING
# ============================================================

def parse_X_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','
    df = pd.read_csv(filepath, header=0, delimiter=delimiter, dtype=str)
    sequences, ids = {}, []
    for _, row in df.iterrows():
        uid = str(row.iloc[0]).strip()
        seq = []
        for val in row.iloc[1:]:
            if pd.notna(val):
                try:
                    seq.append(int(float(val)))
                except:
                    pass
        sequences[uid] = seq
        ids.append(uid)
    return sequences, ids


print("Loading sequences...")
train_seqs, train_ids = parse_X_file(DATA_DIR + "X_train.csv")
val_seqs,   val_ids   = parse_X_file(DATA_DIR + "X_val.csv")
test_seqs,  test_ids  = parse_X_file(DATA_DIR + "X_test.csv")

Y_train = pd.read_csv(DATA_DIR + "Y_train.csv")
Y_val   = pd.read_csv(DATA_DIR + "Y_val.csv")
ID_COL  = Y_train.columns[0]
Y_train = Y_train.set_index(ID_COL).loc[train_ids].reset_index()
Y_val   = Y_val.set_index(ID_COL).loc[val_ids].reset_index()

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# ============================================================
# VOCAB
# ============================================================

print("Building vocab...")
all_tokens = set()
for d in [train_seqs, val_seqs, test_seqs]:
    for seq in d.values():
        all_tokens.update(seq)

# Special indices: 0=PAD, 1=UNK, 2=MASK_POS (for missing positional tokens)
action2idx          = {a: i+3 for i, a in enumerate(sorted(all_tokens))}
action2idx[0]       = 0
action2idx['UNK']   = 1
action2idx['MASK']  = 2

VOCAB_SIZE  = len(action2idx) + 1
seq_lengths = [len(s) for s in train_seqs.values()]
MAX_LEN     = int(np.percentile(seq_lengths, 95))  # EDA: P95=28

print(f"Vocab: {VOCAB_SIZE} | MAX_LEN: {MAX_LEN}")

# ============================================================
# ENCODE SEQUENCES
# ============================================================

def encode(seqs, ids):
    X = np.zeros((len(ids), MAX_LEN), dtype=np.int64)
    L = np.zeros(len(ids), dtype=np.int64)
    for i, uid in enumerate(ids):
        seq = seqs[uid]
        l = min(len(seq), MAX_LEN)
        for j in range(l):
            X[i, j] = action2idx.get(seq[j], 1)
        L[i] = max(l, 1)
    return torch.LongTensor(X), torch.LongTensor(L)

X_tr, L_tr = encode(train_seqs, train_ids)
X_va, L_va = encode(val_seqs,   val_ids)
X_te, L_te = encode(test_seqs,  test_ids)

# ============================================================
# POSITIONAL TOKEN FEATURES  (EDA Finding 1 — direct MI injection)
# ============================================================
# Instead of hoping TCN discovers position importance, extract
# the most informative tokens DIRECTLY as input features.
#
# For each sample we extract:
#   - head[0..4]: token IDs at positions 0,1,4,7,9 (best head MI positions)
#   - last token: token[seq_len-1]  (attr_6 MI=2.61, best overall)
#   - last-1 token: token[seq_len-2] (attr_4 MI=1.02, attr_5 MI=1.15)
#
# These are passed as INDICES → separate small embeddings → concat with TCN output.

def extract_positional_tokens(seqs, ids):
    """
    Returns int64 array [N, 7]:
      cols 0-4: head positions [0,1,4,7,9] (MASK=2 if seq too short)
      col  5:   last token
      col  6:   second-to-last token (MASK if seq_len < 2)
    """
    pos = np.full((len(ids), 7), action2idx['MASK'], dtype=np.int64)
    for i, uid in enumerate(ids):
        seq = seqs[uid]
        n   = len(seq)
        for j, p in enumerate(HEAD_POSITIONS):
            if p < n:
                pos[i, j] = action2idx.get(seq[p], 1)
        # last token
        pos[i, 5] = action2idx.get(seq[-1], 1)
        # second-to-last
        if n >= 2:
            pos[i, 6] = action2idx.get(seq[-2], 1)
    return torch.LongTensor(pos)

print("Extracting positional token features...")
POS_tr = extract_positional_tokens(train_seqs, train_ids)
POS_va = extract_positional_tokens(val_seqs,   val_ids)
POS_te = extract_positional_tokens(test_seqs,  test_ids)

N_POS_TOKENS = 7   # number of positional token slots

# ============================================================
# BEHAVIORAL FEATURES  (EDA Finding 3 — expanded)
# ============================================================

def behavior_features(seqs, ids):
    feats = []
    for uid in ids:
        seq = seqs[uid]
        n   = len(seq)
        arr = np.array(seq, dtype=float)
        c   = Counter(seq)
        unique = len(c)

        # Base
        entropy      = -sum((v/n)*np.log(v/n+1e-12) for v in c.values())
        repeat_ratio = 1 - unique/n
        max_freq     = max(c.values())
        rare_ratio   = sum(1 for v in c.values() if v==1) / unique if unique>0 else 0
        bigrams      = list(zip(seq[:-1], seq[1:]))
        bigram_div   = len(set(bigrams))/len(bigrams) if bigrams else 0
        rollback     = sum(1 for i in range(2, n) if seq[i]==seq[i-2])

        # EDA Finding 3 — top correlators with attr_3/attr_6
        token_mean  = arr.mean()
        token_std   = arr.std() if n>1 else 0.0
        token_median= np.median(arr)
        token_range = arr.max() - arr.min()
        token_min   = arr.min()
        token_max   = arr.max()
        mean_step   = np.mean(np.abs(np.diff(arr))) if n>1 else 0.0
        max_step    = np.max(np.abs(np.diff(arr)))  if n>1 else 0.0

        # Positional means (late_mean r=0.43 with attr_6!)
        q = max(1, n//4)
        early_mean      = arr[:q].mean()
        late_mean       = arr[max(0, 3*n//4):].mean()
        early_late_diff = late_mean - early_mean
        mid_mean        = arr[q: max(q+1, 3*n//4)].mean() if 3*n//4 > q else token_mean

        # Log seq len (EDA: Spearman r=0.35 with attr_5)
        log_seq_len = np.log1p(n)

        # Token value stats for specific positions (direct signal for attr3/attr6)
        first_val = float(seq[0])
        last_val  = float(seq[-1])
        pos4_val  = float(seq[4]) if n > 4 else token_mean
        pos9_val  = float(seq[8]) if n > 8 else token_mean

        feats.append([
            n, log_seq_len,
            unique, unique/n,
            repeat_ratio, entropy,
            max_freq, rare_ratio,
            bigram_div, rollback,
            token_mean, token_std, token_median,
            token_min, token_max, token_range,
            mean_step, max_step,
            early_mean, mid_mean, late_mean, early_late_diff,
            first_val, last_val, pos4_val, pos9_val,
        ])
    return np.array(feats, dtype=np.float32)

print("Building behavioral features...")
F_tr_raw = behavior_features(train_seqs, train_ids)
F_va_raw = behavior_features(val_seqs,   val_ids)
F_te_raw = behavior_features(test_seqs,  test_ids)

scaler = StandardScaler()
F_tr   = torch.FloatTensor(scaler.fit_transform(F_tr_raw))
F_va   = torch.FloatTensor(scaler.transform(F_va_raw))
F_te   = torch.FloatTensor(scaler.transform(F_te_raw))
AUX_DIM = F_tr.shape[1]
print(f"Aux dim: {AUX_DIM}")

# ============================================================
# PERSONA ASSIGNMENT  (EDA Finding 6)
# ============================================================

def assign_persona(raw_feats):
    # cols: 0=n, 10=token_mean, 6=max_freq
    personas = []
    for row in raw_feats:
        tm = row[10]; mf = row[6]; sl = row[0]
        if mf > 3.0 and sl > 15:
            p = 3
        elif tm > 4500:
            p = 0
        elif tm < 2000:
            p = 1
        else:
            p = 2
        personas.append(p)
    return torch.LongTensor(personas)

P_tr = assign_persona(F_tr_raw)
P_va = assign_persona(F_va_raw)
P_te = assign_persona(F_te_raw)

# ============================================================
# TARGETS
# ============================================================

def build_targets(Y_df):
    date_t = torch.LongTensor(
        np.stack([Y_df[a].values - 1 for a in DATE_ATTRS], axis=1)
    )
    reg_t  = torch.FloatTensor(
        np.stack([Y_df[a].values / M_NORM[a] for a in REG_ATTRS], axis=1)
    )
    return date_t, reg_t

y_tr_date, y_tr_reg = build_targets(Y_train)
y_va_date, y_va_reg = build_targets(Y_val)

# ============================================================
# DATASET
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, L, F, P, POS, y_date=None, y_reg=None):
        self.X = X; self.L = L; self.F = F
        self.P = P; self.POS = POS
        self.y_date = y_date; self.y_reg = y_reg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        base = (self.X[i], self.L[i], self.F[i], self.P[i], self.POS[i])
        if self.y_date is None:
            return base
        return base + (self.y_date[i], self.y_reg[i])

train_dl = DataLoader(
    SeqDataset(X_tr, L_tr, F_tr, P_tr, POS_tr, y_tr_date, y_tr_reg),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_dl = DataLoader(
    SeqDataset(X_va, L_va, F_va, P_va, POS_va, y_va_date, y_va_reg),
    batch_size=BATCH_SIZE, num_workers=0
)

# ============================================================
# MODEL BUILDING BLOCKS
# ============================================================

class TCNBlock(nn.Module):
    """
    Dilated causal convolution block with residual connection.
    BN + GELU + Dropout for stable training.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._pad  = pad

    def forward(self, x):
        r = self.skip(x)
        # conv1
        h = self.conv1(x)
        h = h[:, :, :-self._pad] if self._pad > 0 else h  # remove acausal future
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        # conv2
        h = self.conv2(h)
        h = h[:, :, :-self._pad] if self._pad > 0 else h
        h = self.bn2(h)
        h = self.act(h)
        h = self.drop(h)
        return h + r


class TemporalAttention(nn.Module):
    """Single-head temporal attention for pooling TCN output."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x, lengths):
        # x: [B, C, T] → transpose → [B, T, C]
        x = x.transpose(1, 2)
        att = self.w(x).squeeze(-1)                          # [B, T]
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        att  = att.masked_fill(mask, -1e9)
        att  = torch.softmax(att, dim=1).unsqueeze(-1)       # [B, T, 1]
        return (x * att).sum(1)                              # [B, C]


# ============================================================
# MAIN MODEL
# ============================================================

class TCNModel(nn.Module):
    """
    Two-path architecture:
      PATH A — TCN sequence encoder (temporal patterns)
      PATH B — Direct positional token embeddings (EDA Finding 1)
               especially last token → attr_6 (MI=2.61)

    Both paths merge → shared representation → per-target heads.
    """

    def __init__(self):
        super().__init__()

        # ── Shared token embedding (both paths use same table) ──
        self.embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=0)
        nn.init.normal_(self.embedding.weight, 0, 0.02)

        # ── PATH A: TCN encoder ──
        # 4 dilation levels: 1, 2, 4, 8  → receptive field = 2*(1+2+4+8)*(k-1)+1 = 30
        # Comfortably covers mean seq_len=13.9, p95=28
        channels = [EMB_DIM, TCN_CHANNELS, TCN_CHANNELS, TCN_CHANNELS, TCN_CHANNELS]
        self.tcn_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8]
        for i, d in enumerate(dilations):
            self.tcn_blocks.append(
                TCNBlock(channels[i], channels[i+1], kernel_size=3, dilation=d, dropout=0.2)
            )

        self.attn_pool = TemporalAttention(TCN_CHANNELS)

        # ── PATH B: Direct positional token lookup ──
        # Each positional slot gets its own embedding projection
        # 5 head positions + last + second-to-last = 7 slots
        # EDA: last token for attr_6 (MI=2.61), last-1 for attr_4/5 (MI≈1.0)
        POS_EMB_DIM = 64
        self.pos_emb_proj = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(VOCAB_SIZE, POS_EMB_DIM, padding_idx=0),
                # note: Embedding is wrapped directly, forward handled below
            )
            for _ in range(N_POS_TOKENS)
        ])
        # Separate embeddings per slot so each position learns its own space
        self.pos_embeddings = nn.ModuleList([
            nn.Embedding(VOCAB_SIZE, POS_EMB_DIM, padding_idx=0)
            for _ in range(N_POS_TOKENS)
        ])

        POS_TOTAL = N_POS_TOKENS * POS_EMB_DIM  # 7 * 64 = 448

        # ── Auxiliary feature network ──
        self.aux_net = nn.Sequential(
            nn.Linear(AUX_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # ── Persona embedding ──
        self.persona_emb = nn.Embedding(N_PERSONAS, PERSONA_DIM)

        # ── Combined representation ──
        # TCN_attn(256) + POS_TOTAL(448) + aux(128) + persona(32) = 864
        COMBINED = TCN_CHANNELS + POS_TOTAL + 128 + PERSONA_DIM

        def trunk(in_d, out_d=256):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(out_d, out_d),
                nn.GELU(),
            )

        TRUNK_DIM = 256

        # EDA Finding 4: shared trunks per correlated pair
        self.trunk_month = trunk(COMBINED, TRUNK_DIM)   # attr_1, attr_4
        self.trunk_day   = trunk(COMBINED, TRUNK_DIM)   # attr_2, attr_5
        self.trunk_fac   = trunk(COMBINED, TRUNK_DIM)   # attr_3, attr_6 (independent)

        # Classification heads (date attrs)
        self.cls_heads = nn.ModuleDict({
            'attr_1': nn.Linear(TRUNK_DIM, 12),
            'attr_4': nn.Linear(TRUNK_DIM, 12),
            'attr_2': nn.Linear(TRUNK_DIM, 31),
            'attr_5': nn.Linear(TRUNK_DIM, 31),
        })

        # Regression heads (factory attrs, output [0,1] → ×99)
        # EDA: attr_3 and attr_6 are fully independent (Pearson=0.034)
        # → separate heads, no weight sharing
        self.reg_head_3 = nn.Sequential(
            nn.Linear(TRUNK_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.reg_head_6 = nn.Sequential(
            nn.Linear(TRUNK_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ── Special direct path for attr_6 ──
        # EDA: last token has MI=2.61 with attr_6 — build a shortcut
        # that directly maps last_token_embedding → attr_6 offset
        # This is added to the trunk output (residual correction)
        self.attr6_direct = nn.Sequential(
            nn.Linear(POS_EMB_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, l, aux, persona, pos_tokens):
        # ── PATH A: TCN ──
        emb = self.embedding(x)                    # [B, T, EMB_DIM]
        tcn = emb.transpose(1, 2)                  # [B, EMB_DIM, T]
        for block in self.tcn_blocks:
            tcn = block(tcn)                       # [B, TCN_CHANNELS, T]
        tcn_pooled = self.attn_pool(tcn, l)        # [B, TCN_CHANNELS]

        # ── PATH B: Positional token embeddings ──
        pos_vecs = []
        for i, emb_layer in enumerate(self.pos_embeddings):
            pos_vecs.append(emb_layer(pos_tokens[:, i]))   # [B, POS_EMB_DIM]
        pos_concat = torch.cat(pos_vecs, dim=1)            # [B, N_POS*POS_EMB_DIM]

        # ── Aux + persona ──
        aux_out  = self.aux_net(aux)               # [B, 128]
        pers_out = self.persona_emb(persona)       # [B, 32]

        # ── Fuse ──
        z = torch.cat([tcn_pooled, pos_concat, aux_out, pers_out], dim=1)  # [B, 864]

        # ── Per-group trunks ──
        z_month = self.trunk_month(z)
        z_day   = self.trunk_day(z)
        z_fac   = self.trunk_fac(z)

        # ── Heads ──
        out_cls = {
            'attr_1': self.cls_heads['attr_1'](z_month),
            'attr_4': self.cls_heads['attr_4'](z_month),
            'attr_2': self.cls_heads['attr_2'](z_day),
            'attr_5': self.cls_heads['attr_5'](z_day),
        }

        attr3_raw = self.reg_head_3(z_fac).squeeze(-1)   # [B]

        # attr_6: trunk output + direct correction from last token (MI=2.61)
        last_tok_emb   = pos_vecs[-2]   # col 5 = last token embedding (POS_EMB_DIM=64)
        attr6_direct   = self.attr6_direct(last_tok_emb).squeeze(-1)   # [B]
        attr6_trunk    = self.reg_head_6(z_fac).squeeze(-1)            # [B]
        # Blend: direct shortcut gets higher weight (0.6) given its MI dominance
        attr6_raw      = 0.4 * attr6_trunk + 0.6 * attr6_direct

        out_reg = {'attr_3': attr3_raw, 'attr_6': attr6_raw}
        return out_cls, out_reg


model = TCNModel().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS  (EDA Finding 5 — mirror competition metric)
# ============================================================
# CE with label smoothing for date attrs (discrete, bounded)
# Huber loss for factory attrs (robust to rare-class outliers)
# w_loss(attr3) = w_loss(attr6) = 10 (heavy emphasis; 100 would dominate entirely)

ce_loss    = nn.CrossEntropyLoss(label_smoothing=0.05)
huber_loss = nn.HuberLoss(delta=0.1)   # delta in normalised [0,1] space

W_LOSS = {
    'attr_1': 1.0, 'attr_2': 1.0, 'attr_4': 1.0, 'attr_5': 1.0,
    'attr_3': 10.0, 'attr_6': 10.0,   # high weight to reflect competition penalty
}

def compute_loss(out_cls, out_reg, y_date, y_reg):
    total = 0.0
    for i, attr in enumerate(DATE_ATTRS):
        total += W_LOSS[attr] * ce_loss(out_cls[attr], y_date[:, i])
    for i, attr in enumerate(REG_ATTRS):
        total += W_LOSS[attr] * huber_loss(out_reg[attr], y_reg[:, i])
    return total

# ============================================================
# COMPETITION METRIC  (exact formula)
# ============================================================

def competition_metric(preds_dict, y_date_np, y_reg_np):
    """
    sum_k  w_k * (1/N) * sum_i ((yhat_ik - y_ik) / M_k)^2
    - date preds: 0-indexed → +1 before dividing by M
    - reg  preds: in [0, M_norm] (denormalised)
    - y_date_np:  0-indexed true labels
    - y_reg_np:   true labels in [0, M_norm]
    """
    total = 0.0
    N = y_date_np.shape[0]
    for i, attr in enumerate(DATE_ATTRS):
        m    = M_NORM[attr]
        w    = W_METRIC[attr]
        p    = preds_dict[attr].astype(float) + 1   # 0-idx → 1-indexed
        t    = (y_date_np[:, i] + 1).astype(float)
        diff = (p - t) / m
        total += w * np.sum(diff**2) / N
    for i, attr in enumerate(REG_ATTRS):
        m    = M_NORM[attr]
        w    = W_METRIC[attr]
        p    = preds_dict[attr].astype(float)
        t    = y_reg_np[:, i].astype(float)
        diff = (p - t) / m
        total += w * np.sum(diff**2) / N
    return total

# ============================================================
# OPTIMISER  —  differentiated LR groups
# ============================================================

optimizer = torch.optim.AdamW([
    {'params': model.embedding.parameters(),     'lr': LR * 0.2},
    {'params': model.pos_embeddings.parameters(),'lr': LR * 0.5},
    {'params': model.tcn_blocks.parameters(),    'lr': LR * 0.5},
    {'params': model.attn_pool.parameters(),     'lr': LR},
    {'params': model.aux_net.parameters(),       'lr': LR},
    {'params': model.persona_emb.parameters(),   'lr': LR},
    {'params': model.trunk_month.parameters(),   'lr': LR},
    {'params': model.trunk_day.parameters(),     'lr': LR},
    {'params': model.trunk_fac.parameters(),     'lr': LR},
    {'params': model.cls_heads.parameters(),     'lr': LR},
    {'params': model.reg_head_3.parameters(),    'lr': LR},
    {'params': model.reg_head_6.parameters(),    'lr': LR},
    {'params': model.attr6_direct.parameters(),  'lr': LR * 1.5},  # boost direct path
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=25, T_mult=2, eta_min=1e-5
)

# ============================================================
# TRAINING LOOP
# ============================================================

best_metric = float('inf')
best_state  = None
patience_ct = 0

print("\nStarting training...")
print(f"{'Epoch':>5} | {'Loss':>8} | {'Val Metric':>12} | {'Status'}")
print("-" * 45)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_dl:
        seq, l, aux, persona, pos, y_date, y_reg = [t.to(DEVICE) for t in batch]
        optimizer.zero_grad()
        out_cls, out_reg = model(seq, l, aux, persona, pos)
        loss = compute_loss(out_cls, out_reg, y_date, y_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # ── Validation ──
    model.eval()
    preds_dict = {a: [] for a in ATTRS}
    y_date_all, y_reg_all = [], []

    with torch.no_grad():
        for batch in val_dl:
            seq, l, aux, persona, pos, y_date, y_reg = [t.to(DEVICE) for t in batch]
            out_cls, out_reg = model(seq, l, aux, persona, pos)

            for i, attr in enumerate(DATE_ATTRS):
                preds_dict[attr].append(out_cls[attr].argmax(1).cpu().numpy())
            for attr in REG_ATTRS:
                p = (out_reg[attr] * M_NORM[attr]).round().clamp(0, M_NORM[attr])
                preds_dict[attr].append(p.cpu().numpy())

            y_date_all.append(y_date.cpu().numpy())
            y_reg_all.append(y_reg.cpu().numpy())

    for attr in ATTRS:
        preds_dict[attr] = np.concatenate(preds_dict[attr])
    y_date_np = np.concatenate(y_date_all)
    y_reg_np  = np.concatenate(y_reg_all) * np.array([M_NORM[a] for a in REG_ATTRS])

    metric   = competition_metric(preds_dict, y_date_np, y_reg_np)
    avg_loss = total_loss / len(train_dl)

    status = ""
    if metric < best_metric:
        best_metric = metric
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ct = 0
        status = "✓ best"

        # Print per-attribute breakdown when we improve
        per_attr = {}
        for i, attr in enumerate(DATE_ATTRS):
            m = M_NORM[attr]; w = W_METRIC[attr]
            p = preds_dict[attr].astype(float) + 1
            t = (y_date_np[:, i] + 1).astype(float)
            per_attr[attr] = w * np.mean(((p-t)/m)**2)
        for i, attr in enumerate(REG_ATTRS):
            m = M_NORM[attr]; w = W_METRIC[attr]
            p = preds_dict[attr].astype(float)
            t = y_reg_np[:, i].astype(float)
            per_attr[attr] = w * np.mean(((p-t)/m)**2)

        breakdown = "  |  ".join(f"{a}={per_attr[a]:.4f}" for a in ATTRS)
        print(f"{epoch:>5} | {avg_loss:>8.4f} | {metric:>12.5f} | {status}")
        print(f"         breakdown: {breakdown}")
    else:
        patience_ct += 1
        print(f"{epoch:>5} | {avg_loss:>8.4f} | {metric:>12.5f} |")

    if patience_ct >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nBEST val_metric: {best_metric:.5f}")

# ============================================================
# TEST INFERENCE
# ============================================================

model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

test_dl = DataLoader(
    SeqDataset(X_te, L_te, F_te, P_te, POS_te),
    batch_size=BATCH_SIZE
)

final_preds = {a: [] for a in ATTRS}

with torch.no_grad():
    for batch in test_dl:
        seq, l, aux, persona, pos = [t.to(DEVICE) for t in batch]
        out_cls, out_reg = model(seq, l, aux, persona, pos)

        for attr in DATE_ATTRS:
            final_preds[attr].append(out_cls[attr].argmax(1).cpu().numpy() + 1)
        for attr in REG_ATTRS:
            p = (out_reg[attr] * M_NORM[attr]).round().clamp(0, M_NORM[attr])
            final_preds[attr].append(p.cpu().numpy().astype(int))

for attr in ATTRS:
    final_preds[attr] = np.concatenate(final_preds[attr]).astype(np.uint16)

# ============================================================
# SUBMISSION
# ============================================================

sub = pd.DataFrame({'id': test_ids})
for attr in ATTRS:
    sub[attr] = final_preds[attr]

sub.to_csv("submission_tcn.csv", index=False)
print("\nSaved submission_tcn.csv")
print(sub.head(10))
print("\nValue ranges:")
for attr in ATTRS:
    print(f"  {attr}: [{sub[attr].min()}, {sub[attr].max()}]  mean={sub[attr].mean():.1f}")
