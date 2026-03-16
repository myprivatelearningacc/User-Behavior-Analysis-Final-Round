import os
import math
import copy
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "dataset"
ATTRS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
MAX_VAL = [12, 31, 99, 12, 31, 99]
NCLS = [m + 1 for m in MAX_VAL]
W = np.array([1, 1, 100, 1, 1, 100], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 1337, 2026]
BATCH_SIZE = 384
EPOCHS = 28
LR = 2.5e-4
WD = 1e-4
PATIENCE = 6
EMB_DIM = 192
HID_DIM = 256
NUM_LAYERS = 2
HEAD_LEN = 20
TAIL_LEN = 20
FULL_LEN = 40
TOKEN_DROP = 0.06

# ============================================================
# UTILS
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_mse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    return float(np.average(mse, weights=W))


def weighted_mse_torch(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean(dim=0)
    weights = torch.tensor(W, device=y_true.device, dtype=y_true.dtype)
    return (mse * weights).sum() / weights.sum()


def parse_X(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).values
    seqs = {}
    for i, row in enumerate(df.iloc[:, 1:].values):
        seq = [int(x) for x in row if not pd.isna(x)]
        seqs[ids[i]] = seq
    return seqs, ids


def read_split():
    train_seqs, train_ids = parse_X(os.path.join(DATA_DIR, "X_train.csv"))
    val_seqs, val_ids = parse_X(os.path.join(DATA_DIR, "X_val.csv"))
    test_seqs, test_ids = parse_X(os.path.join(DATA_DIR, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(DATA_DIR, "Y_train.csv"))
    y_val = pd.read_csv(os.path.join(DATA_DIR, "Y_val.csv"))
    y_train = y_train.set_index(y_train.columns[0]).loc[train_ids].reset_index()
    y_val = y_val.set_index(y_val.columns[0]).loc[val_ids].reset_index()
    return train_seqs, train_ids, y_train, val_seqs, val_ids, y_val, test_seqs, test_ids


# ============================================================
# EDA-ALIGNED FEATURES
# ============================================================
def seq_entropy(seq):
    n = len(seq)
    if n == 0:
        return 0.0
    c = Counter(seq)
    p = np.array(list(c.values()), dtype=np.float64) / n
    return float(-(p * np.log(p + 1e-12)).sum())


def transition_entropy(seq):
    if len(seq) < 2:
        return 0.0
    pairs = list(zip(seq[:-1], seq[1:]))
    c = Counter(pairs)
    p = np.array(list(c.values()), dtype=np.float64) / len(pairs)
    return float(-(p * np.log(p + 1e-12)).sum())


def build_features(seqs, ids, train_vocab=None, train_transitions=None):
    feats = []
    for uid in ids:
        seq = seqs[uid]
        n = len(seq)
        arr = np.array(seq, dtype=np.float32) if n else np.zeros(1, dtype=np.float32)
        c = Counter(seq)
        unique = len(c)
        uniq_ratio = unique / max(n, 1)
        ent = seq_entropy(seq)
        maxfreq = max(c.values()) if n else 0
        mode_ratio = maxfreq / max(n, 1)
        repeat_ratio = 1.0 - uniq_ratio
        singleton_ratio = sum(v == 1 for v in c.values()) / max(unique, 1)

        diffs = np.diff(arr) if n > 1 else np.array([0.0], dtype=np.float32)
        absdiff = np.abs(diffs)
        signs = np.sign(diffs)
        turns = np.sum(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else 0

        bigrams = list(zip(seq[:-1], seq[1:]))
        tri = list(zip(seq[:-2], seq[1:-1], seq[2:]))
        bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
        trigram_div = len(set(tri)) / max(len(tri), 1)
        trans_ent = transition_entropy(seq)
        self_loop_ratio = sum(a == b for a, b in bigrams) / max(len(bigrams), 1)
        rollback_ratio = sum(seq[i] == seq[i - 2] for i in range(2, n)) / max(n - 2, 1)

        q1 = seq[: max(1, n // 4)]
        q2 = seq[max(1, n // 4): max(2, n // 2)] or q1
        q3 = seq[max(2, n // 2): max(3, 3 * n // 4)] or q2
        q4 = seq[max(3, 3 * n // 4):] or q3

        early_mean = float(np.mean(q1)) if q1 else 0.0
        mid_mean = float(np.mean(q2 + q3)) if (q2 or q3) else early_mean
        late_mean = float(np.mean(q4)) if q4 else 0.0
        early_ent = seq_entropy(q1)
        late_ent = seq_entropy(q4)

        first = seq[0] if n else -1
        second = seq[1] if n > 1 else -1
        third = seq[2] if n > 2 else -1
        fourth = seq[3] if n > 3 else -1
        last = seq[-1] if n else -1
        last2 = seq[-2] if n > 1 else -1
        last3 = seq[-3] if n > 2 else -1
        last4 = seq[-4] if n > 3 else -1

        unseen_vocab_rate = 0.0
        unseen_trans_rate = 0.0
        if train_vocab is not None:
            unseen_vocab_rate = sum(tok not in train_vocab for tok in seq) / max(n, 1)
        if train_transitions is not None and len(bigrams) > 0:
            unseen_trans_rate = sum(bg not in train_transitions for bg in bigrams) / len(bigrams)

        feats.append([
            n,
            math.log1p(n),
            unique,
            uniq_ratio,
            ent,
            maxfreq,
            mode_ratio,
            repeat_ratio,
            singleton_ratio,
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(arr.max() - arr.min()),
            float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            float(diffs.mean()),
            float(absdiff.mean()),
            float(absdiff.std()),
            float(absdiff.max()),
            float(turns / max(n - 2, 1)),
            bigram_div,
            trigram_div,
            trans_ent,
            self_loop_ratio,
            rollback_ratio,
            early_mean,
            mid_mean,
            late_mean,
            late_mean - early_mean,
            early_ent,
            late_ent,
            late_ent - early_ent,
            first,
            second,
            third,
            fourth,
            last,
            last2,
            last3,
            last4,
            unseen_vocab_rate,
            unseen_trans_rate,
        ])
    return np.array(feats, dtype=np.float32)


# ============================================================
# ENCODING
# ============================================================
def build_vocab(all_seq_dicts):
    tokens = set()
    for seqs in all_seq_dicts:
        for seq in seqs.values():
            tokens.update(seq)
    token2idx = {tok: i + 2 for i, tok in enumerate(sorted(tokens))}
    token2idx["PAD"] = 0
    token2idx["UNK"] = 1
    return token2idx


def encode_window(seq, token2idx, max_len, mode="headtail"):
    if len(seq) == 0:
        return [0], 1

    if mode == "head":
        use = seq[:max_len]
    elif mode == "tail":
        use = seq[-max_len:]
    else:
        if len(seq) <= max_len:
            use = seq
        else:
            h = max_len // 2
            t = max_len - h
            use = seq[:h] + seq[-t:]

    ids = [token2idx.get(x, 1) for x in use]
    return ids, max(1, len(ids))


def pad_stack(encoded, pad=0):
    max_len = max(len(x) for x, _ in encoded)
    X = np.full((len(encoded), max_len), pad, dtype=np.int64)
    L = np.zeros(len(encoded), dtype=np.int64)
    for i, (seq, l) in enumerate(encoded):
        X[i, :l] = seq[:l]
        L[i] = l
    return X, L


class SeqDataset(Dataset):
    def __init__(self, head_X, head_L, tail_X, tail_L, full_X, full_L, feats, y=None, train=False):
        self.head_X = torch.LongTensor(head_X)
        self.head_L = torch.LongTensor(head_L)
        self.tail_X = torch.LongTensor(tail_X)
        self.tail_L = torch.LongTensor(tail_L)
        self.full_X = torch.LongTensor(full_X)
        self.full_L = torch.LongTensor(full_L)
        self.feats = torch.FloatTensor(feats)
        self.y = None if y is None else torch.LongTensor(y)
        self.train = train

    def __len__(self):
        return len(self.head_X)

    def token_dropout(self, x, l):
        if not self.train:
            return x
        x = x.clone()
        keep = torch.rand(l) > TOKEN_DROP
        x[:l][~keep] = 1
        return x

    def __getitem__(self, i):
        hx = self.token_dropout(self.head_X[i], int(self.head_L[i]))
        tx = self.token_dropout(self.tail_X[i], int(self.tail_L[i]))
        fx = self.token_dropout(self.full_X[i], int(self.full_L[i]))
        if self.y is None:
            return hx, self.head_L[i], tx, self.tail_L[i], fx, self.full_L[i], self.feats[i]
        return hx, self.head_L[i], tx, self.tail_L[i], fx, self.full_L[i], self.feats[i], self.y[i]


# ============================================================
# MODEL
# ============================================================
class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, x, lengths):
        score = self.proj(x).squeeze(-1)
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        score = score.masked_fill(mask, -1e9)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)
        return (x * attn).sum(1)


class WindowEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.drop = nn.Dropout(0.15)
        self.gru = nn.GRU(
            EMB_DIM,
            HID_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if NUM_LAYERS > 1 else 0.0,
        )
        self.pool = AttentionPool(HID_DIM * 2)

    def forward(self, x, l):
        emb = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, l.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        pooled = self.pool(out, l)
        first = emb[:, 0, :]
        last_idx = (l - 1).clamp(min=0)
        last = emb[torch.arange(x.size(0), device=x.device), last_idx]
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        return torch.cat([pooled, h_last, first, last], dim=1)


class HybridGRU(nn.Module):
    def __init__(self, vocab_size, feat_dim):
        super().__init__()
        self.head_enc = WindowEncoder(vocab_size)
        self.tail_enc = WindowEncoder(vocab_size)
        self.full_enc = WindowEncoder(vocab_size)

        enc_dim = (HID_DIM * 2 + HID_DIM * 2 + EMB_DIM + EMB_DIM)
        total_enc = enc_dim * 3

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(total_enc + 128, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, ncls),
            ) for ncls in NCLS
        ])

    def forward(self, hx, hl, tx, tl, fx, fl, feats):
        zh = self.head_enc(hx, hl)
        zt = self.tail_enc(tx, tl)
        zf = self.full_enc(fx, fl)
        ff = self.feat_mlp(feats)
        z = self.shared(torch.cat([zh, zt, zf, ff], dim=1))
        logits = [head(z) for head in self.heads]
        return logits


def expected_from_logits(logits_list):
    outs = []
    for i, logits in enumerate(logits_list):
        probs = torch.softmax(logits, dim=1)
        values = torch.arange(NCLS[i], device=logits.device, dtype=probs.dtype)[None, :]
        outs.append((probs * values).sum(1, keepdim=True))
    return torch.cat(outs, dim=1)


def loss_fn(logits_list, y):
    losses = []
    for i, logits in enumerate(logits_list):
        ce = F.cross_entropy(logits, y[:, i], label_smoothing=0.02)
        losses.append(ce * float(W[i]))
    ce_loss = sum(losses) / sum(W)
    expv = expected_from_logits(logits_list)
    mse_loss = weighted_mse_torch(y.float(), expv)
    return 0.55 * ce_loss + 0.45 * mse_loss, expv


# ============================================================
# LIGHTGBM
# ============================================================
def fit_lgbm(Xtr, ytr, Xva, yva, Xte):
    pred_va = np.zeros((len(Xva), 6), dtype=np.float32)
    pred_te = np.zeros((len(Xte), 6), dtype=np.float32)
    models = []
    for i, a in enumerate(ATTRS):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3500,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42 + i,
        )
        model.fit(
            Xtr,
            ytr[:, i],
            eval_set=[(Xva, yva[:, i])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        pred_va[:, i] = model.predict(Xva, num_iteration=model.best_iteration_)
        pred_te[:, i] = model.predict(Xte, num_iteration=model.best_iteration_)
        models.append(model)
    return pred_va, pred_te, models


# ============================================================
# DATA PREP
# ============================================================
def make_encoded(seqs, ids, token2idx):
    head = [encode_window(seqs[i], token2idx, HEAD_LEN, "head") for i in ids]
    tail = [encode_window(seqs[i], token2idx, TAIL_LEN, "tail") for i in ids]
    full = [encode_window(seqs[i], token2idx, FULL_LEN, "headtail") for i in ids]
    return (*pad_stack(head), *pad_stack(tail), *pad_stack(full))


# ============================================================
# TRAIN / EVAL
# ============================================================
def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    preds = []
    ys = []

    for batch in loader:
        if train:
            hx, hl, tx, tl, fx, fl, feat, y = batch
            y = y.to(DEVICE)
        else:
            if len(batch) == 8:
                hx, hl, tx, tl, fx, fl, feat, y = batch
                y = y.to(DEVICE)
            else:
                hx, hl, tx, tl, fx, fl, feat = batch
                y = None

        hx = hx.to(DEVICE)
        hl = hl.to(DEVICE)
        tx = tx.to(DEVICE)
        tl = tl.to(DEVICE)
        fx = fx.to(DEVICE)
        fl = fl.to(DEVICE)
        feat = feat.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(hx, hl, tx, tl, fx, fl, feat)
        pred = expected_from_logits(logits)

        if y is not None:
            loss, _ = loss_fn(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * hx.size(0)
            ys.append(y.detach().cpu().numpy())

        preds.append(pred.detach().cpu().numpy())

    pred_arr = np.vstack(preds)
    out = {"pred": pred_arr}
    if ys:
        y_arr = np.vstack(ys)
        out["y"] = y_arr
        out["loss"] = total_loss / len(loader.dataset)
        out["metric"] = weighted_mse_np(y_arr, pred_arr)
    return out


def fit_gru_seed(seed, train_ds, val_ds, test_ds, feat_dim, vocab_size):
    seed_everything(seed)
    model = HybridGRU(vocab_size, feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_state = None
    best_metric = 1e18
    best_epoch = 0
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer)
        va = run_epoch(model, val_loader, optimizer=None)
        scheduler.step(va["metric"])
        print(f"[seed={seed}] epoch={epoch:02d} train_loss={tr['loss']:.5f} val_metric={va['metric']:.6f}")
        if va["metric"] < best_metric:
            best_metric = va["metric"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    val_out = run_epoch(model, val_loader, optimizer=None)["pred"]
    test_out = run_epoch(model, test_loader, optimizer=None)["pred"]
    return {
        "state": best_state,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "val_pred": val_out,
        "test_pred": test_out,
    }


def fit_gru_full(seed, full_train_ds, test_ds, feat_dim, vocab_size, epochs):
    seed_everything(seed)
    model = HybridGRU(vocab_size, feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.train()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loader, optimizer)
        print(f"[full seed={seed}] epoch={epoch:02d} loss={tr['loss']:.5f}")
    test_out = run_epoch(model, test_loader, optimizer=None)["pred"]
    return test_out


def optimize_blend(y_val, gru_val, lgb_val):
    best_ws = []
    blended = np.zeros_like(gru_val)
    for i in range(6):
        best_w = 1.0
        best_s = 1e18
        for w in np.linspace(0.0, 1.0, 101):
            pred = w * gru_val[:, i] + (1 - w) * lgb_val[:, i]
            pred = np.clip(pred, 0, MAX_VAL[i])
            s = mean_squared_error(y_val[:, i], pred)
            if s < best_s:
                best_s = s
                best_w = w
        best_ws.append(best_w)
        blended[:, i] = best_w * gru_val[:, i] + (1 - best_w) * lgb_val[:, i]
    return np.array(best_ws, dtype=np.float32), blended


# ============================================================
# MAIN
# ============================================================
def main():
    train_seqs, train_ids, y_train_df, val_seqs, val_ids, y_val_df, test_seqs, test_ids = read_split()
    y_train = y_train_df[ATTRS].values.astype(np.int64)
    y_val = y_val_df[ATTRS].values.astype(np.int64)

    train_vocab = set(tok for seq in train_seqs.values() for tok in seq)
    train_trans = set(bg for seq in train_seqs.values() for bg in zip(seq[:-1], seq[1:]))

    F_tr = build_features(train_seqs, train_ids, train_vocab, train_trans)
    F_va = build_features(val_seqs, val_ids, train_vocab, train_trans)
    F_te = build_features(test_seqs, test_ids, train_vocab, train_trans)

    scaler = StandardScaler()
    F_tr = scaler.fit_transform(F_tr)
    F_va = scaler.transform(F_va)
    F_te = scaler.transform(F_te)

    token2idx = build_vocab([train_seqs, val_seqs, test_seqs])
    vocab_size = len(token2idx)

    tr_hx, tr_hl, tr_tx, tr_tl, tr_fx, tr_fl = make_encoded(train_seqs, train_ids, token2idx)
    va_hx, va_hl, va_tx, va_tl, va_fx, va_fl = make_encoded(val_seqs, val_ids, token2idx)
    te_hx, te_hl, te_tx, te_tl, te_fx, te_fl = make_encoded(test_seqs, test_ids, token2idx)

    train_ds = SeqDataset(tr_hx, tr_hl, tr_tx, tr_tl, tr_fx, tr_fl, F_tr, y_train, train=True)
    val_ds = SeqDataset(va_hx, va_hl, va_tx, va_tl, va_fx, va_fl, F_va, y_val, train=False)
    test_ds = SeqDataset(te_hx, te_hl, te_tx, te_tl, te_fx, te_fl, F_te, y=None, train=False)

    # LightGBM branch
    lgb_val, lgb_test, _ = fit_lgbm(F_tr, y_train.astype(np.float32), F_va, y_val.astype(np.float32), F_te)
    print("LGBM val weighted MSE:", weighted_mse_np(y_val, lgb_val))

    # GRU branch (multi-seed)
    gru_val_preds = []
    gru_test_preds = []
    best_epochs = []
    for seed in SEEDS:
        res = fit_gru_seed(seed, train_ds, val_ds, test_ds, F_tr.shape[1], vocab_size)
        print(f"Best val metric for seed {seed}: {res['best_metric']:.6f} at epoch {res['best_epoch']}")
        gru_val_preds.append(res["val_pred"])
        gru_test_preds.append(res["test_pred"])
        best_epochs.append(res["best_epoch"])

    gru_val = np.mean(gru_val_preds, axis=0)
    gru_test = np.mean(gru_test_preds, axis=0)
    print("GRU ensemble val weighted MSE:", weighted_mse_np(y_val, gru_val))

    # Blend per target
    blend_w, blend_val = optimize_blend(y_val, gru_val, lgb_val)
    blend_test = blend_w[None, :] * gru_test + (1.0 - blend_w[None, :]) * lgb_test
    print("Blend weights (GRU per target):", dict(zip(ATTRS, blend_w.tolist())))
    print("Blended val weighted MSE:", weighted_mse_np(y_val, blend_val))

    # Retrain GRU on train+val using mean best epoch
    full_ids = np.concatenate([train_ids, val_ids])
    full_y = np.vstack([y_train, y_val])
    full_seqs = {}
    full_seqs.update(train_seqs)
    full_seqs.update(val_seqs)

    F_full = scaler.fit_transform(build_features(full_seqs, full_ids, train_vocab, train_trans))
    full_hx, full_hl, full_tx, full_tl, full_fx, full_fl = make_encoded(full_seqs, full_ids, token2idx)
    full_train_ds = SeqDataset(full_hx, full_hl, full_tx, full_tl, full_fx, full_fl, F_full, full_y, train=True)

    retrain_epochs = max(4, int(round(np.mean(best_epochs))))
    print("Retrain full GRU epochs:", retrain_epochs)
    full_gru_tests = []
    for seed in SEEDS:
        pred = fit_gru_full(seed, full_train_ds, test_ds, F_full.shape[1], vocab_size, retrain_epochs)
        full_gru_tests.append(pred)
    full_gru_test = np.mean(full_gru_tests, axis=0)

    # Retrain LightGBM on full labeled data
    F_lgb_full = F_full
    F_lgb_test = scaler.transform(build_features(test_seqs, test_ids, train_vocab, train_trans))
    full_lgb_test = np.zeros((len(test_ids), 6), dtype=np.float32)
    for i, a in enumerate(ATTRS):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=2600,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=99 + i,
        )
        model.fit(F_lgb_full, full_y[:, i].astype(np.float32))
        full_lgb_test[:, i] = model.predict(F_lgb_test)

    final_test = blend_w[None, :] * full_gru_test + (1.0 - blend_w[None, :]) * full_lgb_test

    # Conservative post-processing for integer-bounded targets
    for i in range(6):
        final_test[:, i] = np.clip(final_test[:, i], 0, MAX_VAL[i])

    submission = pd.DataFrame({"id": test_ids})
    for i, a in enumerate(ATTRS):
        submission[a] = np.rint(final_test[:, i]).astype(np.uint16)
    submission.to_csv("submission_strong_hybrid.csv", index=False)
    print("Saved submission_strong_hybrid.csv")


if __name__ == "__main__":
    main()
