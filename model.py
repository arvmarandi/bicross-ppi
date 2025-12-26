import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# === Self-Attention (same as original) =======================
# =============================================================
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(device)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention



# =============================================================
# === Positionwise Feedforward (same as original) =============
# =============================================================
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))



# =============================================================
# === DecoderLayer (unchanged) ================================
# =============================================================
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.self_attn_norm = nn.LayerNorm(hid_dim)
        self.self_attn = self_attention(hid_dim, n_heads, dropout, device)

        self.cross_attn_norm = nn.LayerNorm(hid_dim)
        self.cross_attn = self_attention(hid_dim, n_heads, dropout, device)

        self.ff_norm = nn.LayerNorm(hid_dim)
        self.ff = positionwise_feedforward(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        # Self-attention
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.self_attn_norm(trg + self.dropout(_trg))

        # Cross-attention
        _trg, _ = self.cross_attn(trg, src, src, src_mask)
        trg = self.cross_attn_norm(trg + self.dropout(_trg))

        # Feed-forward
        _trg = self.ff(trg)
        trg = self.ff_norm(trg + self.dropout(_trg))

        return trg



# =============================================================
# === Decoder (unchanged structure) ===========================
# =============================================================
class Decoder(nn.Module):
    """
    Decoder unchanged structurally.
    Now works with predefined protein embeddings for both src & trg.
    """
    def __init__(self, emb_dim, hid_dim, n_layers, n_heads, pf_dim,
                 decoder_layer, self_attention, positionwise_feedforward,
                 dropout, device):
        super().__init__()

        # Project both protein embeddings to hidden dim
        self.src_proj = nn.Linear(emb_dim, hid_dim)
        self.trg_proj = nn.Linear(emb_dim, hid_dim)

        self.layers = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim,
                          self_attention, positionwise_feedforward,
                          dropout, device)
            for _ in range(n_layers)
        ])

        # Weighted pooling MLP
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg_emb, src_emb, trg_mask, src_mask):
        """
        trg_emb: protein2 embeddings   [B, L2, emb_dim]
        src_emb: protein1 embeddings   [B, L1, emb_dim]
        """
        # Project both embeddings into hidden dim
        trg = self.trg_proj(trg_emb)
        src = self.src_proj(src_emb)

        # Transformer decoder layers (unchanged)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # Weighted pooling
        norm = torch.norm(trg, dim=2)         # [B, L2]
        norm = F.softmax(norm, dim=1)         # weights
        pooled = torch.sum(trg * norm.unsqueeze(-1), dim=1)

        # Classification
        x = F.relu(self.fc_1(pooled))
        logits = self.fc_2(x)

        return logits




# =============================================================
# === PPI Predictor (modified but keeping decoder flow) =======
# =============================================================
class PPIPredictor(nn.Module):
    """
    PPI predictor for pretrained protein embeddings.
    Takes:
        protein1_emb: [B, L1, E]
        protein2_emb: [B, L2, E]
    """
    def __init__(self, decoder, device):
        super().__init__()
        self.decoder = decoder
        self.device = device

    def forward(self, protein1_emb, protein2_emb, mask1, mask2):
        """
        protein1_emb: [B, L1, E]
        protein2_emb: [B, L2, E]
        mask1: [B, L1]
        mask2: [B, L2]
        """
        logits = self.decoder(protein2_emb, protein1_emb, mask2, mask1)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIPredictorUpgrade(nn.Module):
    """
    PPI predictor built on top of your existing Decoder.

    Inputs to forward:
        p1: Tensor or dict
            - If Tensor:   [B, L1, D]   (pre-concatenated features, e.g. emb or emb+logits)
            - If dict:     {"emb": [B,L1,D1], "logits": [B,L1,D2]}
                          → will be concatenated inside.
        p2: same structure as p1, but [B, L2, *]
        p1_mask: [B, L1]   1 = valid, 0 = padding
        p2_mask: [B, L2]

    Optional upgrades (all False by default so old behavior is preserved):

        use_logits:
            If True and you pass dicts {"emb","logits"}, we concat emb+logits along last dim.

        use_attention_pooling:
            If True, use learned attention over residues instead of simple mean pooling.

        bidirectional:
            If True, run decoder in both directions:
                out12 = decoder(trg=p2, src=p1)
                out21 = decoder(trg=p1, src=p2)

        symmetric_scoring:
            If True, enforce order invariance using:
                h_sym = concat(g12 + g21, |g12 - g21|)
            where g12, g21 are pooled vectors from the two directions.
            If False, only g12 is used.

        use_bilinear_scoring:
            If True, add a bilinear term between pooled vectors:
                b = g12^T W g21   (scalar per sample)
            and feed it as an extra feature to the classifier.

    Old behavior (for backward compatibility):
        - If all flags are False:
            - Only decoder(trg=p2, src=p1) is used
            - Mean pooling over residues
            - Simple MLP classifier.

    """

    def __init__(
        self,
        decoder: nn.Module,
        hid_dim: int,
        dropout: float = 0.1,
        use_logits: bool = False,
        use_attention_pooling: bool = False,
        bidirectional: bool = False,
        symmetric_scoring: bool = False,
        use_bilinear_scoring: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.decoder = decoder
        self.hid_dim = hid_dim
        self.device = device

        # ---- upgrade 1: logits support (handled in forward by concatenation) ----
        self.use_logits = use_logits

        # ---- upgrade 5: attention pooling over residues ----
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            # simple 1-layer scoring for each residue: score_i = w^T h_i
            self.attn_proj = nn.Linear(hid_dim, 1)

        # ---- upgrade 2: bi-directional cross-attention (call decoder twice) ----
        self.bidirectional = bidirectional

        # ---- upgrade 4: symmetric scoring ----
        self.symmetric_scoring = symmetric_scoring

        # ---- upgrade 3: bilinear scoring between pooled vectors ----
        self.use_bilinear_scoring = use_bilinear_scoring
        if self.use_bilinear_scoring:
            # bilinear: g12^T W g21
            self.bilinear = nn.Linear(hid_dim, hid_dim, bias=False)

        # determine classifier input dim
        # base feature: g12 (hid_dim)
        clf_in_dim = hid_dim

        # if we use bidirectional + symmetric scoring → concat(g12+g21, |g12-g21|)
        if self.bidirectional or self.symmetric_scoring:
            # g_sym = concat(g12+g21, |g12 - g21|)
            clf_in_dim = hid_dim * 2

        # add 1 dimension for bilinear scalar if used
        if self.use_bilinear_scoring:
            clf_in_dim += 1

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    # ----------------- internal helpers -----------------

    def _merge_emb_logits(self, x):
        """
        x can be:
            - Tensor [B, L, D]
            - dict{"emb": [B,L,D1], "logits": [B,L,D2]}  (only used if self.use_logits is True)
        """
        if isinstance(x, dict):
            emb = x["emb"]
            logits = x["logits"]
            return torch.cat([emb, logits], dim=-1)
        else:
            # already a tensor (e.g., just embeddings or embeddings+logits pre-concatenated)
            return x

    def _pool(self, seq, mask):
        """
        seq:  [B, L, H]
        mask: [B, L], 1=valid, 0=pad
        Returns: [B, H]
        """
        mask = mask.float()

        if not self.use_attention_pooling:
            # simple masked mean pooling (old behavior)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (seq * mask.unsqueeze(-1)).sum(dim=1) / denom

        # attention pooling
        # raw scores: [B, L, 1]
        scores = self.attn_proj(seq).squeeze(-1)  # [B, L]

        # mask out padded positions
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax over valid positions
        attn = F.softmax(scores, dim=1)  # [B, L]
        attn = attn * mask  # just in case

        # weighted sum
        return torch.bmm(attn.unsqueeze(1), seq).squeeze(1)  # [B, H]

    # ----------------- main forward -----------------

    def forward(self, p1, p2, p1_mask, p2_mask):
        """
        p1, p2:
            - Tensor [B, L, D] OR
            - dict{"emb": [B,L,D1], "logits": [B,L,D2]}  if use_logits=True
        p1_mask: [B, L1]
        p2_mask: [B, L2]
        """
        # ---- Upgrade 1: optionally merge embeddings + logits per residue ----
        if self.use_logits:
            p1_feat = self._merge_emb_logits(p1)
            p2_feat = self._merge_emb_logits(p2)
        else:
            p1_feat = p1
            p2_feat = p2

        # decoder expects [B, L, H] already in hid_dim space; if not, you can
        # add a linear projection here. For now, we assume features are in decoder input space.
        # (If you need: add self.in_proj = nn.Linear(input_dim, hid_dim) in __init__.)

        # create "transformer style" masks if your Decoder uses them
        # Here we assume p*_mask is [B, L] with 1=valid, 0=pad, and your decoder
        # expects masks of shape [B, 1, 1, L] (like in your CPI code).
        def make_transformer_mask(mask):
            # [B, L] -> [B, 1, 1, L]
            return mask.unsqueeze(1).unsqueeze(2).to(self.device)

        src_mask_12 = make_transformer_mask(p1_mask)
        trg_mask_12 = make_transformer_mask(p2_mask)

        # ---- Forward direction: P1 -> P2 (old behavior) ----
        out12 = self.decoder(trg=p2_feat, src=p1_feat, trg_mask=trg_mask_12, src_mask=src_mask_12)
        # out12: [B, L2, H]
        g12 = self._pool(out12, p2_mask)  # [B, H]

        g21 = None
        out21 = None
        need_reverse = self.bidirectional or self.symmetric_scoring or self.use_bilinear_scoring

        if need_reverse:
            # ---- Reverse direction: P2 -> P1 (bi-directional cross-attention) ----
            src_mask_21 = make_transformer_mask(p2_mask)
            trg_mask_21 = make_transformer_mask(p1_mask)

            out21 = self.decoder(trg=p1_feat, src=p2_feat, trg_mask=trg_mask_21, src_mask=src_mask_21)
            # out21: [B, L1, H]
            g21 = self._pool(out21, p1_mask)  # [B, H]

        # ---- Upgrade 4: symmetry enforcement (order-invariant fusion) ----
        if self.bidirectional or self.symmetric_scoring:
            # we have both g12 and g21
            if g21 is None:
                # should not happen logically, but just in case
                g21 = g12

            # symmetric fusion: concat(sum, abs diff)
            g_sum = g12 + g21
            g_diff = torch.abs(g12 - g21)
            feat_vec = torch.cat([g_sum, g_diff], dim=-1)  # [B, 2H]
        else:
            # old behavior: just use forward direction pooled vector
            feat_vec = g12  # [B, H]

        # ---- Upgrade 3: optional bilinear scalar between g12 and g21 ----
        if self.use_bilinear_scoring:
            if g21 is None:
                g21 = g12
            # b = g12^T W g21   (scalar per sample)
            # equivalent to: (W g21) dot g12
            b_vec = self.bilinear(g21)          # [B, H]
            b = (b_vec * g12).sum(dim=-1, keepdim=True)  # [B, 1]
            feat_vec = torch.cat([feat_vec, b], dim=-1)  # [B, 2H+1] or [B,H+1]

        # final classifier
        logits = self.classifier(feat_vec)  # [B, 2]
        return logits

