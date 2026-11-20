"""Sign-level masked-diffusion encoder for cuneiform (Stoicheia architecture, cuneiform planes).

Input planes per sign position (summed embeddings):
  sign     OSL sign id, or MASK / BLANK / PAD / <unk-sign> / <x> / <gap>
  read     reading id or <none> (no reading given) / <mask>
  join     relation to the next sign: w - . : + '' line, or UNKNOWN
  det      none / pre / post determinative, or UNKNOWN
  lang     text language (per position, constant within a text), or UNKNOWN
Heads: sign, read, join, det.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import Block, RMSNorm, RoPE, build_attn_mask

from .consts import BLANK, GAP, MASK, N_DET, N_JOIN, PAD, READ_MASK, READ_NONE, UNK_SIGN, X_SIGN  # noqa: F401


@dataclass
class CunConfig:
    n_sign: int
    n_read: int
    n_lang: int
    d_model: int = 768
    n_heads: int = 12
    depth: int = 12
    window: int = 128     # local attention window in signs; every 4th block global
    qk_norm: bool = True
    dropout: float = 0.0  # residual and embedding dropout
    status_plane: bool = False   # R2: input plane clean / damaged / restored / unknown
    elastic_flag: bool = False   # O1: input flag on the slots of a gap of unknown width
    width_head: bool = False     # O1: number of real signs of such a gap, predicted at its first slot


class CunEncoder(nn.Module):
    def __init__(self, cfg: CunConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.e_sign = nn.Embedding(cfg.n_sign, d)
        self.e_read = nn.Embedding(cfg.n_read, d)
        self.e_join = nn.Embedding(N_JOIN + 1, d)          # + unknown
        self.e_det = nn.Embedding(N_DET + 1, d)            # + unknown
        self.e_lang = nn.Embedding(cfg.n_lang + 1, d)      # + unknown
        if cfg.status_plane:
            self.e_status = nn.Embedding(4, d)
        if cfg.elastic_flag:
            self.e_slot = nn.Embedding(2, d)
        if cfg.width_head:
            self.h_width = nn.Linear(d, 32, bias=False)
        rope = RoPE(d // cfg.n_heads)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, rope, window=0 if i % 4 == 3 else cfg.window,
                                          qk_norm=cfg.qk_norm, dropout=cfg.dropout) for i in range(cfg.depth))
        self.drop = nn.Dropout(cfg.dropout)
        self.norm = RMSNorm(d)
        self.h_sign = nn.Linear(d, cfg.n_sign, bias=False)
        self.h_read = nn.Linear(d, cfg.n_read, bias=False)
        self.h_join = nn.Linear(d, N_JOIN, bias=False)
        self.h_det = nn.Linear(d, N_DET, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, b, heads=("sign", "read", "join", "det")):
        x = (self.e_sign(b["sign"]) + self.e_read(b["read"]) + self.e_join(b["join"]) + self.e_det(b["det"])
             + self.e_lang(b["lang"]))
        if self.cfg.status_plane:
            x = x + self.e_status(b["status"])
        if self.cfg.elastic_flag:
            x = x + self.e_slot(b["slot"])
        x = self.drop(x)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        seg = b["seg"]
        loc = build_attn_mask(seg, self.cfg.window, x.device, x.dtype)
        glo = build_attn_mask(seg, 0, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, pos, glo if blk.window == 0 else loc)
        x = self.norm(x)
        out = {"hidden": x}
        for h in heads:
            out[h] = getattr(self, "h_" + h)(x)
        if self.cfg.width_head:
            out["width"] = self.h_width(x)
        return out


def num_params(m):
    return sum(p.numel() for p in m.parameters())
