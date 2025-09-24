# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Decoder for SeqTrack, modified from DETR transformer class.
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


# ------------------------------------------------------------------------
# Utility: simple MLP
# ------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------------
# Decoder Embeddings
# ------------------------------------------------------------------------
class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_embeds = self.word_embeddings(x)
        embeddings = self.LayerNorm(input_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings


# ------------------------------------------------------------------------
# SeqTrack Decoder with Step 0 (bbox_head + conf_head)
# ------------------------------------------------------------------------
class SeqTrackDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, bins=1000, num_frames=9):
        super().__init__()
        self.bins = bins
        self.num_frames = num_frames
        self.num_coordinates = 4  # [x, y, w, h]
        max_position_embeddings = (self.num_coordinates + 1) * num_frames

        self.embedding = DecoderEmbeddings(bins + 2, d_model,
                                           max_position_embeddings, dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.body = TransformerDecoder(decoder_layer, num_decoder_layers,
                                       decoder_norm,
                                       return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # -------------------------
        # Step 0a — Bbox head
        # projects hidden states -> vocab distribution (token logits)
        self.bbox_head = MLP(d_model, d_model, bins, 3)

        # Step 0b — Confidence head
        # projects last hidden -> scalar logit
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, seq):
        """
        Forward pass for training
        Args:
            src: encoder memory (N, B, C)
            pos_embed: positional encoding
            seq: input token sequence (B, T)
        Returns:
            token_logits: (B, T, V)
            conf_logits: (B, 1)
        """
        n, bs, c = src.shape
        tgt = self.embedding(seq).permute(1, 0, 2)

        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        memory = src
        tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        hs = self.body(tgt, memory, pos=pos_embed, query_pos=query_embed[:len(tgt)],
                       tgt_mask=tgt_mask, memory_mask=None)

        hs = hs.transpose(1, 2)  # (num_layers, B, T, D) → (B, T, D)

        # Step 1 — Box token predictions
        token_logits = self.bbox_head(hs)  # (B, T, V)

        # Step 2 — Confidence prediction from last hidden
        last_hidden = hs[:, -1, :]          # (B, D)
        conf_logits = self.conf_head(last_hidden)  # (B, 1)

        return token_logits, conf_logits

    def inference(self, src, pos_embed, seq, vocab_embed,
                  window, seq_format):
        """
        Inference loop: autoregressively predict 4 box tokens.
        Returns both tokens and confidence scores.
        """
        n, bs, c = src.shape
        memory = src
        confidence_list = []
        box_pos = [0, 1, 2, 3]  # positions for box tokens
        center_pos = [0, 1]
        if seq_format == 'whxy':
            center_pos = [2, 3]

        for i in range(self.num_coordinates):  # predict 4 tokens
            tgt = self.embedding(seq).permute(1, 0, 2)
            query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
            query_embed = query_embed.repeat(1, bs, 1)
            tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

            hs = self.body(tgt, memory, pos=pos_embed[:len(memory)],
                           query_pos=query_embed[:len(tgt)],
                           tgt_mask=tgt_mask, memory_mask=None)

            out = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])  # (B, V)
            out = out.softmax(-1)

            if i in box_pos:
                out = out[:, :self.bins]

            if ((i in center_pos) and (window is not None)):
                out = out * window

            confidence, token_generated = out.topk(dim=-1, k=1)
            seq = torch.cat([seq, token_generated], dim=-1)
            confidence_list.append(confidence)

        out_dict = {}
        out_dict['pred_boxes'] = seq[:, -self.num_coordinates:]
        out_dict['confidence'] = torch.cat(confidence_list, dim=-1)[:, :]

        return out_dict


# ------------------------------------------------------------------------
# Transformer Decoder modules
# ------------------------------------------------------------------------
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_decoder(cfg):
    return SeqTrackDecoder(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NHEADS,
        dim_feedforward=cfg.MODEL.DECODER.DIM_FEEDFORWARD,
        num_decoder_layers=cfg.MODEL.DECODER.DEC_LAYERS,
        normalize_before=cfg.MODEL.DECODER.PRE_NORM,
        return_intermediate_dec=False,
        bins=cfg.MODEL.BINS,
        num_frames=cfg.DATA.SEARCH.NUMBER
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
