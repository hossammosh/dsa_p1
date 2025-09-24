"""
SeqTrack Model
"""
import torch
import math
from torch import nn

from lib.models.seqtrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table


class SEQTRACK(nn.Module):
    """ This is the base class for SeqTrack """
    def __init__(self, encoder, decoder, hidden_dim,
                 bins=1000, feature_type='x', num_frames=1, num_template=1):
        """
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim

        # bottleneck to align encoder channels with decoder hidden_dim
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim)

        self.decoder = decoder  # contains bbox_head + conf_head now

        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # position embedding for the decoder
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, hidden_dim, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, images_list=None, xz=None, seq=None, mode="encoder"):
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz, seq)
        else:
            raise ValueError

    def forward_encoder(self, images_list):
        return self.encoder(images_list)

    def forward_decoder(self, xz, sequence):
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # choose features for decoder
        if self.feature_type == 'x':
            dec_mem = xz_mem[:, 0:self.num_patch_x * self.num_frames]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        # align dims
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1, 0, 2)  # (N, B, D)

        token_logits, conf_logits = self.decoder(
            dec_mem, self.pos_embed.permute(1, 0, 2).expand(-1, B, -1), sequence
        )
        return token_logits, conf_logits

    def inference_decoder(self, xz, sequence, window=None, seq_format='xywh'):
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        if self.feature_type == 'x':
            dec_mem = xz_mem[:, 0:self.num_patch_x]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1, 0, 2)

        out = self.decoder.inference(
            dec_mem,
            self.pos_embed.permute(1, 0, 2).expand(-1, B, -1),
            sequence,
            window=window,
            seq_format=seq_format
        )
        return out


def build_seqtrack(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    model = SEQTRACK(
        encoder,
        decoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins=cfg.MODEL.BINS,
        feature_type=cfg.MODEL.FEATURE_TYPE,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER
    )
    return model
