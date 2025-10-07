from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch


class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, target_seqs = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs)

        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape  # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b, dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b, dim=0)

        # run encoder
        feature_xz = self.net(images_list=template_list + search_list, mode='encoder')

        bins = self.BINS
        start = bins + 1  # START token
        end = bins  # END token

        # box targets
        targets = data['search_anno'].permute(1, 0, 2).reshape(-1, data['search_anno'].shape[2])  # (n*b, 4)
        targets = box_xywh_to_xyxy(targets)
        targets = torch.clamp(targets, 0., 1.)  # keep in [0,1]

        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets * (bins - 1)).int()
        box = torch.clamp(box, 0, bins - 1)

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # input seq: [START, tokens...]
        input_start = torch.ones([batch, 1], device=box.device, dtype=box.dtype) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b, n, input_seqs.shape[-1]).flatten(1)

        # target seq: [tokens..., END]
        target_end = torch.ones([batch, 1], device=box.device, dtype=box.dtype) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1]).flatten().long()

        # run decoder
        token_logits, conf_logits = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        # ensure shapes are (B, T, V) and (B, 1)
        if token_logits.dim() == 4 and token_logits.size(0) == 1:
            token_logits = token_logits.squeeze(0)

        return (token_logits, conf_logits), target_seqs

    def compute_losses(self, outputs, targets_seq, return_status=True):
        # Unpack outputs
        token_logits, conf_logits = outputs  # (B, T, V), (B, 1)

        # Flatten token logits for CE loss
        B, T, V = token_logits.shape
        token_logits_flat = token_logits.reshape(B * T, V)

        # Cross-entropy loss on tokens
        ce_loss = self.objective['ce'](token_logits_flat, targets_seq)

        # --- Decode predicted boxes for IoU ---
        probs = token_logits.softmax(-1)
        probs = probs[:, :, :self.BINS]  # only coordinate bins
        _, extra_seq = probs.topk(dim=-1, k=1)  # greedy decode

        net_for_attrs = self.net.module if hasattr(self.net, 'module') else self.net
        tokens_per_seq = net_for_attrs.decoder.num_coordinates + 1  # [START, x, y, w, h, END]
        boxes_pred = extra_seq.squeeze(-1).reshape(-1, tokens_per_seq)[:, :-1]  # drop END
        boxes_target = targets_seq.reshape(-1, tokens_per_seq)[:, :-1]

        boxes_pred = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target = box_cxcywh_to_xyxy(boxes_target)

        ious = box_iou(boxes_pred, boxes_target)[0]  # (B, B)
        iou_per_sample = ious.diag()  # (B,)

        # --- Confidence loss ---
        if conf_logits is not None:
            tau = 0.5  # IoU threshold for supervision
            conf_target = (iou_per_sample >= tau).float().view(-1)  # (B,)
            conf_pred = conf_logits.view(-1)  # (B,)
            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(conf_pred, conf_target)
        else:
            conf_loss = 0.0

        # --- Weighted sum ---
        loss = (
                self.loss_weight['ce'] * ce_loss
                + self.loss_weight.get('conf', 0.0) * conf_loss
        )

        # --- Status dict for logging ---
        if return_status:
            status = {
                "Loss/total": loss.item(),
                "Loss/ce": ce_loss.item(),
                "Loss/conf": float(conf_loss) if not isinstance(conf_loss, float) else 0.0,
                "IoU": iou_per_sample.mean().item(),
                "Confidence": torch.sigmoid(conf_logits).mean().item() if conf_logits is not None else 0.0
            }
            return loss, status
        else:
            return loss

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)