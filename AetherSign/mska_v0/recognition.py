from __future__ import annotations

import math
import warnings
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn

from .tokenizer import GlossTokenizerS2G
from .visual_head import VisualHead

try:
    import tensorflow as tf
except ImportError:
    tf = None


def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    if tf is None:
        if beam_size > 1:
            warnings.warn(
                "TensorFlow is not installed; falling back to greedy CTC decoding instead of beam search.",
                RuntimeWarning,
                stacklevel=2,
            )
        blank_index = tf_gloss_logits.shape[-1] - 1
        greedy_tokens = np.argmax(tf_gloss_logits, axis=-1)
        decoded_sequences = []
        for batch_index in range(greedy_tokens.shape[1]):
            valid_len = int(input_lengths[batch_index].item())
            collapsed = []
            previous = None
            for token in greedy_tokens[:valid_len, batch_index]:
                token = int(token)
                if token == previous:
                    continue
                previous = token
                if token != blank_index:
                    collapsed.append(token + 1)
            decoded_sequences.append(collapsed)
        return decoded_sequences

    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits,
        sequence_length=input_lengths.detach().cpu().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    temp_sequences = [[] for _ in range(input_lengths.shape[0])]
    for value_index, dense_index in enumerate(ctc_decode.indices):
        temp_sequences[dense_index[0]].append(ctc_decode.values[value_index].numpy() + 1)
    decoded_sequences = []
    for seq in temp_sequences:
        decoded_sequences.append([token[0] for token in groupby(seq)])
    return decoded_sequences


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super().__init__()
        positions = []
        for _ in range(time_len):
            for joint_id in range(joint_num):
                positions.append(joint_id)
        position = torch.from_numpy(np.array(positions)).unsqueeze(1).float()
        pe = torch.zeros(time_len * joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :, : x.size(2)]


class STAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        inter_channels,
        num_subset=2,
        num_node=27,
        num_frame=400,
        kernel_size=1,
        stride=1,
        t_kernel=3,
        glo_reg_s=True,
        att_s=True,
        glo_reg_t=False,
        att_t=False,
        use_temporal_att=False,
        use_spatial_att=True,
        attentiondrop=0.0,
        use_pes=True,
        use_pet=False,
    ):
        super().__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet
        self.use_spatial_att = use_spatial_att

        pad = int((kernel_size - 1) / 2)
        if use_spatial_att:
            self.register_buffer("atts", torch.zeros((1, num_subset, num_node, num_node)))
            self.pes = SpatialPositionalEncoding(in_channels, num_node, num_frame)
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(
                    torch.ones(1, num_subset, num_node, num_node) / num_node,
                    requires_grad=True,
                )
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        batch_size, _, time_steps, _ = x.size()
        if self.use_spatial_att:
            attention = self.atts
            y = self.pes(x) if self.use_pes else x
            if self.att_s:
                q, k = torch.chunk(
                    self.in_nets(y).view(batch_size, 2 * self.num_subset, self.inter_channels, time_steps, -1),
                    2,
                    dim=1,
                )
                attention = attention + self.tan(
                    torch.einsum("nsctu,nsctv->nsuv", [q, k]) / (self.inter_channels * time_steps)
                ) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(batch_size, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum("nctu,nsuv->nsctv", [x, attention]).contiguous().view(
                batch_size,
                self.num_subset * self.in_channels,
                time_steps,
                -1,
            )
            y = self.out_nets(y)
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)
        return z


class DSTA(nn.Module):
    def __init__(
        self,
        num_frame=400,
        num_subset=6,
        dropout=0.1,
        cfg=None,
        num_channel=3,
        glo_reg_s=True,
        att_s=True,
        glo_reg_t=False,
        att_t=False,
        use_temporal_att=False,
        use_spatial_att=True,
        attentiondrop=0.1,
        use_pet=False,
        use_pes=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_frame = num_frame
        config = cfg["net"]
        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        shared_param = {
            "num_subset": num_subset,
            "glo_reg_s": glo_reg_s,
            "att_s": att_s,
            "glo_reg_t": glo_reg_t,
            "att_t": att_t,
            "use_spatial_att": use_spatial_att,
            "use_temporal_att": use_temporal_att,
            "use_pet": use_pet,
            "use_pes": use_pes,
            "attentiondrop": attentiondrop,
        }

        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        self.face_graph_layers = self._build_stream_layers(config, len(cfg["face"]), self.num_frame, shared_param)
        self.left_graph_layers = self._build_stream_layers(config, len(cfg["left"]), self.num_frame, shared_param)
        self.right_graph_layers = self._build_stream_layers(config, len(cfg["right"]), self.num_frame, shared_param)
        self.body_graph_layers = self._build_stream_layers(config, len(cfg["body"]), self.num_frame, shared_param)
        self.drop_out = nn.Dropout(dropout)

    def _build_stream_layers(self, net_cfg, num_nodes, num_frame, shared_param):
        layers = nn.ModuleList()
        current_frame = num_frame
        for in_channels, out_channels, inter_channels, t_kernel, stride in net_cfg:
            layers.append(
                STAttentionBlock(
                    in_channels,
                    out_channels,
                    inter_channels,
                    stride=stride,
                    num_node=num_nodes,
                    t_kernel=t_kernel,
                    num_frame=current_frame,
                    **shared_param,
                )
            )
            current_frame = int(current_frame / stride + 0.5)
        return layers

    def forward(self, src_input):
        device = next(self.parameters()).device
        x = src_input["keypoint"].to(device)
        batch_size, channels, time_steps, num_nodes = x.shape
        x = x.view(batch_size, channels, time_steps, num_nodes)

        left = self.left_input_map(x[:, :, :, self.cfg["left"]])
        right = self.right_input_map(x[:, :, :, self.cfg["right"]])
        face = self.face_input_map(x[:, :, :, self.cfg["face"]])
        body = self.body_input_map(x[:, :, :, self.cfg["body"]])

        for layer in self.face_graph_layers:
            face = layer(face)
        for layer in self.left_graph_layers:
            left = layer(left)
        for layer in self.right_graph_layers:
            right = layer(right)
        for layer in self.body_graph_layers:
            body = layer(body)

        left = left.permute(0, 2, 1, 3).contiguous().mean(3)
        right = right.permute(0, 2, 1, 3).contiguous().mean(3)
        face = face.permute(0, 2, 1, 3).contiguous().mean(3)
        body = body.permute(0, 2, 1, 3).contiguous().mean(3)

        fuse_output = torch.cat([left, face, right, body], dim=-1)
        left_output = torch.cat([left, face], dim=-1)
        right_output = torch.cat([right, face], dim=-1)
        return fuse_output, left_output, right_output, body


class Recognition(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["input_type"] != "keypoint":
            raise ValueError("Only keypoint input is supported in the self-contained v0 package")

        self.gloss_tokenizer = GlossTokenizerS2G(cfg["GlossTokenizer"])
        self.visual_backbone_keypoint = DSTA(cfg=cfg["DSTA-Net"], num_channel=3)
        self.fuse_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg["fuse_visual_head"])
        self.body_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg["body_visual_head"])
        self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg["left_visual_head"])
        self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg["right_visual_head"])
        self.recognition_loss_func = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")

    def compute_recognition_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.recognition_loss_func(
            log_probs=gloss_probabilities_log.permute(1, 0, 2),
            targets=gloss_labels,
            input_lengths=input_lengths,
            target_lengths=gloss_lengths,
        )
        return loss / gloss_probabilities_log.shape[0]

    def decode(self, gloss_logits, beam_size, input_lengths):
        logits = gloss_logits.permute(1, 0, 2).detach().cpu().numpy()
        tf_gloss_logits = np.concatenate((logits[:, :, 1:], logits[:, :, 0, None]), axis=-1)
        return ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size,
        )

    def forward(self, src_input):
        device = next(self.parameters()).device
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(src_input)
        mask = src_input["mask"].to(device)
        input_lengths = src_input["new_src_lengths"].to(device)

        body_head = self.body_visual_head(x=body, mask=mask, valid_len_in=input_lengths)
        fuse_head = self.fuse_visual_head(x=fuse, mask=mask, valid_len_in=input_lengths)
        left_head = self.left_visual_head(x=left_output, mask=mask, valid_len_in=input_lengths)
        right_head = self.right_visual_head(x=right_output, mask=mask, valid_len_in=input_lengths)

        ensemble_probabilities = (
            left_head["gloss_probabilities"]
            + right_head["gloss_probabilities"]
            + body_head["gloss_probabilities"]
            + fuse_head["gloss_probabilities"]
        ) / 4.0

        outputs = {
            "ensemble_last_gloss_logits": ensemble_probabilities.clamp_min(1e-8).log(),
            "ensemble_last_gloss_probabilities": ensemble_probabilities,
            "ensemble_last_gloss_probabilities_log": ensemble_probabilities.clamp_min(1e-8).log(),
            "fuse": fuse,
            "fuse_gloss_logits": fuse_head["gloss_logits"],
            "fuse_gloss_probabilities_log": fuse_head["gloss_probabilities_log"],
            "body_gloss_logits": body_head["gloss_logits"],
            "body_gloss_probabilities_log": body_head["gloss_probabilities_log"],
            "left_gloss_logits": left_head["gloss_logits"],
            "left_gloss_probabilities_log": left_head["gloss_probabilities_log"],
            "right_gloss_logits": right_head["gloss_logits"],
            "right_gloss_probabilities_log": right_head["gloss_probabilities_log"],
            "gloss_feature": fuse_head[self.cfg.get("gloss_feature_ensemble", "gloss_feature")],
            "input_lengths": input_lengths,
        }

        gloss_labels = src_input["gloss_input"]["gloss_labels"].to(device)
        gloss_lengths = src_input["gloss_input"]["gls_lengths"].to(device)
        for head_name in ["left", "right", "fuse", "body"]:
            outputs[f"recognition_loss_{head_name}"] = self.compute_recognition_loss(
                gloss_labels=gloss_labels,
                gloss_lengths=gloss_lengths,
                gloss_probabilities_log=outputs[f"{head_name}_gloss_probabilities_log"],
                input_lengths=input_lengths,
            )

        outputs["recognition_loss"] = (
            outputs["recognition_loss_left"]
            + outputs["recognition_loss_right"]
            + outputs["recognition_loss_fuse"]
            + outputs["recognition_loss_body"]
        )

        if self.cfg.get("cross_distillation", False):
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            teacher_prob = outputs["ensemble_last_gloss_probabilities"].detach()
            for student in ["left", "right", "fuse", "body"]:
                student_log_prob = outputs[f"{student}_gloss_probabilities_log"]
                outputs[f"{student}_distill_loss"] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs["recognition_loss"] += outputs[f"{student}_distill_loss"]

        return outputs
