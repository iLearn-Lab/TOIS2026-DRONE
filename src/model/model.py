import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.building_blocks import QueryGRUEncoder, VideoSelfAttentionEncoder, PositionwiseFeedForward, \
    QueryVideoCrossModalEncoder
from src.utils.utils import sliding_window

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

class Concat_SelfAttention(nn.Module):
    def __init__(self, dim, dropout):
        super(Concat_SelfAttention, self).__init__()
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.fc = PositionwiseFeedForward(dim=dim, d_ff=dim, dropout=dropout)

    def forward(self, x, mask):
        """
        Args:
            x: (B, L+N, dim)
            mask: (B, L+N)
        Returns:
            self_attn: (B, L+N, dim)
            self_attn_weights: (B, L+N, L+N)
        """
        temp = x
        self_attn, _ = self.mha(
            query=temp,
            key=temp,
            value=temp,
            key_padding_mask=(mask == 0.0)
        )
        self_attn = self_attn.masked_fill(mask.unsqueeze(2) == 0.0, value=0.0)

        return self_attn

class MoE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(MoE, self).__init__()
        self.w_1 = nn.Linear(in_dim, in_dim//2)
        self.w_2 = nn.Linear(in_dim//2, out_dim)
        self.layernorm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        """
        Args:
            x: (B, dim, L)
        Returns:
            (B, dim, 1)
        """
        inter = self.dropout_1(self.leakyrelu(self.w_1(self.layernorm(x))))
        output = self.dropout_2(self.leakyrelu(self.w_2(inter)))
        return output

class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self._read_model_config()
        self.nce_loss = nn.CrossEntropyLoss(reduction="none")

        # build network
        self.query_encoder = QueryGRUEncoder(
            in_dim=300,
            dim=self.dim // 2,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.fc_q = PositionwiseFeedForward(dim=self.dim, d_ff=2*self.dim, dropout=self.dropout)

        self.down_ffn = MoE(in_dim=self.video_feature_len, out_dim=1, dropout=self.dropout)

        self.video_encoder = VideoSelfAttentionEncoder(
            video_len=self.video_feature_len,
            in_dim=config[self.dataset_name]["feature_dim"],
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

        self.qv_encoder = QueryVideoCrossModalEncoder(
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

        self.concat_encoder = Concat_SelfAttention(
            dim=self.dim,
            dropout=self.dropout
        )

        self.change_word = nn.Linear(300, self.dim)
        self.change_video = nn.Linear(self.dim, self.dim)

        self.pool_word = WeightedPool(self.dim)

        # create optimizer, scheduler
        self._init_miscs()

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()

    def pooling(self, x, dim):
        return torch.max(x, dim=dim)[0]

    def max_pooling(self, x, mask, dim):
        return torch.max(x.masked_fill(mask == 0.0, -torch.inf), dim=dim)[0]

    def mean_pooling(self, x, mask, dim):
        return torch.sum(x * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-8)

    def network_forward(self, batch):
        """ The "Cross-modal Representation Module".

        Returns:
            sentence_feature: (B, dim)
            video_feature: (B, video_feature_len, dim)
            q2v_attn: (B, video_feature_len)

        """

        query_label = batch["query_label"]
        query_mask = batch["query_mask"]
        video = batch["video"]
        video_mask = batch["video_mask"]
        words_feature, temp_sentence, glove_emb = self.query_encoder(query_label, query_mask,
                                                                     word_vectors=batch["word_vectors"])

        words_feature = self.fc_q(words_feature)
        video_feature, video_embedding, position_embeddings = self.video_encoder(video, video_mask)

        glove_emb = self.change_word(glove_emb.detach())
        video_concat = self.change_video(video_embedding.detach())

        concat_feature = torch.concat((glove_emb, video_concat), dim=1)
        concat_mask = torch.concat((query_mask, video_mask), dim=1)

        new_concat_feture = self.concat_encoder(
            x=concat_feature,
            mask=concat_mask
        )

        new_concat_feture = new_concat_feture.masked_fill(concat_mask.unsqueeze(2) == 0.0, value=0.0)
        words_dim = glove_emb.shape[1]
        video_transpose = new_concat_feture[:, words_dim:, :].transpose(-1, -2)
        generate_glance = self.down_ffn(video_transpose).transpose(-1, -2).squeeze(1)

        words_feature = words_feature + generate_glance.unsqueeze(1).detach()

        words_feature, video_feature, q2v_attn = self.qv_encoder(
            query_feature=words_feature,
            query_mask=query_mask,
            video_feature=video_feature,
            video_mask=video_mask
        )

        query_mask = batch["query_mask"]
        sentence_feature = self.pooling(words_feature.masked_fill(query_mask.unsqueeze(2) == 0.0, -torch.inf), dim=1)

        return F.normalize(sentence_feature, dim=1), F.normalize(video_feature, dim=2), q2v_attn, generate_glance, words_feature

    def forward_eval(self, batch):
        """ The "Query Attention Guided Inference" module, use in evaluation.

        Returns:
            (B, topk, 2)
                start and end fractions
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights, _, _ = self.network_forward(batch)

        def generate_proposal(video_feature, video_mask, attn_weight):
            """ Use attn_weight to generate proposals.

            Returns:
                features: (num_proposals, dim)
                indices: (num_proposals, 2)
            """
            indices = []
            video_length = video_feature.shape[0]
            anchor_point = torch.argmax(attn_weight)
            for f in self.moment_length_factors:
                l = round(video_length * f)
                if l == 0:
                    continue
                for o in self.overlapping_factors:
                    l_overlap = round(l * o)
                    if l == l_overlap:
                        continue
                    l_rest = l - l_overlap
                    min_index = max(0, anchor_point - l)  # Ablation 3: no anchor point
                    max_index = min(video_length, anchor_point + l)  # Ablation 3: no anchor point
                    starts = range(min_index, anchor_point + 1, l_rest)  # Ablation 3: no anchor point
                    ends = range(min_index + l, max_index + 1, l_rest)  # Ablation 3: no anchor point
                    # starts = range(0, video_length, l_rest)  # Ablation 3: no anchor point
                    # ends = range(l, video_length + l, l_rest)  # Ablation 3: no anchor point
                    indices.append(torch.stack([torch.tensor([start, end]) for start, end in zip(starts, ends)], dim=0))
            indices = torch.cat(indices, dim=0)
            indices = torch.unique(indices, dim=0)  # remove duplicates
            features = torch.stack(
                [self.pooling_func(video_feature[s: e], video_mask[s: e], dim=0) for s, e in indices], dim=0
            )
            return features, indices

        B = video_feature.shape[0]
        video_mask = batch["video_mask"]
        video_lengths = torch.sum(video_mask, dim=1).to(torch.long)
        res = []
        for i in range(B):
            video_length = video_lengths[i].item()
            video = video_feature[i, :video_length]
            attn_weight = attn_weights[i, :video_length]
            features, indices = generate_proposal(video, video.new_ones(video.shape), attn_weight)
            scores = torch.mm(features, sentence_feature[i, :].unsqueeze(1)).squeeze(1)
            res.append(indices[torch.topk(scores, min(self.topk, indices.shape[0]), dim=0)[1].cpu()])
        res = torch.nn.utils.rnn.pad_sequence(res, batch_first=True).to(self.device)
        res = res / video_lengths.view(B, 1, 1)
        return res

    def forward_train_val(self, batch, epoch, total_epoch):
        """ The "Gaussian Alignment Module", use in training.

        Returns:
            loss: single item tensor
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights, generate_glance, _ = self.network_forward(
            batch)

        def get_gaussian_weight(video_mask, glance_frame):
            """ Get the Gaussian weight of full video feature.
            Args:
                video_mask: (B, L)
                glance_frame: (B)
            Returns:
                weight: (B, L)
            """
            B, L = video_mask.shape

            x = torch.linspace(-1, 1, steps=L, device=self.device).view(1, L).expand(B, L)
            lengths = torch.sum(video_mask, dim=1).to(torch.long)

            # normalize video lengths into range
            sig = lengths / L
            sig = sig.view(B, 1)
            sig *= self.sigma_factor

            # normalize glance frames into range
            u = ((glance_frame - 1) / (L - 1)) * 2 - 1
            u = u.view(B, 1)

            weight = torch.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            weight /= torch.max(weight, dim=1, keepdim=True)[0]  # normalize weight
            weight.masked_fill_(video_mask == 0.0, 0.0)
            return weight

        video_mask = batch["video_mask"]
        glance_frame = batch["glance_frame"]
        weight = get_gaussian_weight(video_mask, glance_frame)  # (B, L)

        select_frame = torch.gather(video_feature, 1, glance_frame.unsqueeze(1).unsqueeze(-1).expand(-1, -1, video_feature.shape[2]))

        gt_glance = select_frame.squeeze(1)

        def slice(video_feature, video_mask, weight, glance_frame, attn_weights):
            video_feature = video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, 0.0)
            B, L, D = video_feature.shape
            clips, clip_masks, clip_weights, clips_in_moment, clips_frame = [], [], [], [], []
            for clip_frame in self.clip_frames:
                temp, temp_attn, slice_mask, idx = sliding_window(video_feature, attn_weights, video_mask.unsqueeze(2), clip_frame, self.stride, dim=1)
                temp_frame = torch.stack([self.mean_pooling(x, mask, dim=1) for x,mask in zip(temp,slice_mask)], dim=1) # (B, N, dim)
                temp = torch.stack([self.max_pooling(x, mask, dim=1) for x,mask in zip(temp,slice_mask)], dim=1)  # (B, N, dim)
                temp_mask = video_mask[:, idx[:, 0]]  # (B, N)
                temp.masked_fill_(temp_mask.unsqueeze(2) == 0.0, 0.0)
                temp_weight = weight[:, idx[:, 0]] * weight[:, idx[:, 1] - 1]
                temp_start_idx = idx[:,0].unsqueeze(0).expand_as(temp_mask)
                temp_end_idx = idx[:,1].unsqueeze(0).expand_as(temp_mask)
                temp_glance_frame = glance_frame.unsqueeze(1).expand_as(temp_mask)
                temp_in_moment = torch.logical_and(temp_start_idx <= temp_glance_frame, temp_glance_frame <= temp_end_idx).long() # (B, N)
                clips.append(temp)
                clip_masks.append(temp_mask)
                clip_weights.append(temp_weight)
                clips_in_moment.append(temp_in_moment)
                clips_frame.append(temp_frame)
            clips = torch.cat(clips, dim=1)
            clip_masks = torch.cat(clip_masks, dim=1)
            clip_weights = torch.cat(clip_weights, dim=1)
            clips_in_moment = torch.cat(clips_in_moment, dim=1)
            clips_frame = torch.cat(clips_frame, dim=1)
            return clips, clip_masks, clip_weights, clips_in_moment, clips_frame, temp_start_idx, temp_end_idx

        clips, clip_masks, clip_weights, clips_in_moment, clips_frame, temp_start_idx, temp_end_idx = slice(
            video_feature, video_mask, weight, glance_frame, attn_weights
        )

        # loss
        frame_score = torch.bmm(clips_frame, sentence_feature.unsqueeze(1).transpose(1,2)).squeeze()  # (B, N)
        frame_score = frame_score / self.temp
        frame_loss = -1.0 * clip_weights * clips_in_moment * F.log_softmax(frame_score, dim=1)

        epoch_ratio = epoch / total_epoch

        training_progress = 0.1 + 0.9 * epoch_ratio ** 2

        attn_loss = F.kl_div(F.log_softmax(attn_weights, dim=1), F.log_softmax(weight, dim=1), reduction="none", log_target=True)
        attn_loss.masked_fill_(video_mask == 0.0, 0.0)

        B, N = temp_start_idx.shape
        L = attn_loss.shape[1]
        frame_indices = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)

        clip_mask = (frame_indices.unsqueeze(1) >= temp_start_idx.unsqueeze(-1)) & \
                    (frame_indices.unsqueeze(1) < temp_end_idx.unsqueeze(-1))


        course_attn_loss = (attn_loss.unsqueeze(1) * clip_mask).sum(dim=-1)
        clip_loss_count = clip_mask.sum(dim=-1).clamp(min=1)
        course_attn_loss = course_attn_loss / clip_loss_count

        attn_loss = torch.sum(attn_loss) / torch.sum(video_mask) * 100000

        gen_loss = F.mse_loss(generate_glance, gt_glance.detach()) * 10 * self.alpha

        course_gen_loss = F.mse_loss(generate_glance, gt_glance.detach(), reduction="none") * 10
        course_gen_loss = course_gen_loss.sum(dim=1, keepdim=True)

        sentence_video_context = torch.cat([sentence_feature.unsqueeze(1), video_feature], dim=1).detach()
        affine_1 = F.cosine_similarity(generate_glance.unsqueeze(1),
                                       sentence_video_context,
                                       dim=-1)
        affine_2 = F.cosine_similarity(gt_glance.unsqueeze(1).detach(),
                                       sentence_video_context,
                                       dim=-1)
        affine_loss = F.kl_div(F.log_softmax(affine_1, dim=1), F.log_softmax(affine_2, dim=1), reduction="none",
                               log_target=True)
        video_mask_temp = torch.cat([torch.ones((B, 1), device=self.device), video_mask], dim=1)
        affine_loss.masked_fill_(video_mask_temp == 0.0, 0.0)

        course_affine_loss = affine_loss[:, 1:]

        course_affine_loss = (course_affine_loss.unsqueeze(1) * clip_mask).sum(dim=-1)
        course_affine_loss = course_affine_loss / clip_loss_count


        affine_loss = torch.sum(affine_loss) / torch.sum(video_mask + 1) * 100 * self.beta


        combined_loss = (course_affine_loss + course_attn_loss) * course_gen_loss
        combined_loss = combined_loss.detach()
        binary_mask = torch.where(combined_loss == 0.0, combined_loss.new_full((), -9999.0), combined_loss.new_full((), 1.0))
        binary_mask_2 = torch.where(combined_loss == 0.0, combined_loss.new_zeros(()), combined_loss.new_full((), 1.0))

        row_min = torch.min(combined_loss, dim=1, keepdim=True)
        combined_loss = 1 + combined_loss - row_min.values
        combined_loss = combined_loss * binary_mask

        inverse_loss = 1.0 / (combined_loss + 1e-6)
        combined_weight = torch.softmax(inverse_loss, dim=1)
        combined_weight = combined_weight * binary_mask_2


        frame_loss = frame_loss * combined_weight.detach()

        frame_loss = torch.sum(frame_loss, dim=(0, 1)) / torch.sum(clips_in_moment, dim=(0, 1)) * 100 * self.gamma


        loss = attn_loss + gen_loss + training_progress*(frame_loss) + affine_loss
        return loss


    def _read_model_config(self):
        self.dataset_name = self.config["dataset_name"]

        # task independent config
        self.dim = self.config["model"]["dim"]
        self.dropout = self.config["model"]["dropout"]
        self.n_layers = self.config["model"]["n_layers"]
        self.temp = self.config["model"]["temp"]
        self.topk = self.config["model"]["topk"]

        # task dependent config
        self.video_feature_len = self.config[self.dataset_name]["video_feature_len"]
        self.clip_frames = self.config[self.dataset_name]["clip_frames"]
        self.stride = self.config[self.dataset_name]["stride"]
        self.sigma_factor = self.config[self.dataset_name]["sigma_factor"]
        self.moment_length_factors = self.config[self.dataset_name]["moment_length_factors"]
        self.overlapping_factors = self.config[self.dataset_name]["overlapping_factors"]

        self.pooling_func = getattr(self, self.config[self.dataset_name]["pooling_func"])

        self.alpha = self.config['alpha']
        self.beta  = self.config['beta']
        self.gamma = self.config['gamma']

    def _init_miscs(self):
        """
        Key attributes created here:
            - self.optimizer
            - self.scheduler
        """
        lr = self.config["train"]["init_lr"]
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=3
        )

    def _prepare_batch(self, batch):
        keys = ["query_label", "query_mask", "video", "video_mask",
                "start_frac", "end_frac", "start_frame", "end_frame",
                "glance_frac", "glance_frame", "word_vectors"]
        for k in keys:
            batch[k] = batch[k].to(self.device)
        return batch

    def optimizer_step(self, loss):
        """ Update the network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config["train"]["clip_norm"])
        self.optimizer.step()

    def scheduler_step(self, valid_loss):
        """
        Args:
            valid_loss: loss on valid set; tensor
        """
        self.scheduler.step(valid_loss)

    def load_checkpoint(self, exp_folder_path, suffix):
        self.load_state_dict(torch.load(os.path.join(exp_folder_path, "model_{}.pt".format(suffix))))
        # self.optimizer.load_state_dict(torch.load(os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix))))
        # self.scheduler.load_state_dict(torch.load(os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix))))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        torch.save(self.state_dict(), os.path.join(exp_folder_path, "model_{}.pt".format(suffix)))
        # torch.save(self.optimizer.state_dict(), os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix)))
        # torch.save(self.scheduler.state_dict(), os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix)))
        print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()



