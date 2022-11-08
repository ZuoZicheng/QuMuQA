import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer, cosine_sim
from torch.autograd import Variable
from .attention import SanAttention, apply_attention, tile_2d_over_nd
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
import math
import numpy as np
import pdb
from collections import Counter
from .swin import SwinTransformerModel
from  .NAT import NATLayer

class SAN_TAIL(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset,embedding_weights=None,rnn_bidirectional=True):
        super(SAN_TAIL, self).__init__()
        embedding_requires_grad = not args.freeze_w2v
        question_features = 1024
        self.que2img = nn.Linear(question_features, question_features*2)
        self.proj = nn.Conv2d(question_features*2, question_features, 1)
        self.act = nn.ReLU()
        #self.final = nn.Linear(2*14*28,question_features)
        #self.pad = nn.ZeroPad2d(padding=(0, 14 * 14 * 6 - question_features))
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
        vision_features = args.output_features
        glimpses = 2
        self.args = args

        # vocab_size = embedding_weights.size(0)
        # vector_dim = embedding_weights.size(1)
        # self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        # self.embedding.weight.data = embedding_weights
        # self.embedding.weight.requires_grad = embedding_requires_grad
        self.w_emb = WordEmbedding(embedding_weights.size(0), 300, .0)
        if args.freeze_w2v:
            self.w_emb.init_embedding(embedding_weights)
            freeze_layer(self.w_emb)

        self.drop = nn.Dropout(0.5)
        self.text = Seq2SeqRNN(
            input_features=embedding_weights.size(1),
            rnn_features=int(rnn_features),
            rnn_type='LSTM',
            rnn_bidirectional=rnn_bidirectional,
        )
        self.gru = nn.GRU(
            input_size=embedding_weights.size(1),
            hidden_size=int(rnn_features),
            num_layers=3,
            dropout=0.5,
            bidirectional=True
        )
        self.attention = SanAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.mlp = GroupMLP(
            in_features=glimpses * vision_features + question_features,   #(2*2048+1024)
            #in_features=vision_features,  # (2*2048+1024)
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        self.mlp_tail = GroupMLP(
            # in_features=glimpses * vision_features + question_features,   #(2*2048+1024)
            in_features=300,  # (2*2048+1024)
            mid_features=4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        # self.cand_ans = GroupMLP(
        #     # in_features=glimpses * vision_features + question_features,   #(2*2048+1024)
        #     in_features=300,  # (2*2048+1024)
        #     mid_features=4 * args.hidden_size,
        #     out_features=4 * args.embedding_size,
        #     drop=0.5,
        #     groups=64,
        # )
        # self.swin = SwinTransformerModel(
        #     input_image_channel=2,
        #     patch_size=4,
        #     model_dim_C=8,
        #     num_classes=4096,
        #     window_size=14,
        #     num_head=4,
        #     merge_size=2,
        # )

        # self.nat = NATLayer(
        #     input_size=(14,14),
        #     dim=vision_features//2,
        #     num_heads=4,
        # )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    # def fusion(self,ques,img):
    #     ques = self.que2img(ques)
    #     ques = ques.unsqueeze(-1)
    #     ques = ques.unsqueeze(-1)
    #     ques = ques.expand_as(img)
    #     fusion = torch.tanh(ques+img)
    #     fusion = self.proj(fusion)
    #     fusion = self.act(fusion)
    #     return self.drop(fusion)

    # def w_rel(self, ques, ent):
    #     outputs, ht = self.gru(self.w_emb(ques))  # (128,25)->(128,25,300)->(128,25,1024)
    #     atten = F.softmax(torch.bmm(ent.unsqueeze(1),outputs.transpose(-2, -1)), dim=-1)
    #     rel = torch.bmm(atten, outputs).squeeze(1)
    #     return rel
    #
    # def img_att(self, img, ent):
    #     bs, dim, H, W = img.shape
    #     img = img.view(bs, -1 ,dim)    #(128,196,1024)
    #     atten = F.softmax(torch.bmm(img, ent.unsqueeze(-1)), dim=-1)   #(128,196,1)
    #     rel = torch.bmm(atten.transpose(-1, -2), img)
    #     return rel.squeeze(1)

    def reason(self, p_head, p_rel, kblist, g_head, g_rel, g_tail):
        p_ans = []
        id_head = 0
        sim_head = (cosine_sim(p_head, g_head) / self.args.loss_temperature).to(torch.float64)
        sim_rel = (cosine_sim(p_rel, g_rel) / self.args.loss_temperature).to(torch.float64)
        _, head_id = sim_head.topk(3,dim=1)
        head_id = head_id.cpu().numpy()
        _, rel_id = sim_rel.topk(3,dim=1)
        rel_id = rel_id.cpu().numpy()
        for step in range(0,head_id.shape[0]):
            p_key = [[],[]]
            flag = True
            while id_head < head_id.shape[1] and flag == True:
                p_key[0] = head_id[step].astype(str)[id_head]
                id_rel = 0
                while id_rel < rel_id.shape[1] and flag == True:
                    p_key[1] = rel_id[step].astype(str)[id_rel]
                    p_kb_key = '-'.join(p_key)
                    id_rel += 1
                    if p_kb_key in kblist:
                        p_ans_id = kblist[p_kb_key]
                        if len(g_tail[p_ans_id]) == 1:
                            p_ans_value = g_tail[p_ans_id[0]]
                        else:
                            p_ans_value = Counter(g_tail[p_ans_id]).most_common()[0][0]
                        p_ans.append(p_ans_value)
                        flag = False
                        #match_ans += 1
                id_head += 1
            id_head = 0
            if flag == True:
                p_ans_value = (torch.zeros((300))+0.0001).cuda()
                p_ans.append(p_ans_value)
        return torch.stack(p_ans)

    # def kbcheck(self, kblist):
    #     value = []
    #     for val_list in kblist.values():
    #         for val in val_list:
    #             value.append(val)
    #     value = list(set(value))
    #     value.sort()
    #     return value

    # def reason_rank(self, p_head, p_rel, kblist, g_head, g_rel, g_tail, q, v):
    #     # img = F.normalize(v, p=2, dim=1)  # (128,2048,14,14)
    #     # img_att = self.attention(img, q)  # (128,2,14,14)
    #     # fusion = apply_attention(img, img_att)  # (128,4096)
    #     p_ans = []
    #     id_head = 0
    #     sim_head = (cosine_sim(p_head, g_head) / self.args.loss_temperature).to(torch.float64)
    #     sim_rel = (cosine_sim(p_rel, g_rel) / self.args.loss_temperature).to(torch.float64)
    #     _, head_id = sim_head.topk(1,dim=1)
    #     head_id = head_id.cpu().numpy()
    #     _, rel_id = sim_rel.topk(3,dim=1)
    #     rel_id = rel_id.cpu().numpy()
    #     for step in range(0,head_id.shape[0]):
    #         p_key = [[],[]]
    #         flag = True
    #         while id_head < head_id.shape[1] and flag == True:
    #             p_key[0] = head_id[step].astype(str)[id_head]
    #             id_rel = 0
    #             while id_rel < rel_id.shape[1] and flag == True:
    #                 p_key[1] = rel_id[step].astype(str)[id_rel]
    #                 p_kb_key = '-'.join(p_key)
    #                 id_rel += 1
    #                 if p_kb_key in kblist:
    #                     p_ans_id = kblist[p_kb_key]
    #                     p_ans_id = list(set(p_ans_id))
    #                     if len(g_tail[p_ans_id]) == 1:
    #                         p_ans_value = g_tail[p_ans_id[0]]
    #                     else:
    #                         #cur_fusion = fusion[step].unsqueeze(0)
    #                         ans_cand = []
    #                         for ans_key in p_ans_id:
    #                             ans_cand.append(g_tail[ans_key].cpu().numpy())
    #                         ans_cand = torch.tensor(ans_cand)
    #                         conv = nn.Conv1d(ans_cand.size(0), 1, 1)
    #                         p_ans_value = conv(ans_cand).squeeze(0).cuda()
    #                     p_ans.append(p_ans_value)
    #                     flag = False
    #             id_head += 1
    #         id_head = 0
    #         if flag == True:
    #             p_ans_value = (torch.zeros((300))+0.0001).cuda()
    #             p_ans.append(p_ans_value)
    #     return torch.stack(p_ans)

    # def test_ans(self, g_ans):
    #     ans_list = []
    #     g_tail_list = torch.from_numpy(np.loadtxt('anslist.txt', delimiter=',').reshape((500, 1024))).float().cuda()
    #     sim_ans = cosine_sim(g_ans, g_tail_list).to(torch.float64)
    #     _, ans_id = sim_ans.topk(1, dim=1)
    #     ans_id = ans_id.cpu().numpy()
    #     for idx in ans_id:
    #         ans_list.append(g_tail_list[idx[0]])
    #     return ans_id

    def forward(self, v, b, q, q_len, head, rel, fact, g_head, g_rel, g_tail):
        # pdb.set_trace()
        #kb = self.kbcheck(fact)
        tail = self.reason(head, rel, fact, g_head, g_rel, g_tail)
        emb_tail = self.mlp_tail(tail)
        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))

        # if zsl use reason_rank
        # tail = self.reason_rank(head, rel, fact, g_head, g_rel, g_tail, q, v)
        # emb_tail = self.mlp_tail(tail)

        v = F.normalize(v, p=2, dim=1)  #(128,2048,14,14)
        a = self.attention(v, emb_tail)    #(128,2,14,14)
        v = apply_attention(v, a)     #(128,4096)

        #v = F.normalize(v, p=2, dim=1)  #(128,2048,14,14)
        #fusion = self.fusion(q,v)  #(128,1024,14,14)
        #fusion = self.fusion(emb_tail, fusion)
        #fusion_tail = self.img_att(fusion, emb_tail)
        #a = self.nat(fusion)
        #combined = apply_attention(v,a)

        combined = torch.cat([v, q], dim=1)
        embedding = self.mlp(combined)

        return embedding