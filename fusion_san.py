import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .attention import SanAttention, apply_attention, tile_2d_over_nd
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
import pdb
from .swin import SwinTransformerModel
from  .NAT import NATLayer

class SAN(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset,embedding_weights=None,rnn_bidirectional=True):
        super(SAN, self).__init__()
        embedding_requires_grad = not args.freeze_w2v
        question_features = 1024
        self.que2img = nn.Linear(question_features, question_features*2)
        #self.proj = nn.Linear(question_features*2, question_features)
        self.proj = nn.Conv2d(question_features*2, question_features, 1)
        self.act = nn.ReLU()
        #self.final = nn.Linear(2*14*28,question_features)
        #self.pad = nn.ZeroPad2d(padding=(0, 14 * 14 * 6 - question_features))
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
        vision_features = args.output_features
        glimpses = 2

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
        self.attention = SanAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.mlp = GroupMLP(
            in_features=glimpses * vision_features + question_features,   #(2*2048+1024)
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        # self.swin = SwinTransformerModel(
        #     input_image_channel=2,
        #     patch_size=4,
        #     model_dim_C=8,
        #     num_classes=4096,
        #     window_size=14,
        #     num_head=4,
        #     merge_size=2,
        # )
        self.nat = NATLayer(
            input_size=(14,14),
            dim=vision_features//2,
            num_heads=4,
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def fusion(self,ques,img):
        ques = self.que2img(ques)
        ques = ques.unsqueeze(-1)
        ques = ques.unsqueeze(-1)
        ques = ques.expand_as(img)
        fusion = torch.tanh(ques+img)
        fusion = self.proj(fusion)
        fusion = self.act(fusion)
        return self.drop(fusion)

    # def img_att(self, img, ent):
    #     bs, dim, H, W = img.shape
    #     img = img.reshape(bs, -1 ,dim)    #(128,196,1024)
    #     atten = F.softmax(torch.bmm(img, ent.unsqueeze(-1)), dim=-1)   #(128,196,1)
    #     rel = torch.bmm(atten.transpose(-1, -2), img)
    #     return rel.squeeze(1)

    def forward(self, v, b, q, q_len):
        # pdb.set_trace()
        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))
        # q = self.text(self.embedding(q), list(q_len.data))

        v = F.normalize(v, p=2, dim=1)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        # fusion = self.fusion(q,v)
        # a = self.nat(fusion)
        # combined = self.img_att(a, q)
        #v = apply_attention(v,a)

        combined = torch.cat([v, q], dim=1)
        embedding = self.mlp(combined)

        return embedding