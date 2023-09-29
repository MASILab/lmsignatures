import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import HEAD

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TEM(nn.Module):
    """
    Models time distance as a flipped sigmoid function
    Learnable parameters for each attention head:
        self.a describes slope of decay
        self.c describes position of decay
    """

    def __init__(self, heads=8):
        super().__init__()
        self.heads = heads
        # initialize from [0,1], which fits decay to scale of fractional months
        self.a = nn.Parameter(torch.rand(heads), requires_grad=True)
        # initialize from [0,12], which fits position to the scale of fractional months
        self.c = nn.Parameter(12 * torch.rand(heads), requires_grad=True)

    def forward(self, x, R):
        *_, n = x.shape
        b, _, t = R.shape
        num_patches = int(n/t) # number of tokens for each timepoint
        # repeat R for each head
        R = repeat(R, 'b t1 t2 -> b h t1 t2', h=self.heads)
        # repeat parameters
        a = repeat(self.a, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        c = repeat(self.c, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        # flipped sigmoid with learnable parameters
        R = 1 / (1 + torch.exp(torch.abs(a) * R - torch.abs(c)))
        # repeat values along last two dimensions according to number of patches
        R = R.repeat(1, 1, num_patches, num_patches)
        return x * R

class TDMaskedAttention(nn.Module):
    """
    Implements distance aware attention weight scaling from https://arxiv.org/pdf/2010.06925.pdf#cite.yan2019tener
    """
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.time_dist = TEM(heads=heads)
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, m, R):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        m = repeat(m, 'b d1 d2 -> b h d1 d2', h=self.heads) 
        qk = torch.matmul(q, k.transpose(-1, -2)) 
        time_scaled_dots = self.time_dist(self.relu(qk), R)

        attn = self.attend((time_scaled_dots + m)*self.scale)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class TDMaskedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TDMaskedAttention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, m, R):
        for attn, ff in self.layers:
            x = attn(x, m, R) + x
            x = ff(x) + x
        return x

class MaskedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, m):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        m = repeat(m, 'b d1 d2 -> b h d1 d2', h=self.heads)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dots = (qk + m) * self.scale # ((Q*K^T) + M )/sqrt(d) 

        # softmax can be nan when attending a padded tokens with each other
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class MaskTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MaskedAttention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x, m):
        for attn, ff in self.layers:
            x = attn(x, m) + x
            x = ff(x) + x
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PositionalTimeEncoding(nn.Module):
    # Fixed time-invariant positional encoding
    def __init__(self, dim, dropout=0.1, seq_len=5, num_patches=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len*num_patches, dim)
        position = torch.arange(0, seq_len).repeat_interleave(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, t):
        x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class AbsTimeEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, num_patches=1):
        super().__init__()
        self.num_patches = num_patches
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, dim, 2) *-(math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)
        self.dim = dim
        
    def forward(self, x, t):
        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)
        
        # repeat times into shape [b, t, dim]
        time_position = repeat(t, 'b t -> b t d', d=int(self.dim/2))
        time_position = time_position.repeat_interleave(self.num_patches, dim=1)
        pe[:, :, 0::2] = torch.sin(time_position * self.div_term.expand_as(time_position))
        pe[:, :, 1::2] = torch.cos(time_position * self.div_term.expand_as(time_position))
        x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)

class FeatViT(nn.Module):
    def __init__(self, *, num_feat, feat_dim, code_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_encoding="AbsTimeEncoding", dim_head=64, dropout=0.1):
        """
        num_feat: # nodule features + # code features 
        feat_dim: dimension of nodule features at input
        code_dim: dimension of code features at input
        dim: dimension after linear projection of input, assumed to be same for each modality
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.img_embedding = nn.Linear(feat_dim, dim)
        self.code_embedding = nn.Linear(code_dim, dim)

        # different types of positional embeddings
        # 1. PositionalEncoding: Fixed alternating sin cos with position
        # 2. AbsTimeEmb: Fixed alternating sin cos with time
        # 3. Learnable: self.time_encedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        time_enc_dict = {
            "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_encoding = time_enc_dict[time_encoding](dim, dropout=0.1, num_patches=num_feat)
        self.mod_encoding = ModalityEncoding(dim, num_modalities=2)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )
        # self.head = 'linear_head' # the property name assigned to the linear head

    def forward(self, img, code, times):

        x_img = self.time_encoding(self.img_embedding(img), times) # b t d
        x_code = self.time_encoding(self.code_embedding(code), times) # b t d
        x_img, x_code = self.mod_encoding([x_img, x_code])
        x = torch.cat((x_img, x_code), dim=1) # b (t t) d
        
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def load_from_mae(self, weights):
        # load all weights from pretrained MAE except linear head
        pretrained_headless = {k: v for k, v in weights.items() if ('encoder' in k) and (HEAD not in k)}
        model_state = self.state_dict()
        model_state.update(pretrained_headless)
        self.load_state_dict(model_state)

class ICACompression(nn.Module):
    def __init__(self, in_channel=2000, latent_channel=1000, out_channel=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, latent_channel),
            nn.ReLU(),
            nn.Linear(latent_channel, latent_channel),
            nn.ReLU(),
            nn.Linear(latent_channel, out_channel),
        )
    def forward(self, x):
        return self.net(x)


class ICAFeatViT(nn.Module):
    def __init__(self, *, feat_dim, code_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                time_encoding="AbsTimeEncoding", dim_head=64, dropout=0.1):
        """
        num_feat: # nodule features + # code features 
        feat_dim: dimension of nodule features at input
        code_dim: dimension of code features at input
        dim: dimension after linear projection of input, assumed to be same for each modality
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.img_embedding = nn.Linear(feat_dim, dim)
        self.code_embedding = ICACompression(in_channel=code_dim, out_channel=dim)

        # different types of positional embeddings
        # 1. PositionalEncoding: Fixed alternating sin cos with position
        # 2. AbsTimeEmb: Fixed alternating sin cos with time
        # 3. Learnable: self.time_encedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        time_enc_dict = {
            "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_encoding = time_enc_dict[time_encoding](dim, dropout=0.1, num_patches=1)
        self.mod_encoding = ModalityEncoding(dim, num_modalities=2)

        self.transformer = MaskTransformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )
        # self.head = 'linear_head' # the property name assigned to the linear head

    def forward(self, img, code, padding, times):
        # b, t, n, d = img.shape
        x_img = self.time_encoding(self.img_embedding(img), times) # b t (n d)
        x_code = self.time_encoding(self.code_embedding(code), times) # b t d

        x_img, x_code = self.mod_encoding([x_img, x_code])

        x = torch.cat((x_img, x_code), dim=1) # b (t t) d
        
        # mask is ordered the same as x
        padding = torch.cat((padding, padding), dim=1)
        mask = torch.einsum('bi, bj -> bij', (padding, padding))
        mask = torch.where(mask==1, 0, -9e15)

        x = self.transformer(x, mask)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def load_from_mae(self, weights):
        # load all weights from pretrained MAE except linear head
        pretrained_headless = {k: v for k, v in weights.items() if ('encoder' in k) and (HEAD not in k)}
        model_state = self.state_dict()
        model_state.update(pretrained_headless)
        self.load_state_dict(model_state)

    def freeze_img_emb(self):
        self.img_embedding.weight.requires_grad = False

    def freeze_ehr_emb(self):
        for param in self.code_embedding.parameters():
            param.requires_grad = False


class ModalityEncoding(nn.Module):
    def __init__(self, dim, num_modalities=1) -> None:
        super().__init__()
        self.mes = nn.ParameterList([nn.Parameter(torch.randn(1, dim), requires_grad=True) for i in range(num_modalities)])
        # self.dim = dim

    def forward(self, ms):
        """ms: list of inputs split by modality"""
        mes = [self.mes[i].expand(*m.shape) for i, m in enumerate(ms)] # expand each modality encoding to shape of that modality
        return [ms[i] + mes[i] for i in range(len(ms))] # return list of inputs + modality encoding
        
    def encoding(self, ms):
        return [self.mes[i].expand(*m.shape) for i, m in enumerate(ms)]


############################################################################################################
# ROI ViT + ICA Expressions
############################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ConvEmbedding(nn.Module):
    def __init__(self, block, layers, latent_dim = 128) -> None:
        super(ConvEmbedding, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm3d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.fc = nn.Linear(1024, latent_dim)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x) # spatial / 2
        x = self.maxpool(x) # spatial / 2
        x = self.layer0(x)  # spatial constant
        x = self.layer1(x) # spatial / 2

        x = self.avgpool(x) # spatial % 7
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class MultiNoduleEmbedding(nn.Module):
    """
    ResNet based embedding for multiple nodule ROIs
    img: b t n w h l
    returns: b t (n latent_dim)
    """
    def __init__(self, topk=5, feat_dim = 320, embedding_dim = 320) -> None:
        super(MultiNoduleEmbedding, self).__init__()
        assert (feat_dim/topk).is_integer(), "feat_dim must be divisible by topk"
        self.latent_dim = int(feat_dim/topk)
        self.conv_embedding = ConvEmbedding(ResidualBlock, [3, 4], latent_dim=self.latent_dim)
        self.fc = nn.Linear(feat_dim, embedding_dim)
    
    def forward(self, img):
        B, T, N = img.shape[:3] # dims of batch, time, num nodules
        
        roi = rearrange(img, 'b t n w h l -> (b t n) w h l').unsqueeze(1) # (b t n) 1 w h l
        emb = self.conv_embedding(roi) #(b t n) d
        emb = rearrange(emb, '(b t n) d -> b t (n d)', b=B, t=T, n=N)

        return self.fc(emb)


class IcaRoiTD(nn.Module):
    def __init__(self, *, feat_dim, code_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                time_encoding=None, dim_head=64, dropout=0.1):
        """
        num_feat: # nodule features + # code features 
        feat_dim: dimension of nodule features for input to join encoder
        code_dim: dimension of code features at input
        dim: dimension after linear projection of input, assumed to be same for each modality
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.img_embedding = MultiNoduleEmbedding(feat_dim=feat_dim, embedding_dim=dim)
        self.code_embedding = ICACompression(in_channel=code_dim, out_channel=dim)

        # self.time_encoding = time_enc_dict[time_encoding](dim, dropout=0.1, num_patches=1)
        self.mod_encoding = ModalityEncoding(dim, num_modalities=2)

        self.transformer = TDMaskedTransformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )
        # self.head = 'linear_head' # the property name assigned to the linear head

    def forward(self, img, code, padding, times):
        # img.shape = b t n w h l
        x_img = self.img_embedding(img) # b t (n d)
        x_code = self.code_embedding(code) # b t d

        x_img, x_code = self.mod_encoding([x_img, x_code])

        x = torch.cat((x_img, x_code), dim=1) # b (t t) d
        
        # mask is ordered the same as x
        padding = torch.cat((padding, padding), dim=1)
        mask = torch.einsum('bi, bj -> bij', (padding, padding))
        mask = torch.where(mask==1, 0, -9e15)

         # Create distance matrix from times
        b, t, *_ = img.shape
        R = torch.zeros(b, t, t, device=x.device, dtype=torch.float32)
        for n in range(b):
            for i in range(t):
                for j in range(t):
                    R[n, i, j] = torch.abs(times[n, 0] - times[n, i]) 

        x = self.transformer(x, mask, R)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def load_from_mae(self, weights):
        # load all weights from pretrained MAE except linear head
        pretrained_headless = {k: v for k, v in weights.items() if ('encoder' in k) and (HEAD not in k)}
        model_state = self.state_dict()
        model_state.update(pretrained_headless)
        self.load_state_dict(model_state)

    def freeze_img_emb(self):
        self.img_embedding.weight.requires_grad = False

    def freeze_ehr_emb(self):
        for param in self.code_embedding.parameters():
            param.requires_grad = False


############################################################################################################
# Masked Autoencoder
############################################################################################################

class MeanCosineLoss(nn.Module):
    def __init__(self, num_modalities=1) -> None:
        super().__init__()
        self.loss = nn.ModuleList([nn.CosineEmbeddingLoss() for i in range(num_modalities)])

    def forward(self, yhat, x):
        loss = 0
        for i, mod_loss in enumerate(self.loss):
            loss += mod_loss(yhat[i], x[i]) # accumulate loss from each modality
        loss = loss/len(self.loss) # simple mean of loss across modalities

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, masking_ratio=0.5, decoder_depth = 1, decoder_heads=8, decoder_dim_head=64) -> None:
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.flatten = Rearrange('b t ... d -> b (t ...) d')

        self.encoder = encoder
        self.num_tokens, self.encoder_dim = encoder.time_encoding.num_patches, encoder.time_encoding.dim
        self.num_masked = int(masking_ratio*self.num_tokens)
        self.encoder_time_enc = AbsTimeEncoding(self.encoder_dim, num_patches=self.num_tokens-self.num_masked)
        self.encoder_mod_enc = ModalityEncoding(self.encoder_dim, num_modalities=2)
        # self.encoder_time_enc = AbsTimeEncoding(self.encoder_dim)

        self.enc_to_dec = nn.Linear(self.encoder_dim, decoder_dim) if self.encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_dim = decoder_dim
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, 
            mlp_dim=decoder_dim*4, qkv_bias=False)
        self.decoder_time_enc = AbsTimeEncoding(decoder_dim, num_patches=self.num_tokens - self.num_masked)
        self.masked_decoder_time_enc = AbsTimeEncoding(decoder_dim, num_patches=self.num_masked)
        self.decoder_mod_enc = ModalityEncoding(decoder_dim, num_modalities=2)
        self.img_projection = nn.Linear(decoder_dim, self.encoder_dim)
        self.code_projection = nn.Linear(decoder_dim, self.encoder_dim)
    
    def forward(self, img, code, times):
        device = img.device

        x_img = self.encoder.img_embedding(img) # b t n d
        x_code = rearrange(self.encoder.code_embedding(code), 'b t d -> b t 1 d')
        x = torch.cat((x_img, x_code), dim=2) # (b t (n+1) d)
        encoder_mod_enc = torch.cat(self.encoder_mod_enc.encoding([x_img, x_code]), dim=2) # (b t (n+1) d)
        batch, t, num_tokens, *_ = x.shape
        # tokens = rearrange(tokens, 'b t ... d -> b (t ...) d')
        # tokens = self.encoder.time_encoding(tokens, times)
        
        # calculate tokens to be masked, get random indices, dividing it up for mask vs unmasked
        rand_indices = torch.rand(batch, t, num_tokens, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :, :self.num_masked], rand_indices[:, :, self.num_masked:]

        # get unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None, None]
        time_range = torch.arange(t, device=device)[:, None] # random masking is stratified across time
        tokens = x[batch_range, time_range, unmasked_indices]
        masked_tokens = x[batch_range, time_range, masked_indices] # get tokens to be masked for the final reconstruction loss

        # attend with encoder
        tokens = self.flatten(tokens)
        tokens = self.encoder_time_enc(tokens, times) # add time encodings
        encoder_mod_enc = rearrange(encoder_mod_enc[batch_range, time_range, unmasked_indices], 'b t ... d -> b (t ...) d')
        tokens = tokens + encoder_mod_enc # add modality encodings
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply time embeddings to unmasked tokens
        decoder_tokens = self.decoder_time_enc(decoder_tokens, times)
        decoder_mod_enc = torch.cat(self.decoder_mod_enc.encoding([self.enc_to_dec(x_img), self.enc_to_dec(x_code)]), dim=2)
        decoder_mod_enc = self.flatten(decoder_mod_enc[batch_range, time_range, unmasked_indices])
        decoder_tokens = decoder_tokens + decoder_mod_enc

        # repeat mask tokens for number of masked, and add times using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b (t n) d', b = batch, n = self.num_masked, t=t)
        mask_tokens = self.masked_decoder_time_enc(mask_tokens, times)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        # rearrange tokens to input format (b t n d)
        masked, unmasked = decoded_tokens[:, :(self.num_masked*t)], decoded_tokens[:, (self.num_masked*t):]
        masked = rearrange(masked, 'b (t n) d -> b t n d', t=t)
        # unmasked = rearrange(unmasked, 'b (t n) d -> b t n d', t=t)
        output = torch.zeros(batch, t, num_tokens, self.decoder_dim, device=device)
        output[batch_range, time_range, masked_indices] = masked
        # output[batch_range, time_range, unmasked_indices] = unmasked

        # separate modalities and compute loss on each
        yhat_img, yhat_code = self.img_projection(output[:,:,:x_img.shape[2]]), self.code_projection(output[:, :, x_img.shape[2]:])
        yhat = torch.cat((yhat_img, yhat_code), dim=2)
        yhat = yhat[batch_range, time_range, masked_indices]
        recon_loss = 1 - F.cosine_similarity(self.flatten(yhat), self.flatten(masked_tokens), dim=2)

        return recon_loss.mean(dim=1).mean()

