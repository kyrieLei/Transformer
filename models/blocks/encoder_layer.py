from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention=MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm=LayerNorm(d_model=d_model)
        self.dropout=nn.Dropout(drop_prob)

        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)

    def forward(self,x,src_mask):
        _x=x
        x=self.attention(q=x,k=x,v=x,mask=src_mask)
        x=self.dropout(x)
        x=self.norm(x+_x)

        _x=x
        x=self.ffn(x)
        x=self.dropout(x)
        x=self.norm(_x+x)

        return x


