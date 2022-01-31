
import torch
import torch.nn as nn
import torch.nn.functional as F
from sag.models.es_module import ESModule
from sag.models.nn_utils import get_activation_function


class Encoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=False,
            bidirectional=False,
        )

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        print(gru_out.shape)
        print(hidden.shape)
        
        hidden.squeeze_(0)
        
        return gru_out, hidden


class Decoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size):
        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.attention = False

    def forward(self, y, prev_hidden):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class Model(ESModule):
    def __init__(self, input_feature_len, hidden_size, output_size=24 ):
        super().__init__()
        self.encoder = Encoder(input_feature_len=input_feature_len, hidden_size=hidden_size)
        self.decoder_cell = Decoder(input_feature_len=input_feature_len, hidden_size=hidden_size)
        self.output_size = output_size

    def forward(self, xb, yb=None):
        decoder_input = xb[-1]
        input_seq = xb[0]
        if len(xb) > 2:
            encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
        else:
            encoder_output, encoder_hidden = self.encoder(input_seq)
        
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        
        return outputs


if __name__ == "__main__":
    x = torch.randn(120, 1)
    encoder = Encoder(input_feature_len=1)
    out, hidden = encoder(x)
    decoder = Decoder(input_feature_len=1, hidden_size=100)
    decoder(out, hidden)

