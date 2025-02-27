import torch
import torch.nn as nn
from mambapy.vim import VMamba,MambaConfig


# VM Simplified implementation
def VMamaba(d_model):
    config = MambaConfig(d_model=d_model, n_layers=1)
    return VMamba(config)

class VimBlock(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.layer = VMamaba(d_model=hidden_dims)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer(x)
        xt = x.transpose(1, 2)
        xt = xt.reshape(B, -1, H, W)
        return xt

class SVLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SVLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                VimBlock(num_hidden * 7),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,stride=stride, padding=self.padding, bias=False),
                # VimBlock(num_hidden * 4),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                # VimBlock(num_hidden * 3),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                # VimBlock(num_hidden ),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                VimBlock(num_hidden * 7)
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,stride=stride, padding=self.padding, bias=False),
                # VimBlock(num_hidden * 4)
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                VimBlock(num_hidden * 3)
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                # VimBlock(num_hidden),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class SVRNN_Model(nn.Module):
    def __init__(self, configs):
        super(SVRNN_Model, self).__init__()

        T, C, H, W = configs.in_shape
        self.num_layers = configs.num_layers
        self.num_hidden = configs.num_hidden
        self.frame_channel = C
        self.configs = configs
        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SVLSTMCell(in_channel, self.num_hidden[i], H, W, configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers  - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        B, L, H, W, C = frames_tensor.shape
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()

        #input
        input_tensor = frames[:, :self.configs.num_frames_input]
        last_frames = frames[:, self.configs.num_frames_input]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden[i], H, W], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([B, self.num_hidden[0], H, W], device=device)

        for t in range(self.configs.num_frames_input - 1):
            net = input_tensor[:, t]
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            out = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(out)

        for t in range(L-self.configs.num_frames_input):
            net = last_frames
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            out = self.conv_last(h_t[self.num_layers - 1])
            last_frames = out
            next_frames.append(out)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames






