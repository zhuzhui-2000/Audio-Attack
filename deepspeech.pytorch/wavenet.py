import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=pad, dilation=dilation)
 
    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]  
        return x

    
class ResidualLayer(nn.Module):    
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)        
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)
        
   
    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)  
        fx = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        fx = self.resconv1_1(fx) 
        skip = self.skipconv1_1(fx) 
        residual = fx + x  
        #residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual
    
class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth, dilation_speed=2):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, dilation_speed**layer)
                         for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)
        
    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            #skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]
    
class WaveNet(nn.Module):
 
    def __init__(self,input_size,out_size, residual_size, skip_size, dilation_cycles, dilation_depth, dilation_size):
 
        super(WaveNet, self).__init__()
 
        self.input_conv = CausalConv1d(input_size,residual_szie, kernel_size=2)        
 
        self.dilated_stacks = nn.ModuleList(
 
            [DilatedStack(residual_size, skip_size, dilation_depth)
 
             for cycle in range(dilation_cycles)]
 
        )
 
        self.convout_1 = nn.Conv1d(skip_size, out_size, kernel_size=1)
 
        #self.convout_2 = nn.Conv1d(skip_size, out_size, kernel_size=1)
 
    def forward(self, x):
 
        x = x.permute(0,2,1)# [batch,input_feature_dim, seq_len]
 
        x = self.input_conv(x) # [batch,residual_size, seq_len]             
 
        skip_connections = []
 
        for cycle in self.dilated_stacks:
 
            skips, x = cycle(x)             
            skip_connections.append(skips)
 
        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)        
 
        # gather all output skip connections to generate output, discard last residual output
 
        out = skip_connections.sum(dim=0) # [batch,skip_size,seq_len]
 
        out = F.tanh(out)
 
        out = self.convout_1(out) # [batch,out_size,seq_len]
        out = 0.1 * F.tanh(out)
 
        #out=self.convout_2(out)
 
        out=out.permute(0,2,1)
        #[bacth,seq_len,out_size]
        return out     