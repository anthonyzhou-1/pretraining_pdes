from torch import nn
from einops.layers.torch import Rearrange

class MLP_classifier(nn.Module):
    def __init__(self, config):
        super(MLP_classifier, self).__init__()
        h = config.MLP_hidden_dim

        if len(config.base_resolution) == 2:
            flatten = Rearrange('b t x -> b (t x)')
            in_dim = config.base_resolution[1] * config.time_future
        elif len(config.base_resolution) == 3:
            flatten = Rearrange('b t x y -> b (t x y)')
            in_dim = config.base_resolution[1] * config.base_resolution[2] * config.time_future

        hidden_layer = nn.Linear(in_dim, h)
        gelu = nn.GELU()
        layernorm = nn.LayerNorm(h)
        output = nn.Linear(h, config.n_classes)

        self.projector = nn.Sequential(flatten,
                                    hidden_layer,
                                    gelu,
                                    layernorm,
                                    output)
        
        print("Initialized MLP Projection Head")

    def forward(self, x):
        x = self.projector(x)
        return x
    
class CNN_classifier(nn.Module):
    def __init__(self, config):
        super(CNN_classifier, self).__init__()
        kernel_size = config.kernel_size
        channels = config.channels
        stride = config.stride
        h = config.CNN_hidden_dim

        if len(config.base_resolution) == 2:
            # Consider time as channel dim 
            conv1 = nn.Conv1d(in_channels=config.time_future, out_channels=channels, kernel_size=kernel_size, stride=stride)
            gelu = nn.GELU()
            conv2 = nn.Conv1d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, stride=stride)
            avgpool = nn.AdaptiveAvgPool1d(h) # pool spatial dimension to consistent size
            flatten = Rearrange('b c x -> b (c x)')
            output = nn.Linear(h*channels*2, config.n_classes)
            
            self.projector = nn.Sequential(conv1, # (b, t, x) -> (b, c, x)
                                        gelu,
                                        conv2,
                                        gelu,
                                        avgpool, # (b, c, x) -> (b, c, h)
                                        flatten, # (b, c, h) -> (b, c*h)
                                        output)

        elif len(config.base_resolution) == 3:
            conv1 = nn.Conv2d(in_channels=config.time_future, out_channels=channels//2, kernel_size=kernel_size, stride=stride)
            gelu = nn.GELU()
            conv2 = nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=kernel_size, stride=stride)
            conv3 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=1, stride=stride)
            flatten = Rearrange('b c x y -> b (c x y)')
            avgpool = nn.AdaptiveAvgPool2d(h)
            output = nn.Linear(h*h*channels*2, config.n_classes)

            self.projector = nn.Sequential(conv1, # (b, t, x, y) -> (b, c, x, y)
                                        gelu, 
                                        conv2,
                                        gelu,
                                        conv3,
                                        gelu,
                                        avgpool, # (b, c, x, y) -> (b, c, h, h)
                                        flatten, # (b, c, h, h) -> (b, c*h*h)
                                        output)
            
        print("Initialized CNN Projection Head")
    
    def forward(self, x):
        x = self.projector(x)
        return x

class Fixed_Future(nn.Module):
    def __init__(self, args):
        super(Fixed_Future, self).__init__()
        self.args = args
        time_window = args.time_window
        channels = args.tf_channels
        kernel_size = args.kernel_size
        stride = args.stride
        conv1 = nn.Conv2d(in_channels=time_window, out_channels=channels, kernel_size=kernel_size, stride=stride, padding="same")
        gelu = nn.GELU()
        conv2 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, stride=stride, padding="same")
        conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=1, stride=stride, padding="same")
        flatten = Rearrange('b c x y -> b x y c')
        output = nn.Linear(channels*4, 1)
        reshape_out = Rearrange('b x y 1 -> b 1 x y')

        self.projector = nn.Sequential(conv1, # (b, t, x, y) -> (b, c, x, y)
                                    gelu, 
                                    conv2,
                                    gelu,
                                    conv3,
                                    gelu,
                                    flatten, # (b, 4*c, x, y) -> (b, x, y 4*c)
                                    output, # (b, x, y, 4*c) -> (b, x, y, 1)
                                    reshape_out) # (b, x, y, 1) -> (b, 1, x, y)

    def forward(self, u):
        return self.projector(u)
    
class CNN_Decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, hidden_size=128, kernel_size=3, stride=1, padding="same"):
        super(CNN_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, 1, stride, padding)
        self.gelu = nn.GELU()
        self.rearrange = Rearrange('b c x y -> b x y c')
        self.ln = nn.LayerNorm(hidden_size*2)
        self.linear = nn.Linear(hidden_size*2, out_channels)
        self.rearrange_out = Rearrange('b x y c -> b c x y')

        print("Initialized CNN Decoder")
    
    def forward(self, x):
        x = self.conv1(x) # [b, 1, nx, ny] -> [b, hidden_size, nx, ny]
        x = self.gelu(x) 
        x = self.conv2(x) # [b, hidden_size, nx, ny] -> [b, hidden_size*2, nx, ny]
        x = self.gelu(x)
        x = self.rearrange(x) # [b, hidden_size*2, nx, ny] -> [b, nx, ny, hidden_size*2]
        x = self.ln(x)
        x = self.linear(x) # [b, nx, ny, hidden_size*2] -> [b, nx, ny, out_channels]
        x = self.rearrange_out(x) # [b, nx, ny, out_channels] -> [b, out_channels, nx, ny]

        return x