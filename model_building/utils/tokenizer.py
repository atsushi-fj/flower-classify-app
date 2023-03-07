from torch import nn


class Tokenizer(nn.Module):
    """
    Creats the Convolutional Tokenizer
    """
    
    def __init__(self,
                 kernel_size=7, stride=2, padding=3,  # Kernel, Stride from table 6
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=2,  # Convs from table 6
                 n_input_channels=3,
                 n_output_channels=256,
                 in_planes=64):
        super(Tokenizer, self).__init__()

        # the numbers of filters
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers-1)] + \
                        [n_output_channels]

        self.conv_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding)
            ) for i in range(n_conv_layers)])
        
        self.flatten_layer = nn.Flatten(2, 3)  # (b, c, h, w) -> (b, d, n) 

    def forward(self, x):
        return self.flatten_layer(self.conv_blocks(x)).transpose(-2, -1)  # (b, n, d)
    