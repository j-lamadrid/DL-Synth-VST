import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data
from params import *


class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None,
        output_padding=1,
        use_batch_norm=False,
    ):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.transpose_ops(x)


class Conv1D(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        alpha=0.2,
        shift_factor=2,
        stride=4,
        padding=11,
        use_batch_norm=False,
        drop_prob=0,
    ):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_drop:
            x = self.dropout(x)
        return x


class WaveGANGenerator(nn.Module):
    def __init__(
        self,
        model_size=64,
        ngpus=1,
        num_channels=2,
        noise_latent_dim=100,
        verbose=False,
        upsample=True,
        use_batch_norm=False,
    ):
        super(WaveGANGenerator, self).__init__()

        self.slice_len = 16384
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        latent_dim = noise_latent_dim
        self.verbose = verbose
        self.use_batch_norm = use_batch_norm

        self.dim_mul = 16

        self.fc1 = nn.Linear(latent_dim, 4 * 4 * model_size * self.dim_mul)
        self.bn1 = nn.BatchNorm1d(num_features=model_size * self.dim_mul)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4

        deconv_layers = [
            Transpose1dLayer(
                self.dim_mul * model_size,
                (self.dim_mul * model_size) // 2,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 2,
                (self.dim_mul * model_size) // 4,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 4,
                (self.dim_mul * model_size) // 8,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 8,
                (self.dim_mul * model_size) // 16,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
            ),
            Transpose1dLayer(
                (self.dim_mul * model_size) // 16,
                num_channels,
                25,
                stride,
                upsample=upsample,
            )
        ]

        self.deconv_list = nn.ModuleList(deconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            if self.verbose:
                print(x.shape)
        output = torch.tanh(self.deconv_list[-1](x))
        return output


class WaveGANDiscriminator(nn.Module):
    def __init__(
        self,
        model_size=64,
        ngpus=1,
        num_channels=2,
        shift_factor=2,
        alpha=0.2,
        verbose=False,
        use_batch_norm=False,
    ):
        super(WaveGANDiscriminator, self).__init__()

        self.slice_len = 16384
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1D(
                num_channels,
                model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                model_size,
                2 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                2 * model_size,
                4 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                4 * model_size,
                8 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                8 * model_size,
                16 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=0 if slice_len == 16384 else shift_factor,
            ),
        ]
        self.fc_input_size = 256 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)


if __name__ == "__main__":
    from torch.autograd import Variable

    G = WaveGANGenerator(
        verbose=True, upsample=True, use_batch_norm=True
    )
    out = G(Variable(torch.randn(10, noise_latent_dim)))
    print(out.shape)
    assert out.shape == (10, 1, slice_len)
    print("==========================")

    D = WaveGANDiscriminator(verbose=True, use_batch_norm=True)
    out2 = D(Variable(torch.randn(10, 1, 16384)))
    print(out2.shape)
    assert out2.shape == (10, 1)
    print("==========================")