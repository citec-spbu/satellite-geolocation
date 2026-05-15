import torch.nn as nn



class FPN_I4(nn.Module):
    def __init__(self, input_dims, output_dims=384, **kwargs):  # 384 1
        super(FPN_I4, self).__init__()

        C3_size, C4_size, C5_size, C6_size = input_dims
        self.P6_1 = nn.Conv2d(C6_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P6_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P6_2 = nn.Conv2d(
            output_dims, output_dims, kernel_size=3, stride=1, padding=1
        )

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(
            output_dims, output_dims, kernel_size=3, stride=1, padding=1
        )

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

    def forward(self, inputs):
        C3, C4, C5, C6 = inputs

        P6_x = self.P6_1(C6)
        P6_upsampled_x = self.P6_upsampled(P6_x)
        P6_x = self.P6_2(P6_x)

        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x, P6_x]


class FPN_I3(nn.Module):
    def __init__(self, input_dims, output_dims=384, **kwargs):  # 384 1
        super(FPN_I3, self).__init__()

        C3_size, C4_size, C5_size = input_dims
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(
            output_dims, output_dims, kernel_size=3, stride=1, padding=1
        )

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, output_dims, kernel_size=3, stride=2, padding=1)

        # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=2, padding=1)

        # self.apply(weight_init)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # P6_x = self.P6(C5)
        # # panet
        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x]
