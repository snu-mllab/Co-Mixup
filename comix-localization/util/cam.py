# @ based on InfoCAm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    def __init__(self, model, feature, linear, factor, ksize, padding):
        super().__init__()
        self.model = model
        self.feature = feature
        self.linear = linear
        self.factor = factor
        self.ksize = ksize
        self.padding = padding

    def forward(self, input, target=None):
        """
        if target is None: 
                estimate target label by inferencing 'model(input)' and generate cam based on the estimated target label.
        else:
                generate cam based on the given target

        :param input:	image # b3yx, torch.float32, 0~1 value, 
        :param target: 	target label # b, torch.int64
        :return: 		cam # b1yx, torch.float32, 0~1 value

        notes on comment:
                b # batch size
                k=1000 # num_class
                c=2048 # num of channel in the last feature
                y', x': the last feature spatial size
        """
        score = self.model(input)  # bk
        feature = self.feature(input)  # bcy'x'
        weight = self.linear.weight.clone().detach()  # kc
        channel = feature.shape[1]  # c

        if target is None:
            _, target = score.topk(1, 1, True, True)  # b1
            target = target[:, 0]  # b

        cam_weight = weight[target]  # bc

        cam = cam_weight[:, :, None, None] * feature  # bc11 * bcy'x'

        cam_filter = torch.ones(1, channel, self.ksize, self.ksize).to(input.device)
        cam = F.conv2d(cam, cam_filter, padding=self.padding)

        # upsample
        cam = F.interpolate(cam, size=[input.shape[3], input.shape[2]], mode="bicubic")

        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
