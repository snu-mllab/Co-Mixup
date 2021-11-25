import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
from util.ioparser import readpickle, join


class ImageNetTestWithCenterCropAndNormalize(data.Dataset):
    def __init__(self, root, resize=224):
        super().__init__()
        self.input = readpickle(join(root, "input.pkl"))  # b*(y,x,3)
        self.label = readpickle(join(root, "label.pkl"))  # b
        self.bbox = readpickle(join(root, "bbox.pkl"))  # b*(n,4)
        self.resize = resize
        assert len(self.input) == len(self.label) == len(self.bbox)
        self.input_transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(resize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.label[idx]
        bbox = self.bbox[idx].copy()
        y, x, _ = input.shape

        # modify bbox according to resize
        bbox[:,
             0] = np.clip(bbox[:, 0] * x,
                          (x - self.resize) / 2, x - (x - self.resize) / 2) - (x - self.resize) / 2
        bbox[:,
             1] = np.clip(bbox[:, 1] * y,
                          (y - self.resize) / 2, y - (y - self.resize) / 2) - (y - self.resize) / 2
        bbox[:,
             2] = np.clip(bbox[:, 2] * x,
                          (x - self.resize) / 2, x - (x - self.resize) / 2) - (x - self.resize) / 2
        bbox[:,
             3] = np.clip(bbox[:, 3] * y,
                          (y - self.resize) / 2, y - (y - self.resize) / 2) - (y - self.resize) / 2
        bbox = bbox.astype(np.int32)
        input = self.input_transform(input)
        return input, label, bbox


# ======================================================== #
# unit test
# ======================================================== #
if __name__ == "__main__":
    from util.all import imshowc_bbox, Denorm, FixedSubsetSampler
    import matplotlib.pyplot as plt
    import config
    import torchvision.transforms as T

    testset = ImageNetTestWithCenterCropAndNormalize(config.path_dataset["clean"])
    # testset = ImageNetTestWithCenterCropAndNormalize(config.path_dataset["rep"])
    # testset = ImageNetTestWithCenterCropAndNormalize(config.path_dataset["noise"])
    testloader = data.DataLoader(testset, sampler=FixedSubsetSampler(torch.arange(0, 10)))

    denorm = Denorm()
    for input, label, bbox_gt in testloader:
        print(input.shape, input.dtype)
        print(label.shape, label.dtype)
        print(bbox_gt.shape, bbox_gt.dtype)
        imshowc_bbox(denorm(input), bbox_gt)
        # imshowc_bbox(denorm(input), bbox_gt, path="test.png")
        print(bbox_gt)
        break
