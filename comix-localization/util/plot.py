import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from util.ioparser import mkdir, parent


def imshowc_bbox(imgs, bboxs, nx=None, color="r", mode="pascal", path=None):
    """
    imgs: 
            if torch: b3yx, float32, 0~1 value
            elif np: byx3, float32, 0~1 value
    bboxs: bn4, int, pixel coordinate
    """
    assert len(imgs) == len(bboxs)
    assert isinstance(imgs, (np.ndarray, torch.Tensor))
    assert imgs.ndim == 4

    if isinstance(imgs, torch.Tensor):
        assert imgs.shape[1] in [3, 4]
        imgs = imgs.cpu().detach().numpy()
        imgs = imgs.transpose(0, 2, 3, 1)  # b3yx --> byx3
    elif isinstance(imgs, np.ndarray):
        assert imgs.shape[3] in [3, 4]

    bboxs2 = []
    for bbox in bboxs:
        assert isinstance(bbox, (np.ndarray, torch.Tensor))
        assert bbox.ndim == 2
        if isinstance(bbox, torch.Tensor):
            bboxs2.append(bbox.cpu().detach().numpy())
        else:
            bboxs2.append(bbox)
    bboxs = bboxs2

    if nx is None:
        nx = min(10, len(imgs))

    ny = (len(imgs) - 1) // nx + 1
    for idx, (img, bbox) in enumerate(zip(imgs, bboxs)):
        plt.subplot(ny, nx, idx + 1)

        # img
        fig = plt.imshow(img, vmin=0, vmax=1)

        # box
        ax = plt.gca()

        for bbox2 in bbox:
            if mode == "coco":
                x, y, w, h = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
            elif mode == "pascal":
                x, y, w, h = bbox2[0], bbox2[1], bbox2[2] - \
                    bbox2[0], bbox2[3]-bbox2[1]
            rect = patches.Rectangle((x, y), w, h, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    if path is None:
        plt.show()
    else:
        mkdir(parent(path))
        plt.savefig(path)
        plt.close()


#%% # ==================================================== #
# debug
# ======================================================== #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # imgs = np.arange(2*128*128*3).reshape(2,128,128,3) / (2*128*128*3)
    # bboxs = [np.array([[0,0,50,50], [10,10,60,60]]), np.array([[10,10,60,60]])]
    # imshowc_bbox(imgs, bboxs)

    # imgs = torch.arange(2*128*128*3).reshape(2,3,128,128) / (2*128*128*3)
    # bboxs = [torch.tensor([[0,0,50,50], [10,10,60,60]]), torch.tensor([[10,10,60,60]])]
    # imshowc_bbox(imgs, bboxs)

    # imgs = torch.arange(2*128*128*3).reshape(2,3,128,128) / (2*128*128*3)
    # bboxs = [torch.tensor([[100,100,110,110]]), torch.tensor([[100,100,110,110]])]
    # imshowc_bbox(imgs, bboxs)

    imgs = torch.arange(1 * 128 * 128 * 3).reshape(1, 3, 128, 128) / (1 * 128 * 128 * 3)
    bboxs = torch.tensor([[[0, 0, 50, 50], [10, 10, 60, 60]]])
    # imshowc_bbox(imgs, bboxs, path="test0.jpg")
    imshowc_bbox(imgs, bboxs)

    imgs = torch.arange(1 * 128 * 128 * 3).reshape(1, 3, 128, 128) / (1 * 128 * 128 * 3)
    bboxs = torch.tensor([[[50, 50, 100, 100]]])
    # imshowc_bbox(imgs, bboxs, path="test1.jpg")
    imshowc_bbox(imgs, bboxs)
