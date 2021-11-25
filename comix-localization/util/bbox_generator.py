# @ based on InfoCAM
import numpy as np
import cv2
import torch
import torch.nn as nn

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


class BboxGenerator(nn.Module):
    def __init__(self, th):
        """
        th: 0~1 threshold
        """
        super().__init__()
        self.th = th

    def generate_bbox(self, cam, thr_val):
        '''
        cam: np.float32 [y,x,1]
        thr_val: float value 0~1
        return: [1,4], torch.float32
                the last for channel means x0,y0,x1,y1
        '''
        gray_heatmap = np.clip((cam - cam.min()) / (cam.max() - cam.min()) * 255, 0,
                               255).astype(np.uint8)  # np.uint8, yx, 0~255

        _, thr_gray_heatmap = cv2.threshold(gray_heatmap, int(thr_val * np.max(gray_heatmap)), 255,
                                            cv2.THRESH_BINARY)

        contours = cv2.findContours(image=thr_gray_heatmap,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            estimated_bbox = torch.tensor([[x, y, x + w, y + h]]).float()
        else:
            estimated_bbox = torch.tensor([[0, 0, 1, 1]]).float()

        return estimated_bbox

    def forward(self, cam):
        """
        cam: [b,1,y,x], torch.float32
        return: [b,1,4], torch.float32
                the last for channel means x0,y0,x1,y1
        """
        device = cam.device
        out = []
        for i in range(cam.shape[0]):
            bbox = self.generate_bbox(cam[i].detach().cpu().numpy().transpose(1, 2, 0),
                                      self.th)  # torch.float32 [1,4]
            out.append(bbox.to(device))
        out = torch.stack(out)  # torch.float32 [b,1,4]
        return out
