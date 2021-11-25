def eval_iou(b1, b2, mode="pascal"):
    """
    :param b1:	4, torch.float
            mode="pascal": (x0,y0,x1,y1)
            mode="coco": (x,y,w,h)
    :param b2: 	4, torch.float
            mode="pascal": (x0,y0,x1,y1)
            mode="coco": (x,y,w,h)
    :return: 	iou
    """
    if mode == "coco":
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
    elif mode == "pascal":
        x1, y1, w1, h1 = b1[0], b1[1], b1[2] - b1[0], b1[3] - b1[1]
        x2, y2, w2, h2 = b2[0], b2[1], b2[2] - b2[0], b2[3] - b2[1]
    xs = max(x1, x2)
    ys = max(y1, y2)
    xe = min(x1 + w1, x2 + w2)
    ye = min(y1 + h1, y2 + h2)
    intersection = (xe - xs) * (ye - ys)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union
