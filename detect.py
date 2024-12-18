import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
import math
# import lanms
from datetime import datetime
# from dataset import get_rotate_mat
from model import EAST
# from manga_ocr import MangaOcr
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def non_max_suppression_np(boxes, scores, iou_threshold=0.2):
    """NMS implementation for numpy arrays."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def resize_img(img):
    '''resize image to be divisible by 32
    '''
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :] # 4 x N
    angle = valid_geo[4, :] # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0,:] += x
        res[1,:] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.8, nms_thresh=0.05):
    '''get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    score = score[0,:,:]
    xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    
    scores = boxes[:, 8]
    nms_indices = non_max_suppression_np(boxes[:, :8], scores, iou_threshold=nms_thresh)
    
    # Keep only the boxes that survived NMS
    boxes = boxes[nms_indices]

    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    '''refine boxes
    Input:
        boxes  : detected polys <numpy.ndarray, (n,9)>
        ratio_w: ratio of width
        ratio_h: ratio of height
    Output:
        refined boxes
    '''
    if boxes is None or boxes.size == 0:
        return None
    boxes[:,[0,2,4,6]] /= ratio_w
    boxes[:,[1,3,5,7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device):
    '''detect text regions of img using model
    Input:
        img   : PIL Image
        model : detection model
        device: gpu if gpu is available
    Output:
        detected polys
    '''
    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    '''plot boxes on image
    '''
    if boxes is None:
        return img
    name=1
    if not os.path.exists(os.path.join(os.getcwd(),'saved_cropped')):
        os.mkdir(os.path.join(os.getcwd(),'saved_cropped'))
    path=os.path.join(os.getcwd(),'saved_cropped')
    temp=img.copy()
    cropped_images=[]
    draw = ImageDraw.Draw(temp)
    # print(boxes)
    for box in boxes:
        img_name=str(name).zfill(4)+'.jpg'
        # plt.imshow(img.crop([box[0],box[1],box[4],box[5]]))
        # plt.show()
        try:
            cropped_img=img.crop([box[0],box[1],box[4],box[5]])
            cropped_images.append(cropped_img)
            cropped_img.save(os.path.join(path,img_name))
        except Exception as e:
            print(e)
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0),width=3)
        name+=1
    # temp.save(os.path.join(path,'ground_truth.jpg'))
    return temp,cropped_images


def detect_dataset(model, device, test_img_path, submit_path):
    '''detection on whole dataset, save .txt results in submit_path
    Input:
        model        : detection model
        device       : gpu if gpu is available
        test_img_path: dataset path
        submit_path  : submit result for evaluation
    '''
    img_files = os.listdir(test_img_path)
    img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])

    for i, img_file in enumerate(img_files):
        print('evaluating {} image'.format(i), end='\r')
        boxes = detect(Image.open(img_file), model, device)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
            f.writelines(seq)

# !wget 'https://cdn.kumacdn.club/wp-content/uploads/K/Kurameru Kagari/Chapter 05/01.jpg'
class EastModel:
    def __init__(self):
        super(EastModel).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=EAST()
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(),'east_2024-11-04.pth'),weights_only=False,map_location=self.device))

    def detect(self,img_path):
        res_img     = './res.bmp'
        self.model.eval().to(self.device)
        img = Image.open(img_path).resize((1024,1024))

        boxes = detect(img, self.model, self.device)
        plot_img,cropped_images = plot_boxes(img, boxes)
        # for cropped_img in cropped_images:
        #     print(mocr(cropped_img),flush=True)
        #     plt.imshow(cropped_img)
        #     plt.show()

model=EastModel()
model.detect("C:/Users/tirth/Desktop/manga_text_ocr-main/Atama wa Katai ga Nyotai wa Yawaraka Ganbare TS Iinchou/0010.jpg")