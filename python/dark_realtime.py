from ctypes import *
import random
import cv2
import numpy as np
import os


def randomcolor():
    '''
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color
    :return:
    '''
    color = np.random.randint(0, 256, size=[1,3])
    color = color.tolist()[0]
    return color



def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("../python/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    #arr = np.asarray(arr, dtype='float64') # add by dengjie
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)

    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    print('num', num)
    print('num.type',type(num))
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                            (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


if __name__ == "__main__":
    k = 0
    out_video = 1
    path = '../test_pic'
    net = load_net(b"/home/dengjie/dengjie/project/detection/darknet/cfg/yolov3.cfg",
                   b"/home/dengjie/dengjie/project/detection/darknet/yolov3.weights",
                   0)
    # meta = load_meta(b"home/dengjie/dengjie/project/detection/darknet/cfg/coco.data")
    meta = load_meta("/home/dengjie/dengjie/project/detection/darknet/cfg/coco.data".encode('utf-8'))
    state = 'picture'  # 检测模式选择,state = 'video','picture','real_time'
    if state == 'video' or state == 'real_time':
        if state == 'real_time':
            video = "http://admin:admin@192.168.0.13:8081"
        elif state == 'video':
            video = '/home/dengjie/dengjie/project/detection/darknet/video3.mp4'
            # video = 0
        cap = cv2.VideoCapture(video)
        #ret, img = cap.read()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        print('start detect')


        while True:
            ret, img = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            print('fps', fps)
            if ret:
                r = detect(net, meta, img)
                print(r)
                # [(b'person', 0.6372514963150024,
                # (414.55322265625, 279.70245361328125, 483.99005126953125, 394.2349853515625))]
                # 类别，识别概率，识别物体的X坐标，识别物体的Y坐标，识别物体的长度，识别物体的高度

                for i in r:
                    color = randomcolor()
                    # x, y, w, h = i[2][0], i[2][17], i[2][18], i[2][19]
                    x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
                    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    cv2.rectangle(img, pt1, pt2, color, 2)
                    cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                cv2.imshow("img", img)
                if state == 'video':
                    cv2.imwrite('/home/dengjie/dengjie/project/detection/darknet/result_frame/result_frame_%d.jpg' % k, img)
                    k += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
        if state == 'video':
            if out_video:
                img = cv2.imread('/home/dengjie/dengjie/project/detection/darknet//result_frame/result_frame_0.jpg', 1)
                isColor = 1
                FPS = 20.0
                frameWidth = img.shape[1]
                frameHeight = img.shape[0]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('/home/dengjie/dengjie/project/detection/darknet/result_video.avi', fourcc, FPS, (frameWidth, frameHeight), isColor)
                root = '/home/dengjie/dengjie/project/detection/darknet//result_frame'
                list = os.listdir(root)
                print(len(list))
                for i in range(len(list)):
                    frame = cv2.imread('/home/dengjie/dengjie/project/detection/darknet/result_frame/result_frame_%d.jpg' % i, 1)
                    out.write(frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                out.release()
                print('video has already saved.')
    else:
        test_list = os.listdir(path)
        print('test_list',test_list)
        for k in range(len(test_list)):

            img = cv2.imread('/home/dengjie/dengjie/project/detection/darknet/test_pic/'+test_list[k],1)
            #print('img',img)
            r = detect(net, meta, img)
            print(r)
            # [(b'person', 0.6372514963150024,
            # (414.55322265625, 279.70245361328125, 483.99005126953125, 394.2349853515625))]
            # 类别，识别概率，识别物体的X坐标，识别物体的Y坐标，识别物体的长度，识别物体的高度
            for i in r:
                color = randomcolor()
                print(color)
                # x, y, w, h = i[2][0], i[2][17], i[2][18], i[2][19]
                x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 3)
                cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0]+10, pt1[1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("img", img)
            cv2.imwrite('/home/dengjie/dengjie/project/detection/darknet/result_pic/result_%d.jpg' % k, img)
            cv2.waitKey()
        cv2.destroyAllWindows()