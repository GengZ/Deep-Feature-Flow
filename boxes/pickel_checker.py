import cPickle
import pprint
import sys
import os

classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']

def convert_path(src = './ImageNetVID_VID_val_videos_0_draw.pkl', dst = './boxes.pkl'):
    """
    changing path from './*' to '../*'
    changing data format
    output: dict{image_name_with_abs_path: list(boxes_per_image)}
    """

    import collections

    det_result_file = dst

    print 'reading files...'
    with open(src) as f:
        detections = cPickle.load(f)

    print 'collect by image'
    det_dict = collections.OrderedDict()
    total_count = len(detections)

    for count, box in enumerate(detections):
        img_name = '.' + box[0]     # convert path from './*' to '../*'
        box_info = box[1:]

        if len(det_dict) == 0 or img_name != det_dict.keys()[-1]:
            det_dict[img_name] = list()
            det_dict[img_name].append(box_info)
        else:
            det_dict[img_name].append(box_info)

        if count % 1000 == 0:
            print 'converting {}/{} detections'.format(count, total_count)

    print 'writing results'
    with open(det_result_file, 'wb') as f:
        cPickle.dump(det_dict, f, protocol = -1)
    print 'done!'

def list_one_item(fname, index):

    with open(fname) as f:
        detections = cPickle.load(f)
    pprint.pprint(detections[index])

def mkdir_per_seq(detections, output_dir = './result_sequence'):
    """
    boosting version of convert_path
    TO BE DONE ...
    """
    # one file
    # one image
    if not os.exists(output_dir):
        os.mkdir(output_dir)

    prev_name = None
    flag = 0
    for det in detections:
        if flag == 0:
            flag = 1
            prev_name = det[0].split('/')[-2]
            cur_seq_det = dict()

        if det[0].split('/')[-2] != prev_name:
            cur_seq_path = os.path.join(output_dir, det[0].split('/')[-2])
            if not os.path.exists(cur_seq_path):
                os.mkdir(cur_seq_path)
            cPickle.dump(cur_seq_det, cur_seq_path, protocol = -1 )
            print 'writing {}'.format(det[0].split('/')[-2])
            cur_seq_det = dict()

        # cur_seq_det{det[0]} =

    # for det in detections:
    #     cur_seq_name = detections[0].split('/')[-2]
    #     cur_seq_path = os.path.join(output_dir, cur_seq_name)
    #     if not os.path.exists(cur_seq_path):
    #         os.mkdir(cur_seq_path)
    #         flag = 1


def big_draw(detections, output_dir = './result_sequence'):
    "draw boxes on all pics"
    import cv2

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print 'creating root_path: {}'.format(output_dir)

    snippets_num = 0
    for frame in detections:
        seq_name = frame.split('/')[-2]
        img_name = frame.split('/')[-1]

        if not os.path.exists(os.path.join(output_dir, seq_name)):
            os.mkdir(os.path.join(output_dir, seq_name))
            snippets_num += 1
            print 'drawing seq: {}'.format(seq_name)

        write_path = os.path.join(output_dir, seq_name, img_name)

        boxes = detections[frame]
        # if len(boxes) > 1:
            # print frame
            # pprint.pprint(boxes)
        img = draw_all_detection(frame, boxes, 1)
        cv2.imwrite(write_path, img)

    print '-' * 30
    print 'totally {} snippets processed'.format(snippets_num)
    print 'totally {} images drawn'.format(len(detections))
    print '-' * 30

def draw_all_detection(img_name, detections, scale, threshold=0, class_names=None ):
    """
    visualize all detections in one image
    change class_index to class_name
    :param im_name: abs PATH of img_name
    :param detections: list(class, box, score)
    :param scale: deseted param, scale boxes and image accordingly
    :param threshold: vis boxes above threshold
    :param class_names: string class replacing detections[0]
    :return: image_with_boxes
    """

    import cv2
    import random

    color_white = (255, 255, 255)
    im = cv2.imread(img_name)
    # for j, name in enumerate(class_names):
    #     if name == '__background__':
    #         continue
    for det in detections:
        bbox = det[1:5] # * scale
        score = det[-1]
        cl = classes[det[0]]
        # if score < threshold:
        #     continue
        bbox = map(int, bbox)
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        cv2.putText(im, '%s %.3f' % (cl, score), (bbox[0], bbox[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def check_boxes_per_img(src = './boxes.pkl'):
    import cPickle

    with open(src) as f:
        boxes_per_img = cPickle.load(f)

    # print type(boxes_per_img)
    # sys.exit()
    for cnt, k in boxes_per_img:
        if len(boxes_per_img[k]) > 1:
            print '{}:\t{}'.format(k, boxes_per_img[k])

if __name__ == "__main__":
    # convert_path()
    with open('./boxes.pkl') as f:
        detections = cPickle.load(f)
    big_draw(detections)
    # for cnt, det in enumerate(detections):
    #     if cnt < 5:
    #         pprint.pprint(det)
    #         pprint.pprint(detections[det])
    # check_boxes_per_img()
