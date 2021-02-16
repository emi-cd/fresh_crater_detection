import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from planetaryimage import PDS3Image
from PIL import Image

import download as dl
import const


class TemporalPair():
    MAX = 25000
    registered = False
    transed = False
    dif_place = []
    R = 3000
    over = 1000
    R_min = 250
    over_min = 50


    def __init__(self, before, after):
        self.before = before
        self.after = after
        self.all_num = 0
        self.detect_num = 0
        self.M = self.calc_M()
        self.trans = self.trans_latlon(self.before.image, self.after.image)

    def make_dif(self, path='', output=True):
        if not self.registered:
            self.M = self.calc_M()
        if not self.transed:
            self.trans = self.trans_latlon(self.before.image, self.after.image)
        if output:
            self.trans_regist(self.before.image, self.trans, path)


    def trans_latlon(self, image_b, image_a):
        if len(self.M) == 1: 
            img = cv2.warpPerspective(image_a, self.M[0], (image_b.shape[1], image_b.shape[0]))
        else:
            img_1 = cv2.warpPerspective(image_a[:self.MAX], self.M[0], (image_b.shape[1], image_b.shape[0]))
            img_2 = cv2.warpPerspective(image_a[self.MAX:], self.M[1], (image_b.shape[1], image_b.shape[0]))
            img = img_1 + img_2
        self.transed = True
        return img


    def trans_regist(self, image_b, image_a, output_path, debug=False):
        path = output_path + self.before.file_name + '/'
        os.makedirs(path, exist_ok=True)
        for i in range(0, image_b.shape[0]-self.over, self.R):  
            img1_big = np.array(image_b[i: i+self.R+self.over], dtype=np.int16)
            img2_big = np.array(image_a[i: i+self.R+self.over], dtype=np.int16)
            if len(img2_big[np.where(img2_big > 0)]) == 0 or np.count_nonzero(img2_big) < img2_big.shape[0]*img2_big.shape[1]*0.3:
                continue
            try:
                img_translation_big = surf_flann(img2_big, img1_big)
            except :
                # print(i, sys.exc_info())
                continue
            for j in range(0, img1_big.shape[0]-self.over_min, self.R_min):
                for k in range(0, img1_big.shape[1]-self.over_min, self.R_min):
                    img1 = np.array(img1_big[j: j+self.R_min+self.over_min, k: k+self.R_min+self.over_min], dtype=np.uint16)
                    img2 = np.array(img_translation_big[j: j+self.R_min+self.over_min, k: k+self.R_min+self.over_min], dtype=np.uint16)
                    if(len(img2[np.where(img2 > 0)]) == 0 or np.count_nonzero(img2) < img2.shape[0]*img2.shape[1]*0.3):
                        continue
                    try:
                        img_translation = surf_flann(img2, img1)
                    except :
                        # print(i+j, k, sys.exc_info())
                        continue
                    # save_as_plt([img1, img_translation], path+ str(i+j)+ '-' +str(k)+'-'+self.after.file_name)
                    kernel = np.ones((5,5),np.uint8)
                    mask = np.zeros(img_translation.shape, dtype=np.uint8)
                    mask[np.where(img_translation!=0)] = 255
                    mask[0, :] = 0
                    mask[-1, :] = 0
                    mask[:, 0] = 0
                    mask[:, -1] = 0
                    mask = cv2.erode(mask, kernel, iterations = 1)
                    img_translation[np.where(mask==0)] = 0
                    img1[np.where(mask==0)] = 0

                    img_diff, img_diff_th = get_diff(img1, img_translation)
                    self.all_num += 1
                    img_diff_th[0, :] = 0
                    img_diff_th[-1, :] = 0
                    img_diff_th[:, 0] = 0
                    img_diff_th[:, -1] = 0
                    if np.sum(img_diff_th) > 0:
                        self.dif_place.append((i+j, k))
                        save_as_plt([img1, img_translation, img_diff, img_diff_th], path+str(i+j)+'-'+str(k)+'-'+self.after.file_name)
                            # save_for_check([img1, img_translation, img_diff], path+str(i+j)+'-'+str(k)+'-'+self.after.file_name)
                        self.detect_num += 1
                            # save_as_tiff([img1, img_translation, img_diff, img_diff_th], './test/'+str(i+j)+'-'+str(k)+'-'+self.after.file_name)
        print("Detected image number: {} / {}".format(self.detect_num, self.all_num))


    def _calc_another_M(self, img, M):
        origin = np.float64([[img.shape[1], self.MAX], [img.shape[1], img.shape[0]+self.MAX], [0, img.shape[0]+self.MAX], [0, self.MAX]])
        tmp = []
        for x, y in origin:
            lst = M.dot([x, y, 1])
            tmp.append([lst[0]/lst[2], lst[1]/lst[2]])
        pts1 = np.float32([[img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]], [0, 0]])
        pts2 = np.float32([tmp[0], tmp[1], tmp[2], tmp[3]])
        return cv2.getPerspectiveTransform(pts1, pts2)


    def calc_M(self):
        pos1 = np.array(self.before.pos)
        img1 = self.before.image
        img1_line_samples = self.before.file.label['IMAGE']['LINE_SAMPLES']
        pos2 = np.array(self.after.pos)
        img2 = self.after.image
        img2_line_samples = self.after.file.label['IMAGE']['LINE_SAMPLES']
        self.registered = True

        pts1 = np.float32([pos1[2:4][::-1],pos1[4:6][::-1], pos1[6:8][::-1], pos1[8:][::-1]])
        pts2 = np.float32([[img1_line_samples-self.before.left_padding, 0],[img1_line_samples-self.before.left_padding, img1.shape[0]],\
                [0-self.before.left_padding, img1.shape[0]],[0-self.before.left_padding, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        tmp = []
        for i in range(2, len(pos2), 2):
            lst = M.dot([pos2[i+1], pos2[i], 1])
            tmp.append([lst[0]/lst[2], lst[1]/lst[2]])

        pts1 = np.float32([[img2_line_samples-self.after.left_padding, 0],[img2_line_samples-self.after.left_padding, img2.shape[0]],\
                [0-self.after.left_padding, img2.shape[0]],[0-self.after.left_padding, 0]])
        pts2 = np.float32([tmp[0], tmp[1], tmp[2], tmp[3]])
        M = [cv2.getPerspectiveTransform(pts1, pts2)]

        if not img2.shape[0] < self.MAX:
            M.append(self._calc_another_M(img2[self.MAX:], M[0]))
        
        return M


    def extract(self, x, y, output_path='./test', save=True):
        image_b, image_a =  self.before.image, self.trans
        i = (y // self.R) * self.R
        img1_big = np.array(image_b[i: i+self.R+self.over], dtype=np.int16)
        img2_big = np.array(image_a[i: i+self.R+self.over], dtype=np.int16)
        try:
            img_translation_big = surf_flann(img2_big, img1_big)
        except :
            # print(i, sys.exc/_info())
            return 0
        j = y - i 
        k = x
        img1 = np.array(img1_big[j: j+self.R_min+self.over_min, k: k+self.R_min+self.over_min], dtype=np.uint16)
        img2 = np.array(img_translation_big[j: j+self.R_min+self.over_min, k: k+self.R_min+self.over_min], dtype=np.uint16)
        try:
            img_translation = surf_flann(img2, img1)
        except :
            print('*', i+j, k, sys.exc_info())
            return 0
        kernel = np.ones((5,5),np.uint8)
        mask = np.zeros(img_translation.shape, dtype=np.uint8)
        mask[np.where(img_translation!=0)] = 255
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        mask = cv2.erode(mask, kernel, iterations = 1)
        img_translation[np.where(mask==0)] = 0
        img1[np.where(mask==0)] = 0

        img_diff, _ = get_diff(img1, img_translation)
        if save:
            save_for_check([img1, img_translation, img_diff], output_path)
        else:
            return [img1, img_translation, img_diff]


    def display(self, center=[0,0], r=1000):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(131)
        ax.imshow(self.before.image[center[0]:center[0]+r, center[1]:center[1]+r], cmap='gray', vmin = 0, vmax=255)
        ax = fig.add_subplot(132)
        ax.imshow(self.trans[center[0]:center[0]+r, center[1]:center[1]+r], cmap='gray', vmin = 0, vmax=255)
        # ax = fig.add_subplot(133)
        # ax.imshow(self.dif_img[center[0]:center[0]+r, center[1]:center[1]+r], cmap='gray', vmin = 0, vmax=255)
        plt.show()


# For LEOC NAC
class NacImage():
    label = {
        'Center_latitude': 0, 'Center_longitude': 1, 'Upper_right_latitude': 2, 'Upper_right_longitude': 3, 'Lower_right_latitude': 4, 
        'Lower_right_longitude': 5, 'Lower_left_latitude': 6, 'Lower_left_longitude': 7,  'Upper_left_latitude': 8, 'Upper_left_longitude': 9
    }

    def __init__(self, data, pos=None, pos_file_name=None, img=True):
        self.data = data.to_dict()
        self.file_name = self.data['PRODUCT_ID'].replace('"', '')
        self.pos = data.values.tolist()[27:37]
        # self.pos = data.values.tolist()[28:38]
        self.pos = [float(arg) for arg in self.pos]
        self.resolution = float(self.data['RESOLUTION'])
        self.scaled_pixel_width = float(self.data['SCALED_PIXEL_WIDTH'])
        self.scaled_pixel_height = float(self.data['SCALED_PIXEL_HEIGHT'])
        self.image_lines = float(self.data['IMAGE_LINES'])
        self.line_samples = float(self.data['LINE_SAMPLES'])

        if img:
            self.file, self.image, self.left_padding, self.right_padding = self.get_img_from_IMG(self.file_name)
        if self.pos[self.label['Upper_right_longitude']] < self.pos[self.label['Upper_left_longitude']]:
            if img:
                self.image = np.fliplr(self.image)
            self.pos[2:] =   self.pos[8], self.pos[9], self.pos[6], self.pos[7], self.pos[4], self.pos[5], self.pos[2], self.pos[3]
        if self.pos[self.label['Upper_left_latitude']] < self.pos[self.label['Lower_left_latitude']]:
            if img:
                self.image = np.flipud(self.image)
            self.pos[2:] = self.pos[4], self.pos[5], self.pos[2], self.pos[3], self.pos[8], self.pos[9], self.pos[6], self.pos[7]


    def __str__(self):
        return 'NAME:\t{}\nSTART_TIME:\t{}\nRESOLUTION:\t{}\nPHASE_ANGLE:\t{}\nSIZE:\t{}*{}\nPOS:\t{}'.format(
            self.file_name, self.data['START_TIME'], self.data['RESOLUTION'], 
            self.data['PHASE_ANGLE'], self.data['IMAGE_LINES'], self.data['LINE_SAMPLES'], self.pos)


    def display(self):
        """
        Show self.image
        """
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(self.image, cmap='gray', vmin = 0, vmax=255)
        plt.show()


    def read_NAC(self, file_name):
        nac_paths = const.NAC_IMAGE_PATH.split(';')
        for path in nac_paths:
            if os.path.exists(path + file_name + '.IMG'):
                print(file_name)
                file = PDS3Image.open(path + file_name + '.IMG')
                return file
        raise ValueError('There are no {}!'.format(file_name))


    def get_img_from_IMG(self, file_name):
        file = self.read_NAC(file_name)
        left_padding, right_padding = 0, 0
        if file.label['PRODUCT_TYPE']  == 'CDR':
            for i in range(file.image.shape[1]):
                if len(np.where(file.image[:, i] != file.label['IMAGE']['NULL'])[0]) > 0:
                    left_padding = i
                    break
            for i in reversed(range(file.image.shape[1]-1)):
                if len(np.where(file.image[:, i] != file.label['IMAGE']['NULL'])[0]) > 0:
                    right_padding = file.image.shape[1] - i - 1
                    break
        else:
            sys.exit(0)
        return file, file.image[:, left_padding: -right_padding], left_padding, right_padding


# Mapping
def convert_to_uint8(img, std=10, ave=100):
    img = np.asarray(img)
    if len(img[np.where(img > 0)]) > 0:
        s = img[np.where(img > 0)].std()
        m = img[np.where(img > 0)].mean()
        img = (img - m) / s * std + ave
    img = np.clip(img, 0, 255)
    return np.array(img, dtype=np.uint8)


def perspective_trans(img1, kp1, img2, kp2, good):
    MIN_MATCH_COUNT = 10
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        img_translation = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        return img_translation
    raise ValueError('Not enough matches are found - {}/{}'.format(len(good),MIN_MATCH_COUNT))


def orb_flann(img1, img2):
    orb = cv2.ORB_create()
    img1_ = convert_to_uint8(img1)
    img2_ = convert_to_uint8(img2)
    kp1, des1 = orb.detectAndCompute(img1_, None)
    kp2, des2 = orb.detectAndCompute(img2_, None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 12, # 12 6
                       key_size = 20,     # 20 12
                       multi_probe_level = 2) #2 1
    search_params = dict(checks=50)  
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for p in matches:
        if len(p) == 2:
            if p[0].distance < 0.7*p[1].distance:
                good.append([p[0]])
        elif len(p) == 1:
            good.append([p[0]])

    img_translation = perspective_trans(img1, kp1, img2, kp2, good)
    return img_translation


def sift_flann(img1, img2, checks=50, save=None, debug=False):
    sift = cv2.xfeatures2d.SIFT_create()
    img1_ = convert_to_uint8(img1)
    img2_ = convert_to_uint8(img2)
    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for p in matches:
        if len(p) == 2:
            if p[0].distance < 0.7*p[1].distance:
                good.append([p[0]])
        elif len(p) == 1:
            good.append([p[0]])

    img_translation = perspective_trans(img1, kp1, img2, kp2, good)
    return img_translation


def surf_flann(img1, img2, checks=50, save=None, debug=False):
    sift = cv2.xfeatures2d.SURF_create()
    img1_ = convert_to_uint8(img1)
    img2_ = convert_to_uint8(img2)
    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for p in matches:
        if len(p) == 2:
            if p[0].distance < 0.6*p[1].distance:
                good.append([p[0]])
        elif len(p) == 1:
            good.append([p[0]])

    img_translation = perspective_trans(img1, kp1, img2, kp2, good)
    return img_translation



def save_as_tiff(imgs, path='./test'):
    stack = []
    for img in imgs:
        stack.append(Image.fromarray(img))
    stack[0].save(path + '.tif', compression="tiff_deflate", save_all=True, append_images=stack[1:])


def save_as_plt(imgs, path='./test'):
    fig = plt.figure(figsize=(28,7))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i+1)
        ax.imshow(img, cmap='gray')
    plt.savefig(path + '.png')
    plt.close()


def save_for_check(imgs, path='./test'):
    stack = []
    img1 = convert_to_uint8(imgs[0])
    stack.append(Image.fromarray(255 - img1))
    img2 = convert_to_uint8(imgs[1])
    stack.append(Image.fromarray(255 - img2))
    img3 = convert_to_uint8(imgs[2], std=10, ave=0)
    stack.append(make_mask(img2, img3))
    stack.append(Image.fromarray(255 - img3))
    stack[0].save(path + '.tif', compression="tiff_deflate", save_all=True, append_images=stack[1:])


def make_mask(after, dif):
    height, width = dif.shape # 幅・高さ・色を取得
    dif_mask = np.zeros((height, width, 3), dtype = "uint8")
    dif_mask[:,:,0] = dif
    after_cl = np.zeros((height, width, 3), dtype = "uint8")
    after_cl[:,:,0] = after
    after_cl[:,:,1] = after
    after_cl[:,:,2] = after
    blended = cv2.addWeighted(src1=after_cl,alpha=0.5,src2=dif_mask, beta=0.5, gamma=0)
    return Image.fromarray(blended)


def get_diff(img1, img2):
    img_diff = cv2.absdiff(img1, img2)
    _, img_diff_th = cv2.threshold(img_diff, img_diff.mean()+img_diff.std()*3, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    img_diff_th = cv2.morphologyEx(img_diff_th, cv2.MORPH_OPEN, kernel)
    return img_diff, img_diff_th


def read_nac(path):
    file = PDS3Image.open(path + '.IMG')
    if file.label['PRODUCT_TYPE']  == 'CDR':
        for i in range(file.image.shape[1]):
            if len(np.where(file.image[:, i] != file.label['IMAGE']['NULL'])[0]) > 0:
                left_padding = i
                break
        for i in reversed(range(file.image.shape[1]-1)):
            if len(np.where(file.image[:, i] != file.label['IMAGE']['NULL'])[0]) > 0:
                right_padding = file.image.shape[1] - i - 1
                break
    return file.image[:, left_padding: -right_padding]