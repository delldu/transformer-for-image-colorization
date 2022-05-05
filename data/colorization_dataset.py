import os.path
from data.base_dataset import BaseDataset, get_transform
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from collections import  Counter
from tqdm import tqdm
import pdb

class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the nubmer of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # Namespace(dataroot='./dataset/', name='imagenet', gpu_ids=[0], 
        #     checkpoints_dir='./checkpoints/', input_nc=1, bias_input_nc=198, 
        #     output_nc=2, ndf=64, norm='instance', init_type='normal', 
        #     init_gain=0.02, serial_batches=True, num_threads=0, batch_size=1, 
        #     load_size=288, crop_size=256, max_dataset_size=inf, 
        #     preprocess='none', no_flip=True, display_winsize=256, 
        #     targetImage_path='./imgs/target.JPEG', referenceImage_path='./imgs/reference.JPEG', 
        #     use_D=False, epoch='latest', load_iter=0, verbose=False, suffix='', ntest=inf, 
        #     results_dir='./results/', aspect_ratio=1.0, phase='test', eval=False, num_test=50, 
        #     model='colorization', dataset_mode='colorization', direction='AtoB', 
        #     isTrain=False, A=23.0, display_id=-1)


        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = [[self.opt.targetImage_path, self.opt.referenceImage_path]]
        # self.AB_paths -- [['./imgs/target.JPEG', './imgs/reference.JPEG']]
        self.ab_constant = np.load('./doc/ab_constant_filter.npy')
        # self.ab_constant.shape -- (198, 2)

        self.transform_A = get_transform(self.opt, convert=False)
        self.transform_R = get_transform(self.opt, convert=False, must_crop=True)
        assert(opt.input_nc == 1 and opt.output_nc == 2)

        # (Pdb) self.transform_A
        # Compose(Lambda())
        # (Pdb) self.transform_R
        # Compose(
        #     Resize(size=[256, 256], interpolation=bicubic, max_size=None, antialias=None)
        #     Lambda()
        # )

    def __getitem__(self, index):
        path_A, path_R = self.AB_paths[index]
        im_A_l, im_A_ab, _ = self.process_img(path_A, self.transform_A)
        im_R_l, im_R_ab, hist = self.process_img(path_R, self.transform_R)

        im_dict = {
            'A_l': im_A_l,
            'A_ab': im_A_ab,
            'R_l': im_R_l,
            'R_ab': im_R_ab,
            'ab': self.ab_constant,
            'hist': hist,
            'A_paths': path_A
        }
        return im_dict


    def process_img(self, im_path, transform):

        weights_index = np.load('./doc/weight_index.npy')
        # (Pdb) weights_index.shape -- (198,)
        # array([ 78,  79,  80,  96,  97,  98,  99, 100, 101, 114, 115, 116, 117,
        #        118, 119, 120, 121, 122, 123, 134, 135, 136, 137, 138, 139, 140,
        #        141, 142, 143, 144, 154, 155, 156, 157, 158, 159, 160, 161, 162,
        #        163, 164, 165, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
        #        184, 185, 186, 187, 194, 195, 196, 197, 198, 199, 200, 201, 202,
        #        203, 204, 205, 206, 207, 215, 216, 217, 218, 219, 220, 221, 222,
        #        223, 224, 225, 226, 227, 228, 235, 236, 237, 238, 239, 240, 241,
        #        242, 243, 244, 245, 246, 247, 248, 249, 255, 256, 257, 258, 259,
        #        260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 275, 276, 277,
        #        278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290,
        #        296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
        #        309, 310, 311, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
        #        326, 327, 328, 329, 330, 331, 332, 337, 338, 339, 340, 341, 342,
        #        343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 357, 358, 359,
        #        360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
        #        373, 383, 384])

        im = Image.open(im_path).convert('RGB')
        im = transform(im)
        im = self.__scale_width(im, 256)
        im = np.array(im)
        im = im[:16 * int(im.shape[0] / 16.0), :16 * int(im.shape[1] / 16.0), :]
        l_ts, ab_ts, gt_keys = [], [], []
        hist_total_new = np.zeros((441,), dtype=np.float32)
        for ratio in [0.25, 0.5, 1]:
            if ratio == 1:
                im_ratio = im
            else:
                im_ratio = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
            lab = color.rgb2lab(im_ratio).astype(np.float32)

            if ratio == 1:
                ab_index_1 = np.round(lab[:, :, 1:] / 110.0 / 0.1) + 10.0
                # ab_index_1.shape -- (336, 256, 2)

                keys_t = ab_index_1[:,:,0] * 21+ ab_index_1[:,:,1]
                keys_t_flatten = keys_t.flatten().astype(np.int32)
                dict_counter = dict(Counter(keys_t_flatten))
                for k, v in dict_counter.items():
                    hist_total_new[k] += v

                hist = hist_total_new[weights_index]
                hist = hist / np.sum(hist)

            lab_t = transforms.ToTensor()(lab)
            l_t = lab_t[[0], ...] / 50.0 - 1.0
            ab_t = lab_t[[1, 2], ...] / 110.0
            l_ts.append(l_t)
            ab_ts.append(ab_t)

        # len(l_ts), l_ts[0].size(), l_ts[1].size(), l_ts[2].size()
        # (3, [1, 84, 64], [1, 168, 128], [1, 336, 256])
        # ab_ts[0].size(), ab_ts[1].size(), ab_ts[2].size()
        # ([2, 84, 64], [2, 168, 128], [2, 336, 256])
        # hist.shape -- (198,)

        return l_ts, ab_ts, hist


    def __scale_width(self, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if ow <= oh:
            if (ow == target_width):
                return img
            w = target_width
            h = int(target_width * oh / ow)
        else:
            if (oh == target_width):
                return img
            h = target_width
            w = int(target_width * ow / oh)
        return img.resize((w, h), method)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
