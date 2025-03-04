import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import pdb

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # model -- ColorizationModel

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    # web_dir -- './results/imagenet/test_latest'

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    scores = []
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        # data.keys() -- dict_keys(['A_l', 'A_ab', 'R_l', 'R_ab', 'ab', 'hist', 'A_paths'])
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        metrics = model.compute_scores()
        scores.extend(metrics)
        print('processing (%04d)-th image... %s' % (i, img_path))
        # opt.aspect_ratio -- 1.0
        # opt.display_winsize -- 256

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))