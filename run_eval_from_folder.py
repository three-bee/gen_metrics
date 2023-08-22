from PIL import Image
import os
import argparse
import torch
import torchvision.transforms as trans
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from tqdm import tqdm
import glob
import numpy as np

from pytorch_msssim import MS_SSIM

from model_irse import IR_101
from mtcnn.mtcnn import MTCNN
from lpips.lpips import LPIPS
from imresize import imresize

from inception import build_inception_model
from fid import compute_fid
from cleanfid import fid

class ImageDataset(Dataset):
    """
    Basic image loader.
    Can preload all the dataset to RAM via multiprocessing, or load images one by one in __getitem__.
    """
    def __init__(self, opt, preload=True, use_pool=True, return_names=False):
        r_path = opt.real_path
        f_path = opt.fake_path
        self.real_downscale_rate = opt.downscale_rate
        self.use_pool = use_pool
        self.preload = preload
        self.return_names = return_names

        self._real_img_names = self.list_all_imgs(r_path) #self.get_names(r_path)
        self._fake_img_names = self.list_all_imgs(f_path) #self.get_names(f_path)

        if self.preload:
            self.real_img_list = [None] * len(self._real_img_names)
            self.fake_img_list = [None] * len(self._fake_img_names)
            self.load_real_imgs_to_RAM()
            self.load_fake_imgs_to_RAM()

        assert len(self._real_img_names) == len(self._fake_img_names)

    def __getitem__(self, item):
        idx = item % len(self)
        if self.preload:
            img_real = self.real_img_list[idx]
            img_fake = self.fake_img_list[idx]
        else:
            img_real = self.loader_real(self._real_img_names[idx])
            img_fake = self.loader_fake(self._fake_img_names[idx])
        
        if self.return_names:
            return img_real.copy(), img_fake.copy(), self._real_img_names[idx], self._fake_img_names[idx]
        else:
            return img_real.copy(), img_fake.copy()

    def __len__(self):
        return len(self._real_img_names)

    @staticmethod
    def get_names(folder):
        img_names = sorted(glob.glob(folder + '/*.png') +
                           glob.glob(folder + '/*.PNG') +
                           glob.glob(folder + '/*.jpg') +
                           glob.glob(folder + '/*.jpeg') +
                           glob.glob(folder + '/*.JPG') +
                           glob.glob(folder + '/*.JPEG'))
        return img_names
    
    @staticmethod
    def list_all_imgs(out_path, ext='.png'):
        img_paths = []
        for root, dirs, files in os.walk(out_path):
            for file in files:
                if file.endswith(ext):
                    img_paths.append(os.path.join(root, file))
        return sorted(img_paths)

    def loader_real(self, path):
        if self.real_downscale_rate is not None:
            return imresize(np.array(Image.open(path)),
                            scalar_scale=1 / self.real_downscale_rate).transpose(2, 0, 1)
        else:
            return np.array(Image.open(path)).transpose(2, 0, 1)

    @staticmethod
    def loader_fake(path):
        return np.array(Image.open(path)).transpose(2, 0, 1)

    def load_real_imgs_to_RAM(self):
        if self.use_pool:
            pool = Pool(processes=os.cpu_count())
            for i, val in tqdm(enumerate(pool.imap(self.loader_real, self._real_img_names), 0),
                               desc="Preloading real images",
                               total=len(self._real_img_names)):
                self.real_img_list[i] = val
            pool.close()
            pool.join()
        else:
            for i in tqdm(range(len(self._real_img_names)), desc="Preloading real images"):
                self.real_img_list[i] = self.loader_real(self._real_img_names[i])

    def load_fake_imgs_to_RAM(self):
        if self.use_pool:
            pool = Pool(processes=os.cpu_count())
            for i, val in tqdm(enumerate(pool.imap(self.loader_fake, self._fake_img_names), 0),
                               desc="Preloading fake images",
                               total=len(self._fake_img_names)):
                self.fake_img_list[i] = val
            pool.close()
            pool.join()
        else:
            for i in tqdm(range(len(self._fake_img_names)), desc="Preloading fake images"):
                self.fake_img_list[i] = self.loader_fake(self._fake_img_names[i])

class Metrics:
    """
    Evaluate metrics given two image folders.
    Images are in the range [0,1] and in float32.
    """
    def __init__(self, opt, view_list=None):
        curr_face_ckpt = opt.curr_face_ckpt

        self.lpips_module = LPIPS(net='alex').cuda()
        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
        
        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(curr_face_ckpt, map_location=torch.device('cpu')))
        self.facenet.eval()
        self.mtcnn = MTCNN()

        self.inception_model = build_inception_model(align_tf=True).cuda()
        self.fake_inception_feats = []
        self.real_inception_feats = []
        if view_list is not None:
            self.fake_inception_feats_multiview = {view:[] for view in view_list}
            self.real_inception_feats_multiview = {view:[] for view in view_list}

        self.id_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1] --> [0,1]
        ])
        self.ds_to_256 = torch.nn.AdaptiveAvgPool2d((256,256))

    # TODO: MTCNN backbone uses PIL which prevents from using .cuda() & larger batches than size 1
    def calc_id(self, X, Y):
        X = F.to_pil_image(X)
        Y = F.to_pil_image(Y)
        with torch.no_grad():
            X, _ = self.mtcnn.align(X)

        with torch.no_grad():
            try:
                X_id = self.facenet(self.id_transform(X).unsqueeze(0))[0]
            except:
                return None

        with torch.no_grad():
            Y, _ = self.mtcnn.align(Y)

        with torch.no_grad():
            try:
                Y_id = self.facenet(self.id_transform(Y).unsqueeze(0))[0]
            except:
                return None
        return float(X_id.dot(Y_id))

    def calc_lpips(self, X, Y):
        with torch.no_grad():
            return float(self.lpips_module(X, Y))

    def calc_ms_ssim(self, X, Y):
        with torch.no_grad():
            return float(self.ms_ssim_module(X, Y))

    def calc_mse(self, X, Y):
        return float(torch.mean(torch.square(X - Y)))

    def extract_inception_feats(self, X, Y, current_view=None):
        with torch.no_grad():
            if current_view is None:
                self.real_inception_feats.append(self.inception_model(X).cpu().numpy())
                self.fake_inception_feats.append(self.inception_model(Y).cpu().numpy())
            else:
                self.real_inception_feats_multiview[current_view].append(self.inception_model(X).cpu().numpy())
                self.fake_inception_feats_multiview[current_view].append(self.inception_model(Y).cpu().numpy())
    
    def calc_fid(self, current_view=None):
        if current_view is None:
            self.real_inception_feats = np.array(self.real_inception_feats).reshape(-1,2048)
            self.fake_inception_feats = np.array(self.fake_inception_feats).reshape(-1,2048)
            return compute_fid(self.fake_inception_feats, self.real_inception_feats)
        else:
            self.real_inception_feats_multiview[current_view] = np.array(self.real_inception_feats_multiview[current_view]).reshape(-1,2048)
            self.fake_inception_feats_multiview[current_view] = np.array(self.fake_inception_feats_multiview[current_view]).reshape(-1,2048)
            return compute_fid(self.fake_inception_feats_multiview[current_view], self.real_inception_feats_multiview[current_view])

def calc_metrics(opt):
    scores = {'id': [],
              'ms_ssim': [],
              'mse': [],
              'lpips': [],
              'fid': []}

    test_dataset = ImageDataset(opt,
                                preload=False,
                                use_pool=True,
                                return_names=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)  # Keep batch=1 for now because of MTCNN
    metrics = Metrics(opt)

    for real, fake in tqdm(test_dataloader):
        real, fake = real.type(torch.float32) / 255, fake.type(torch.float32) / 255
        if real.shape != fake.shape:
            fake = metrics.ds_to_256(fake)
            
        scores['id'].append(metrics.calc_id(real.squeeze(0), fake.squeeze(0)))

        real, fake = real.cuda(), fake.cuda()
        scores['ms_ssim'].append(metrics.calc_ms_ssim(real, fake))
        scores['mse'].append(metrics.calc_mse(real, fake))
        scores['lpips'].append(metrics.calc_lpips(real, fake))
        
        metrics.extract_inception_feats(real, fake)

    scores['fid'].append(metrics.calc_fid())
    #scores['fid'].append(fid.compute_fid(opt.real_path, opt.fake_path, device=torch.device('cpu')))

    scores = {key: sum(value) / len(value) for key, value in scores.items()}
    print(scores)

def calc_metrics_multiview(opt, 
                           views=['left_60', 'left_30', 'front', 'right_30', 'right_60', 'down', 'top']):
    scores = {key:{'id': [],
                   'ms_ssim': [],
                   'mse': [],
                   'lpips': [],
                   'fid': []} for key in views}

    test_dataset = ImageDataset(opt,
                                preload=False,
                                use_pool=True,
                                return_names=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)  # Keep batch=1 for now because of MTCNN
    metrics = Metrics(opt, view_list=views)

    for real, fake, real_name, fake_name in tqdm(test_dataloader):
        real_name = real_name[0]
        fake_name = fake_name[0]
        assert any([view_str in real_name for view_str in views]) and any([view_str in fake_name for view_str in views])
        
        current_view_real = next(view_str for view_str in views if view_str in real_name)
        current_view_fake = next(view_str for view_str in views if view_str in fake_name)

        assert current_view_real == current_view_fake
        current_view = current_view_fake

        real, fake = real.type(torch.float32) / 255, fake.type(torch.float32) / 255
        scores[current_view]['id'].append(metrics.calc_id(real.squeeze(0), fake.squeeze(0)))

        real, fake = real.cuda(), fake.cuda()
        scores[current_view]['ms_ssim'].append(metrics.calc_ms_ssim(real, fake))
        scores[current_view]['mse'].append(metrics.calc_mse(real, fake))
        scores[current_view]['lpips'].append(metrics.calc_lpips(real, fake))
        
        metrics.extract_inception_feats(real, fake, current_view=current_view)

    for current_view in views:
        scores[current_view]['fid'].append(metrics.calc_fid(current_view=current_view))
    
    final_scores = {}
    for current_view in views:
        final_scores[current_view] = {key: sum(value) / len(value) for key, value in scores[current_view].items()}
    print(final_scores)

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, help='Path to real image folder')
    parser.add_argument('--fake_path', type=str, help='Path to fake image folder')
    parser.add_argument('--curr_face_ckpt', type=str, help='Path to Curricular Face backbone ckpt for ID metric',
                        default='./pretrained/CurricularFace_Backbone.pth')
    parser.add_argument('--downscale_rate', type=int, help='Downscale rate of real images', default=2)
    return parser.parse_args()


if __name__ == '__main__':
    opts = get_opts()
    calc_metrics(opts)
    #calc_metrics_multiview(opts)
