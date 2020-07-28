import torch
from torchvision import transforms
import os
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from PIL import Image
import cPickle as pickle
import torch.utils.data as data

class BirdDataset(data.Dataset):
    def __init__(self, dataDir, split='train', imgSize=64, transform=None):
        super(BirdDataset,self).__init__()
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imgSize = imgSize
        self.dataDir = dataDir
        self.filenames, self.caps = self.load_info(dataDir, split)
        self.bbox = self.load_bbox()
        self.classes = self.load_class(dataDir, split)
              
    def load_info(self, dataDir, split):
        filenames = self.load_filenames(dataDir, split)
        captionFile = os.path.join(dataDir, 'birds', split, 'char-CNN-RNN-embeddings.pickle')
        with open (captionFile, 'rb') as f:
            captions = pickle.load(f)
            captions = np.array(captions)
            print('captions shape: ', captions.shape)
        
        return filenames, captions
        
    def load_filenames(self, dataDir, split):
        path = os.path.join(dataDir, 'birds', split, 'filenames.pickle')
        with open(path, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (path, len(filenames)))
        return filenames
    
    def load_bbox(self):
        path = os.path.join(self.dataDir, 'CUB_200_2011', 'bounding_boxes.txt')
        bbox_data = pd.read_csv(path, delim_whitespace=True, header=None).astype(int)

        filepath = os.path.join(self.dataDir, 'CUB_200_2011','images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = sorted( list(df_filenames[1]))
        fname_bbox_dict = {x[:-4]:[] for x in filenames} # use filename without '.jpg' extension as a key
        for i in range(len(filenames)):
            data = list(bbox_data.iloc[1][1:])
            k = filenames[i][:-4]
            fname_bbox_dict[k] = data
        return fname_bbox_dict
    
    def load_class(self, dataDir, split):
        path = os.path.join(dataDir, 'birds', split, 'class_info.pickle')
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                classId = pickle.load(f)
        else:
            classId = np.arange(len(self.filenames))
        return classId

    def get_img(self, img_path, bbox=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imgSize * 76 / 64)
        img = img.resize((load_size, load_size), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    
    def __getitem__(self, idx):
        key = self.filenames[idx]
        
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        emb = self.caps[idx, :, :]
        imagePath = os.path.join(self.dataDir, 'CUB_200_2011', 'images',self.filenames[idx]+'.jpg')
        image = self.get_img(imagePath, bbox)
        
        # random select a sentence
        sample = np.random.randint(0, emb.shape[0]-1)
        cap = emb[sample, :]
        return image, cap
    
    def __len__(self):
        return len(self.filenames)