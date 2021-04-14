import torch
import json
import numpy as np

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

class TC_images(torch.utils.data.Dataset):
    def __init__(self, base_dir, image_dir, label_dir, truncate=None, image_dim=256):
        """
        Tropical Cyclone Image Dataset
        Args:
            base_dir  (string) : Base directory of files
            image_dir (string) : Directory with all the images files.
            label_dir (string) : Directory with all the label json files.
            truncate  (int)    : Truncate dataset if necessary
            image_dim (int)    : Dimension of image (256 is default as used in Maskey et al 2020)
        """
        with open(base_dir / image_dir / 'collection.json', 'r') as f:
            coll = json.load(f)['links']

        df = pd.DataFrame(coll)
        self.tags = df["href"].str.strip('/stac.json').apply(lambda row: row[-7:]).to_list()

        if isinstance(truncate, int):
            self.tags = self.tags[:truncate]
            
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_dim = image_dim

        
    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        tag = self.tags[idx]
        img_sd = self.image_dir + '_' + tag
        lab_sd = self.label_dir + '_' + tag
        image_path = self.base_dir / self.image_dir / img_sd / 'image.jpg'
        label_path = self.base_dir / self.label_dir / lab_sd / 'labels.json'

        image = io.imread(image_path)
        target = self.get_label(label_path)

        # check for image channel and shape
        if len(image.shape) > 2:
            image = rgb2gray(image)
        if image.shape[1] != self.image_dim:
            image = resize(image, (self.image_dim, self.image_dim))
            
        image = torch.tensor(image)
        image = image.unsqueeze(0) # add a channel index

        sample = {'image': image, 'wind_speed': target, 'tag': tag}

        return sample

    def get_label(self, name):
        with open(name, 'r') as f:
            target = float(json.load(f)['wind_speed'])
        return torch.tensor(target).unsqueeze(-1)
        

# dataset class
class STORM_data(torch.utils.data.Dataset):
    def __init__(self, base_dir, data_txt_file, normalize=False, truncate=None):
        """
        Args:
            base_dir  (string)     : Base directory of files
            data_txt_file (string) : .txt file with all the data.
            truncate  (int)        : truncate dataset if necessary
        """
        # full_path = base_dir / data_txt_file
        full_path = base_dir + data_txt_file
        self.data = pd.read_csv(
            full_path, header=0, names=['Year','Month','TC number','Time step',
                'Basin ID','Latitude','Longitude','Minimum pressure','Maximum wind speed',
                'Radius to maximum winds','Category','Landfall','Distance to land']
            )
        self.norm = normalize

        # calculate log coriolis factor
        self.data['log_f'] = np.log( 2.0*7.2921e-5 * np.sin(self.data['Latitude'].values*np.pi/180.0 ) )

        if normalize:
            self.maxs_mins = {
                'min_press' : [self.data['Minimum pressure'].max(), self.data['Minimum pressure'].min()],
                'wind_speed' : [self.data['Maximum wind speed'].max(), self.data['Maximum wind speed'].min()],
                'r' : [self.data['Radius to maximum winds'].max(), self.data['Radius to maximum winds'].min()]
            }
        if isinstance(truncate, int):
            self.data = self.data[:truncate]

    def normalize(self, label, value):
        if label not in self.maxs_mins.keys():
            return value
        max, min = self.maxs_mins[label]
        return (value - min) / (max - min)

    def unnormalize(self, label, value):
        if label not in self.maxs_mins.keys():
            return value
        max, min = self.maxs_mins[label]
        return value * (max - min) + min

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, normalize = True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data.iloc[idx]

        sample = {
            'wind_speed': torch.tensor(datum['Maximum wind speed']).unsqueeze(0).float(), 
            'r': torch.tensor(datum['Radius to maximum winds']).unsqueeze(0).float(),
            'f': torch.tensor(np.exp(datum['log_f'])).unsqueeze(0).float(),
            'min_press': torch.tensor(datum['Minimum pressure']).unsqueeze(0).float()
        }

        if self.norm:
            normalized_sample = {}
            for label, value in sample.items():
                normalized_sample[label] = self.normalize(label, value)
            return normalized_sample
        else:
            return sample
        