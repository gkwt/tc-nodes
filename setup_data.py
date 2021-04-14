import wget
import os
import tarfile
import glob
from zipfile import ZipFile
from pathlib import Path

from radiant_mlhub import Dataset, Collection, client
os.environ['MLHUB_API_KEY'] = '264c5b1ca1d51c92b849746bf8828eef64208cb0a89c35371e8b0e038257cc7a'

# make directories
if not os.path.isdir('data/'):
    os.mkdir('data/')
if not os.path.isdir('data/STORM'):
    os.mkdir('data/STORM')
if not os.path.isdir('data/NASA'):
    os.mkdir('data/NASA')
    
download_dir = Path('data').expanduser().resolve()

# get STORM dataset
download_dir_storm = download_dir / "STORM"
if len(list(download_dir_storm.glob('*'))) == 0:
    url = "https://data.4tu.nl/ndownloader/articles/12706085/versions/2"
    filename = wget.download(url)
    os.rename(filename, f'data/STORM/STORM.zip')

with ZipFile('data/STORM/STORM.zip','r') as z:
    z.extractall('data/STORM/')
with ZipFile('data/STORM/STORM_data.zip', 'r') as z:    
    z.extractall('data/STORM/')

os.remove('data/STORM/STORM.zip')
os.remove('data/STORM/STORM_data.zip')
    
print('Done STORM!')
    
# get 
download_dir_nasa = download_dir / "NASA"

if len(list(download_dir_nasa.glob("*"))) < 8:
    dataset = Dataset.fetch('nasa_tropical_storm_competition')
    archive_paths = dataset.download(output_dir=download_dir_nasa)

    for archive_path in archive_paths:  
        print(f'Extracting {archive_path}')
        with tarfile.open(archive_path) as tfile:
            tfile.extractall(path=download_dir_nasa)
        
print('Done NASA TC!')
