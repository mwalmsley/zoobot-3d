from setuptools import setup, find_packages

setup(
    name='zoobot-3d',
    version='0.1.0',
    packages=find_packages(include=['zoobot_3d', 'zoobot_3d.*']),
    install_requires=[
        'astropy',
        'pandas',
        'matplotlib',
        'pyarrow',
        'numpy',
        'opencv-python',
        'jupyter',
        'tqdm',
        'beautifulsoup4',  # only for scraping the GZ3D fits
        'wget',  # only for querying sparcfire 
        'shapely',
        # 'torch >= 2.',  # probably you want to install this already yourself, with CUDA etc
        'torchvision',
        'pytorch-lightning >= 2',
        # probably more
    ]
)

