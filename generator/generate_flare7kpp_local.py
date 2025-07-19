
from matplotlib import pyplot as plt
import os

from generator import Flare_Image_Loader

transform_base= {
	'img_size': 512
}


transform_flare= {
    "scale_min": 0.8,
    "scale_max": 1.5,
    "translate": 300,
    "shear": 20
}

length = 64

if __name__ == '__main__':
    loader = Flare_Image_Loader('E:/dataset/Flickr24K', transform_base, transform_flare,length=length)
    loader.load_scattering_flare('E:/dataset/Flare7Kpp/Flare7K','E:/dataset/Flare7Kpp/Flare-R/Compound_Flare')
    loader.load_light_source('E:/dataset/Flare7Kpp/Flare7k','E:/dataset/Flare7Kpp/Flare-R/Light_Source')
    # loader.load_reflective_flare('E:/dataset/Flare7Kpp/Flare7K','E:/dataset/Flare7Kpp/Flare7k/Reflective_Flare')
    channels = [0,1,2,3]
    for c in channels:
        try:
            os.makedirs(f'../dataset/flare7k_local/input/c{c}')
            os.makedirs(f'../dataset/flare7k_local/gt/c{c}')
        except FileExistsError:
            print(f'Folder ../dataset/flare7k_local/input/c{c} already exists.')
            print(f'Folder ../dataset/flare7k_local/gt/c{c} already exists.')
        for i in range(length):
            dic = loader[i]
            gt, input = dic['gt'], dic['lq']

            plt.imsave(f'../dataset/flare7k_local/input/c{c}/{i}.png', input.numpy().transpose(1, 2, 0))
            plt.imsave(f'../dataset/flare7k_local/gt/c{c}/{i}.png', gt.numpy().transpose(1, 2, 0))

            if i%100==0:
                print(f'Processed {i} images for channel {c}.')

