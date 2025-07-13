import os

from torchvision.transforms import transforms

from DataPro.dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest
from utils.flare7k_dataloader import Flare_Image_Loader


def get_training_data(flare_dir,gt_dir, img_options):
    assert os.path.exists(flare_dir)
    assert os.path.exists(gt_dir)

    transform_base = transforms.Compose([transforms.RandomCrop((512, 512), pad_if_needed=True, padding_mode='reflect'),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip()
                                         ])

    transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(0.8, 1.5),
                                                                  translate=(300 / 1440, 300 / 1440), shear=(-20, 20)),
                                          transforms.CenterCrop((512, 512)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip()
                                          ])

    dataloader = Flare_Image_Loader(gt_dir, transform_base,transform_flare,mask_type=None,options=img_options)
    dataloader.load_scattering_flare(flare_dir,os.path.join(flare_dir,'Scattering_Flare/Compound_Flare'))
    dataloader.load_reflective_flare(flare_dir,os.path.join(flare_dir,'Reflective_Flare'))
    dataloader.load_lightsource(flare_dir,os.path.join(flare_dir,'Scattering_Flare/Light_Source'))
    return dataloader
    # return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
