
import glob
import os
import argparse
import footsteps
import splits
footsteps.initialize(output_root="/playpen-ssd/tgreer/ICON_brain_preprocessed_data/")


import torch.nn.functional as F
def process(iA, isSeg=False):
    iA = iA[None, None, :, :, :]
    iA = iA.float()
    iA = iA / (.9 * torch.max(iA) )
    iA = F.interpolate(iA, image_shape[2:], mode="trilinear")
    return iA

for split in ["train"]:
    
    image_paths = getattr(splits, split)
    import torch

    import itk
    import tqdm
    import numpy as np
    import glob

    ds = []

    for name1, name2 in tqdm.tqdm(image_paths):
        

        imageA = torch.tensor(np.asarray(itk.imread(name1)))
        imageB = torch.tensor(np.asarray(itk.imread(name2)))

        ds.append((process(imageA), process(imageB)))

    torch.save(ds, f"{footsteps.output_dir}/BRATS_train_{split}.torch")

