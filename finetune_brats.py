import torch
import random

import footsteps

import icon_registration
import icon_registration.pretrained_models
import glob

model = icon_registration.pretrained_models.brain_registration_model(pretrained=True)
GPUS=4
BATCH_SIZE=4
if GPUS == 1:
    model_par = model.cuda()
else:
    model_par = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model_par.parameters(), lr=0.00005)

model_par.train()

dataset = torch.load("/playpen-ssd/tgreer/ICON_brain_preprocessed_data/BRATS_.9_mosttrain-6/BRATS_train_train.torch")

def make_batch():
    pairs = [random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)]

    for i in range(GPUS * BATCH_SIZE):
        if random.random() > .5:
            pairs[i] = (pairs[i][1], pairs[i][0])

    image_A = torch.cat([p[0] for p in pairs]).cuda()
    image_B = torch.cat([p[1] for p in pairs]).cuda()
    return image_A, image_B


icon_registration.train_batchfunction(
    model_par, opt, make_batch, steps=50000, unwrapped_net=model
)
