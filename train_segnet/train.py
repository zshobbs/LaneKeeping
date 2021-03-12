import torch
import torch.nn as nn
import glob
from effnet import Effnet
from sklearn import model_selection
from dataset import MaskImDataLoader
import config
import utils
import cv2
from tqdm import tqdm
import numpy as np


def pred_to_hooman(pred):
    pred = torch.argmax(pred, 0).type(torch.uint8)
    pred = pred.detach().cpu().numpy()
    ret = np.zeros((pred.shape[0], pred.shape[1], 3))
    for class_num, colour in enumerate([[128,255,0], [0,255,255], [255,0,127],
                                        [255,0,255], [255,0,0]]):
        ret[pred==class_num] = colour

    return ret


# Load data
mask_paths = glob.glob(f'./comma10k/cat_masks/*.png')
image_paths = [f'./comma10k/imgs/{x.split("/")[-1]}' for x in mask_paths]

# Make test, train split
(
    train_im,
    val_im,
    train_mask,
    val_mask
) = model_selection.train_test_split(
    image_paths, mask_paths, test_size=0.1, random_state=69
)


# Make training data loader
train_dataset = MaskImDataLoader(
        image_paths = train_im,
        mask_paths = train_mask,
        resize = config.RESIZE,
        mode='train'
)

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
)

# Make val data loader
val_dataset = MaskImDataLoader(
        image_paths=val_im,
        mask_paths=val_mask,
        resize=config.RESIZE,
        mode='val'
)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
)

# load model with 5 classes
model = Effnet()

optimiser = torch.optim.Adam(model.parameters(),  lr=3e-4)
model.to(config.DEVICE)

# fp16 training
if config.FP16:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.3, patience=3, verbose=True
)

best_loss = 99999
for epoch in range(config.EPOCHS):

    _, train_loss = utils.train_fn(model, train_dataloader, criterion, optimiser, scaler)
    prediction, val_loss = utils.test_fn(model, val_dataloader, criterion)
    print(f'\rEpoch {epoch} Train loss {train_loss} Val loss {val_loss}')

    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': val_loss
            }, f'./models/car_perseption.pt')
        print('Model saved')

    # Dont show if headless (server)
    if not config.HEADLESS:
        show_training_im = pred_to_hooman(prediction[0,:,:,:])
        cv2.imshow('training image', show_training_im)
        cv2.waitKey(30)

# Dont show if headless (server)
if not config.HEADLESS:
    cv2.destroyAllWindows()


