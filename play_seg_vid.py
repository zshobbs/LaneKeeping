import albumentations
import torch
import numpy as np
import cv2

from effnet import Effnetb7

model_path = './models/models/b7_512_352.pth'
vid_path = './driving_data/videos/5.mp4'
vid_out = './driving_data/videos/out_vid.mp4'

DEVICE = 'cuda'
# Load model set in eval mode
model = Effnetb7()
cp = torch.load(model_path)
model.load_state_dict(cp['state_dict'])

model.to(DEVICE)
model.eval()

# Augmentation (normilisation) init
aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

# cv2 captuer vid
cap = cv2.VideoCapture(vid_path)

#set up vid writer
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
writer = cv2.VideoWriter(
        vid_out,
        fourcc,
        20.0,
        (512,352)
)

def pred_to_hooman(pred):
    pred = torch.squeeze(pred, 0)
    pred = torch.argmax(pred, 0).type(torch.uint8)
    pred = pred.detach().cpu().numpy()
    ret = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_num, colour in enumerate([[128,255,0], [0,255,255], [255,0,127],
                                        [255,0,255], [0,0,255]]):
        ret[pred==class_num] = colour

    return ret

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (512,352))
        show_frame = np.copy(frame)
        augmented = aug(image=frame)
        frame = augmented["image"]
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame, dtype=torch.float)
        frame = torch.unsqueeze(frame, 0)
        frame = frame.to(DEVICE)
        
        prediction = model(frame)
        prediction = pred_to_hooman(prediction)
        cv2.addWeighted(prediction, 0.4, show_frame, 1-0.4, 0, show_frame)
        writer.write(show_frame)
        cv2.imshow('Overlay', show_frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    
cv2.destroyAllWindows()
cap.release()
writer.release()
