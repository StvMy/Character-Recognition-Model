import cv2
from torchvision import transforms
from PIL import Image
from IPython.display import clear_output
# OPEN CAM WITH PREDICTION
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((255,189)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

else:
    rval = False

while rval:
    rval, frame = vc.read()
    frame_rgb = transform(Image.fromarray(frame))   ## look for shape because the shape must be (H, W, C)

    predic = cv_model(torch.tensor(frame_rgb).unsqueeze(dim=0).to(device))
    predlabel = torch.argmax(predic, dim=1)

    print(frame.shape)
    print(f"pred: {unique[predlabel]}")
    clear_output(wait=True)

    cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")


