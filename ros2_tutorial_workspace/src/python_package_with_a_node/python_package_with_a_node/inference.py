import torch
from model import DAVE2v3
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize
from PIL import Image as IM

TRANSFORM = Compose([
    ToPILImage(),
    # PILToTensor(),
    # transforms.Resize((100,100)),
    # transforms.Grayscale(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
])

TRANSFORM2 = Compose([

    ToTensor(),
    Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
    # PILToTensor(),
    # transforms.Resize((100,100)),
    # transforms.Grayscale(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
])
# load model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

# inference
def inference(model, image):
    # image is a RGB np array
    # transform image
    image = TRANSFORM2(image)
    # add batch dimension
    #image = image[:,:,:image.shape[-1]//2]
    image = TRANSFORM(image)
    image = image.crop((0,0, image.size[0]//2, image.size[1]*0.85))
    image = image.resize((672//2,188), IM.ANTIALIAS)
    image = TRANSFORM2(image)
    image = image.unsqueeze(0)
    print(image.shape)
    # get prediction
    prediction = model(image).item()
    return prediction

