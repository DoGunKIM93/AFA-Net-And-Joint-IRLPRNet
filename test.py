import torchvision.models
from torchvision.models.vgg import model_urls

model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
vgg16 = torchvision.models.vgg16(pretrained=True)