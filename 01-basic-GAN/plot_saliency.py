import torch
from torchvision import transforms
import torchvision.models as models
from flashtorch.activmax import GradientAscent
import matplotlib.pyplot as plt
import numpy as np

import net_architectures 

DATADIR = '/lustre/projects/metoffice/ml-tc/output/01-basic-GAN/20220124T1701'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained models
dmodel = torch.load(DATADIR+'/netD.pt', map_location=device).module
dmodel.eval()

gmodel = torch.load(DATADIR+'/netG.pt', map_location=device).module
gmodel.eval()

img_list = np.load(DATADIR+'/produced.npy')

# List model features
list(gmodel.main.named_children())

# Look at Saliency Maps and Activation maximization plots
# based on https://github.com/MisaOgura/flashtorch

# Pick a nice image (TC simulation)
one_img = img_list[-1,19,0,:,:]  # This particular index looks nice

transform_ = transforms.Compose([transforms.ToTensor()])
image = transform_(one_img).unsqueeze(0)  
image = image.to(device)
image.requires_grad_()

output = dmodel(image)  # Retrieve output from the image
output_idx = output.argmax()
output_max = output[0, output_idx]
output_max.backward()  # Do backpropagation to get the derivative of the output based on the image

# Retrieve saliency map 
saliency, _ = torch.max(image.grad.data.abs(), dim=1)
saliency = saliency.reshape(256, 256)

# Plot results
fig, ax = plt.subplots(1, 3)
ax[0].imshow(one_img)
ax[1].imshow(saliency.cpu(), cmap='hot')
ax[2].imshow(one_img, alpha=saliency.cpu())
plt.tight_layout()
plt.savefig('saliency.png')



# Not working --------------------------------
# Extract layers and filters
# gconv1_5 = gmodel.main[5]

# d_ascent = GradientAscent(dmodel.main)
# d_ascent.use_gpu = False

# g_ascent = GradientAscent(gmodel.main)
# g_ascent.use_gpu = False

# d_ascent.visualize(conv1_5)

# g_gascent.visualize(conv1_5)