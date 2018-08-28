#---------------------------------------------Hi Andy! This code is for the autoencoder rain to rain model ----------------------------------------------------------------
import torch
import cv2
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torchvision import transforms, utils
from torch.autograd import Variable

# torch.manual_seed(1)    # reproducible
def progbar(curr, total, full_progbar, is_done) :
    """
        Plot progress bar on terminal
        Args :
            curr (int) : current progress
            total (int) : total progress
            full_progbar (int) : length of progress bar
            is_done (bool) : is already done
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)

    if is_done == True :
        print('\r|'+'#'*full_progbar + '|  [{:>7.2%}]'.format(1) , end='')
    else :
        print('\r|'+'#'*filled_progbar + '-'*(full_progbar-filled_progbar) + '|  [{:>7.2%}]'.format(frac) , end='')

np.set_printoptions(threshold=np.nan)

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 16
NUM_SHOW_IMG = 4
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
STEP_NUM = 24000


train_data = torchvision.datasets.ImageFolder(
    './project_derain/training_data/',
    transform=transforms.Compose([
        #transforms.Resize((256, 256), 3),
        transforms.ToTensor()
    ])
)


print('train_data is: ',train_data)
#print('train_data[:BATCH_SIZE] is: ',train_data[:BATCH_SIZE])
#print('train_data[:BATCH_SIZE] size is: ',train_data[:BATCH_SIZE].size())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print('train_ltoader is: ',train_loader) #train_ltoader is:  <torch.utils.data.dataloader.DataLoader object at 0x0000028E0D8A3BA8>
print('train_ltoader type is: ',type(train_loader))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(         # input shape (3, 512, 512)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size
                stride=2,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 512, 512)
            nn.LeakyReLU(),                      # activation
        
            nn.Conv2d(64, 128, 3, 2, 1),     # output shape (32, 256, 256)
            nn.BatchNorm2d(num_features = 128, affine = True),
            nn.LeakyReLU(),                      # activation
    
            nn.Conv2d(128, 256, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 256, affine = True),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 512, affine = True),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1024, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 1024, affine = True),
            nn.LeakyReLU(),
            )                      
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                ),
            nn.BatchNorm2d(num_features = 512, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(num_features = 256, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(num_features = 128, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(num_features = 64, affine = True),
            nn.LeakyReLU(),
        
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.LeakyReLU(),
            )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


autoencoder = AutoEncoder().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.L1Loss().cuda()

# initialize figure
f, a = plt.subplots(2, NUM_SHOW_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

img_ground_list = None

# original data (first row) for viewing
for i,data in enumerate(train_loader,0):
    images, labels = data
    img_ground_list = images[:NUM_SHOW_IMG] #0-1 float
#    print(A.size()) #torch.Size([5, 3, 512, 512])
    break 

for i in range(NUM_SHOW_IMG):
    a[0][i].imshow(transforms.ToPILImage()(img_ground_list[i]))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(EPOCH):
    progress = 0
    for step, (b_img, b_label) in enumerate(train_loader):
        b_x = b_img.cuda()#Variable(x.view(-1, 3*512*512))   # batch x, shape (batch, 512*512)
        b_y = b_x.detach().cuda()
        
        encoded, decoded = autoencoder(b_x)
        #print(decoded.size())


        #if step % STEP_NUM == 0:
           #img_to_save = decoded.data
           #save_image(img_to_save,'res/%s-%s.jpg'%(epoch,step))
           #io.imsave('res/{}.jpg'.format(epoch),img_to_save[0])

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        #print('[{}][{}/{}]'.format(epoch, step, STEP_NUM))

        if step % STEP_NUM == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

        progbar(progress, STEP_NUM, 40, (progress == STEP_NUM-1))
        progress += 1

        if step % 60 == 0:            
            # plotting decoded image (second row)
            _, decoded_data = autoencoder(img_ground_list.cuda())

            for i in range(NUM_SHOW_IMG):
                '''
                final_decoded_data = torch.mul(decoded_data.data[i].detach(), 255.0)
                #final_decoded_data = final_decoded_data.type(torch.ByteTensor).cpu().numpy()
                #final_decoded_data = transforms.ToPILImage()(final_decoded_data)
                final_decoded_data = np.reshape(final_decoded_data.type(torch.ByteTensor).cpu().numpy(), (256, 256, 3))
                #final_decoded_datas = final_decoded_datas.type(torch.ByteTensor).cpu().numpy()
                final_decoded_data = transforms.ToPILImage(mode = 'RGB')(final_decoded_data)
                #final_decoded_data = final_decoded_datas[i]
                '''
                img_to_save = decoded_data.data[i]
                save_image(img_to_save, 'res/{}-{}-{}.jpg'.format(i, epoch, step))
                img_tmp = cv2.imread('res/{}-{}-{}.jpg'.format(i, epoch, step))
                img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
                a[1][i].clear()
                a[1][i].imshow(img_tmp)
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

            plt.draw()
            plt.pause(0.05)

# a = torch.Tensor(3,4)
# index = [1,2,0]
# a = a[index]

plt.ioff()
plt.show()
'''
# visualize in 3D plot
view_data = train_data.train_data[:200].view(-1, 512*512).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
'''