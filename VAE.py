import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 200
batch_size = 128
learning_rate = 1e-3


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(True),
            nn.Linear(400, 40))
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(True),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid())

    def reparametrize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)

    def forward(self, x):
        h = self.encoder(x)
        mu = h[:, :20]
        logvar = h[:, 20:]
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encoder(x_one)
        hidden_two = self.encoder(x_two)
        mu_one = hidden_one[:, :20]
        logvar_one = hidden_one[:, 20:]
        mu_two = hidden_two[:, :20]
        logvar_two = hidden_two[:, 20:]
        mu = (1 - alpha) * mu_one + alpha * mu_two
        logvar = (1 - alpha) * logvar_one + alpha * logvar_two
        z = self.reparametrize(mu, logvar)
        generated_image = self.decoder(z)
        return generated_image

model = VariationalAutoencoder().cuda()
BCE = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        x_hat, mu, logvar = model(img)
        NKLD = mu.pow(2).add(logvar.exp()).mul(-1).add(logvar.add(1))
        KLD = torch.sum(NKLD).mul(-0.5)
        KLD /= 128 * 784
        loss = BCE(x_hat, img) + KLD
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(x_hat.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        batch = iter(dataloader).next()[0]
        batch = batch.view(batch.size(0), -1)
        batch = Variable(batch).cuda()
        x_one = batch[0:1]
        x_two = batch[1:2]
        generated_images = []
        for alpha in torch.arange(0.0, 1.0, 0.1):
            generated_images.append(model.generation_with_interpolation(
                x_one, x_two, alpha))
        generated_images = torch.cat(generated_images, 0).cpu().data
        save_image(generated_images.view(-1, 1, 28, 28),
                   './generated/output_interpolate_{}.png'.format(epoch),
                   nrow=1)
torch.save(model.state_dict(), './sim_variational_autoencoder.pth')
