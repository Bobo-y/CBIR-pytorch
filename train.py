import torch
import torch.optim as optim
import torch.utils.data as Data
import time
import datetime
from torchvision import datasets, transforms
from network import Net


BATCH_SIZE = 32

train_dataset = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

gpu = torch.cuda.is_available()
model = Net()
num_epochs = 20
epoch_size = train_dataset.__len__() // BATCH_SIZE
max_iter = epoch_size * num_epochs
loss = torch.nn.MSELoss()
if gpu:
    model = model.cuda()
    loss = loss.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


iteration = 0
for epoch in range(num_epochs):
    model.train()
    for i, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        load_t0 = time.time()
        if gpu:
            img = img.cuda()
        out = model(img)
        loss_v = loss(out, img)
        loss_v.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f} || Batchtime: {:.4f} s ||ETA: {}'.format(
            epoch, num_epochs, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss_v, batch_time,
            str(datetime.timedelta(seconds=eta))))
        iteration += 1
    torch.save(model.state_dict(), "./data/model.pth")
