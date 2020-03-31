from network import Net
import torch
import pickle
import torch.utils.data as Data
from torchvision import datasets, transforms


train_dataset = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
model = Net()
model.load_state_dict(torch.load("./data/model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

features = []
for i, (img, lab) in enumerate(train_loader):
    latent = model.encoder(img)
    if i == 0:
        features = latent
    else:
        features = torch.cat((features, latent), 0)

indexes = list(range(0, train_dataset.train_data.shape[0]))
data = {"indexes": indexes, "features": features}
f = open("./data/features_dict.pickle", "wb")
f.write(pickle.dumps(data))
f.close()