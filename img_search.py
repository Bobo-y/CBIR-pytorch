import numpy as np
import pickle
import cv2
import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
from imutils import build_montages
from network import Net
torch.set_grad_enabled(False)


def euclidean(a, b):
    return np.linalg.norm(a - b)


def search(query_feature, features_dict, top_k=100):
    results = []
    for i in range(0, len(features_dict["features"])):
        d = euclidean(query_feature, features_dict["features"][i])
        results.append((d, i))
    results = sorted(results)[:top_k]
    return results


train_dataset = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
feature_dict = pickle.loads(open("./data/features_dict.pickle", "rb").read())
model = Net()
model.eval()
model.load_state_dict(torch.load("./data/model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
encoder = model.encoder


for i, (img, lab) in enumerate(test_loader):
    query_feature = encoder(img)
    res = search(query_feature, feature_dict)
    images = []
    for (d, j) in res:
        image = train_dataset.train_data[j, ...]
        image = np.dstack([image] * 3)
        images.append(image)
    query = img.squeeze(0).permute(1, 2, 0)
    query = np.dstack([query] * 3) * 255
    cv2.imwrite("./result/{}_query.jpg".format(i), query)
    montage = build_montages(images, (28, 28), (10, 10))[0]
    cv2.imwrite("./result/{}_query_results.jpg".format(i), montage)