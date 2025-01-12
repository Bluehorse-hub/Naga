import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import naga
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

time_path, best_judge_list = naga.preparation()
params_value_conbinations, params_name_list = naga.loadyaml('./config/params.yaml')
for i, (lr, batch_size) in enumerate(params_value_conbinations):
    print(i)
    
    model = SimpleNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 5
    loss_list, dirs_path = naga.init()
    dirs_path = naga.update_dirs_path(i, time_path, dirs_path)
    print(dirs_path)
    params_value_list = [lr, batch_size]
    naga.makedirs(dirs_path)
    naga.makeyaml(dirs_path, params_name_list, params_value_list)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        loss_list.append(running_loss/len(train_loader))
        
    best_judge_list.append(sum(loss_list))

    naga.plot_loss_history(loss_list, dirs_path)
    torch.save(model.state_dict(),dirs_path + '/weight.pth')
    print("モデルが保存されました。")
print(len(best_judge_list))
naga.best_study(time_path, best_judge_list)
