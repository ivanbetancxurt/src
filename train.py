from torch.utils import data
from nca import NCA
import torch as th

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

print(f'DEVICE: {device}')
model = NCA()
model = model.to(device)

model.fit(data_directory='../data/arc-1/training', epochs=3)

th.save({
    'model': model.state_dict(),
    'configs': {
        'n_hidden_channels': model.n_hidden_channels,
        'out_channels': 64
    },
    'epochs': 3,
    'device': str(device)
}, '../checkpoints/nca_test_02.pth')

print('Model saved.')