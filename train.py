from nca import NCA
import torch as th

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

print(f'DEVICE: {device}')
model = NCA()
model = model.to(device)

model.fit(data_directory='../data/arc-1/training')

th.save({
    'model': model.state_dict(),
    'configs': {
        'n_hidden_channels': model.n_hidden_channels,
        'out_channels': 34
    },
    'epochs': 800,
    'device': str(device)
}, '../checkpoints/full_arc1_run_01.pth')

print('Model saved.')