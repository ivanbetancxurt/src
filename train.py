from nca import NCA
import torch as th
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Model name')
    parser.add_argument('--bytask', action='store_true', help='Training by task')
    parser.add_argument('--nhidden', default=20, type=int, help='Number of hidden channels')
    parser.add_argument('--temp', default=5, type=int, help='Temperature for softmaxing')
    parser.add_argument('--epochs', default=800, type=int, help='Number of epochs')
    parser.add_argument('--steps', default=10, type=int, help='Number of steps allowed')
    parser.add_argument('--trials', default=128, type=int, help='Number of trials')
    parser.add_argument('--lr', default=0.002, type=float, help='Learning rate')
    parser.add_argument('--mplow', default=0.0, type=float, help='Mask probability low')
    parser.add_argument('--mphigh', default=0.75, type=float, help='Mask probability high')
    args = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    print(f'DEVICE: {device}')
    model = NCA(n_hidden_channels=args.nhidden, temperature=args.temp)
    model = model.to(device)

    if args.bytask:
        model.fit_by_task(
            task_path='../data/arc-1/training/task_1.json',
            epochs=args.epochs,
            steps=args.steps,
            trials=args.trials,
            learning_rate=args.lr,
            mask_prob_low=args.mplow,
            mask_prob_high=args.mphigh
        )
    else: 
        model.fit(
            data_directory='../data/arc-1/training',
            epochs=args.epochs,
            steps=args.steps,
            trials=args.trials,
            learning_rate=args.lr,
            mask_prob_low=args.mplow,
            mask_prob_high=args.mphigh
        )

    th.save({
        'model': model.state_dict(),
        'configs': {
            'n_hidden_channels': model.n_hidden_channels,
            'temperature': model.temperature,
            'steps': args.steps,
            'trials': args.trials,
            'learning_rate': args.lr,
            'mask_prob_low': args.mplow,
            'mask_prob_high': args.mphigh
        },
        'epochs': args.epochs,
        'device': str(device)
    }, f'../checkpoints/f{args.name}.pth')

    print('Model saved.')

if __name__ == '__main__':
    main()