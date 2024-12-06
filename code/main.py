import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--data_type', type=str, choices=['spiral', 'saddle', 'center', 'uniform'], default='spiral')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.data_type == 'spiral':
    true_y0 = torch.tensor([[0.1, 0.1]]).to(device)
    t = torch.linspace(0, 10., args.data_size).to(device)
    true_A = torch.tensor([[0.1, -1], [1, 0.2]]).to(device)
    test_true_y0 = torch.tensor([[-0.2, -0.3]]).to(device) 
    test_t = torch.linspace(0, 10., args.data_size).to(device) 

elif args.data_type == 'saddle':
    true_y0 = torch.tensor([[0.1, 0.4]]).to(device)
    t = torch.linspace(0, 3., args.data_size).to(device)
    true_A = torch.tensor([[1, 0], [0, -1.5]]).to(device)
    test_true_y0 = torch.tensor([[-0.2, -0.3]]).to(device) 
    test_t = torch.linspace(0, 3., args.data_size).to(device) 

elif args.data_type == 'center':
    true_y0 = torch.tensor([[0.1, 0.4]]).to(device)
    t = torch.linspace(0, 3., args.data_size).to(device)
    true_A = torch.tensor([[0, -1], [1, 0.]]).to(device)
    test_true_y0 = torch.tensor([[-1.3, -0.3]]).to(device) 
    test_t = torch.linspace(0, 3., args.data_size).to(device) 

elif args.data_type == 'uniform':
    true_y0 = torch.tensor([[0.1, 0.4]]).to(device)
    t = torch.linspace(0, 3., args.data_size).to(device)
    true_A = torch.tensor([[1, 0.], [0, 1.]]).to(device)
    test_true_y0 = torch.tensor([[-0.3, 0.2]]).to(device) 
    test_t = torch.linspace(0, 3., args.data_size).to(device) 



class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y, true_A.T)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    test_true_y = odeint(Lambda(), test_true_y0, test_t, method='dopri5')  # Получаем новые истинные данные



def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    ax_traj = fig.add_subplot(221, frameon=False)
    ax_vecfield = fig.add_subplot(222, frameon=False)
    ax_circle = fig.add_subplot(223, frameon=False)
    ax_loss = fig.add_subplot(224, frameon=False)
    plt.show(block=False)

def visualize(true_y, pred_y, test_true_y, test_pred_y, odefunc, itr, loss_list, test_loss_list, max_itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title(f'Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-', label='True x', color='darkblue') 
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', label='Pred x', color='blue')
        ax_traj.plot(t.cpu().numpy(), test_true_y.cpu().numpy()[:, 0, 0], '-', label='Test True x', color='darkorange')  
        ax_traj.plot(t.cpu().numpy(), test_pred_y.cpu().numpy()[:, 0, 0], '--', label='Test Pred x', color='orange') 
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()


        ax_vecfield.cla()
        ax_vecfield.set_title('Vector Field and Trajectories')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')


        ax_vecfield.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], '-', color='darkblue', linewidth=2, label='True')
        ax_vecfield.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], '--', color='blue', linewidth=2, label='Pred')


        ax_vecfield.plot(test_true_y.cpu().numpy()[:, 0, 0], test_true_y.cpu().numpy()[:, 0, 1], '-', color='darkorange', linewidth=2, label='Test True')
        ax_vecfield.plot(test_pred_y.cpu().numpy()[:, 0, 0], test_pred_y.cpu().numpy()[:, 0, 1], '--', color='orange', linewidth=2, label='Test Pred')


        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        
        # Правильное поле направлений
        true_dydt = torch.mm(torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device), true_A.T).cpu().detach().numpy()
        true_mag = np.sqrt(true_dydt[:, 0]**2 + true_dydt[:, 1]**2).reshape(-1, 1)
        true_dydt = (true_dydt / true_mag)
        true_dydt = true_dydt.reshape(21, 21, 2)

        # Обученное поле направлений
        learned_dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        learned_mag = np.sqrt(learned_dydt[:, 0]**2 + learned_dydt[:, 1]**2).reshape(-1, 1)
        learned_dydt = (learned_dydt / learned_mag)
        learned_dydt = learned_dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, true_dydt[:, :, 0], true_dydt[:, :, 1], color="gray", density=[0.5, 1], linewidth=1)
        ax_vecfield.streamplot(x, y, learned_dydt[:, :, 0], learned_dydt[:, :, 1], color="black", density=[0.5, 1], linewidth=1)
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)
        ax_vecfield.legend(loc='upper right')

        # Новый график для трансформации единичного шара
        ax_circle.cla()
        ax_circle.set_title('Effect on Unit Circle')
        theta = np.linspace(0, 2 * np.pi, 100)
        unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        # Трансформация матрицей true_A
        transformed_by_A = torch.mm(torch.Tensor(unit_circle).to(device), true_A.T).cpu().numpy()

        # Трансформация функцией odefunc
        transformed_by_odefunc = odefunc(0, torch.Tensor(unit_circle).to(device)).cpu().detach().numpy()

        ax_circle.plot(unit_circle[:, 0], unit_circle[:, 1], '-', label='Unit Circle', color='black')
        ax_circle.plot(transformed_by_A[:, 0], transformed_by_A[:, 1], '--', label='true_A', color='blue')
        ax_circle.plot(transformed_by_odefunc[:, 0], transformed_by_odefunc[:, 1], '--', label='ODEFunc', color='orange')

        ax_circle.set_xlim(-2, 2)
        ax_circle.set_ylim(-2, 2)
        ax_circle.set_xlabel('x')
        ax_circle.set_ylabel('y')
        ax_circle.legend(loc='upper right')


        ax_loss.cla()
        ax_loss.set_title('Loss')
        ax_loss.semilogy(loss_list, label='Train', color='blue')
        ax_loss.semilogy(test_loss_list, label='Test', color='orange')
        ax_loss.set_xlim(0, max_itr)
        ax_loss.set_ylim(0, 1.1 * max(test_loss_list))
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()


        fig.tight_layout()
        plt.savefig('{}-gif/frame-{}'.format(args.data_type, itr))
        plt.draw()
        plt.pause(0.001)




class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    test_loss_meter = RunningAverageMeter(0.97)

    loss_list = []
    test_loss_list = []
    # Добавляем новые начальные данные

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        loss_list.append(loss_meter.avg)
        with torch.no_grad():
            test_pred_y = odeint(func, test_true_y0, test_t)
            test_loss = torch.mean(torch.abs(test_pred_y - test_true_y))
            test_loss_meter.update(test_loss.item())
            test_loss_list.append(test_loss_meter.avg)

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                
                # Считаем loss для новых данных
                test_pred_y = odeint(func, test_true_y0, test_t)
                test_loss = torch.mean(torch.abs(test_pred_y - test_true_y))
                print('Iter {:04d} | Test Data Loss {:.6f}'.format(itr, test_loss.item()))
                
                visualize(true_y, pred_y, test_true_y, test_pred_y, func, ii, loss_list, test_loss_list, args.niters)
                ii += 1

        end = time.time()