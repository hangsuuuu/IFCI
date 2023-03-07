import numpy as np
import pickle
import torch
import argparse
import os
from data import envgen, training_eval, testing
from casual_infer import casual_inference
import visdom

mse = torch.nn.MSELoss(reduction="none")
torch.set_default_dtype(torch.float64)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--N', type=int, default=1000, help='data in one component')
    parser.add_argument('--K', type=int, default=5, help='number of component')
    parser.add_argument('--sigma', type=float, default=1., help='std of GMM')
    parser.add_argument('--lmbd', type=float, default=30.0, help='weight for coco term')
    parser.add_argument('--lmbd_irm', type=float, default=100.0, help='weight for irm')
    parser.add_argument('--lmbd_rex', type=float, default=10000.0, help='weight for rex')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_env', type=int, default=5)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--spurious', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='ERM', help='ERM, IRM, Rex, CoCo')
    args = parser.parse_args()

    path = os.path.join(args.path, f'gmm_{args.method}_{args.lmbd_rex}')
    result = []
    if not os.path.exists(path):
        os.makedirs(path)
    # 生成数据，因果推断
    envs = envgen(args)
    casual_var = casual_inference(args, envs)
    args.casual = casual_var
    new_envs = []
    for i in range(len(envs)):
        new_data = envs[i][0][:, casual_var]
        new_envs.append([new_data, envs[i][1]])

    dim_in = len(casual_var)
    dim_out = args.K
    dims = [dim_in, 10, 10, dim_out]
    net = torch.nn.Sequential(
        torch.nn.Linear(dims[0], dims[1]),
        torch.nn.Sigmoid(),
        torch.nn.Linear(dims[1], dims[2]),
        torch.nn.Sigmoid(),
        torch.nn.Linear(dims[2], dims[3])
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


    test_r = []
    train_r = []
    iter_r = []
    for epoch in range(args.steps):
        if args.method == 'ERM':
            risk = 0
            for [inputs, labels] in new_envs:
                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                outputs = net(inputs)
                risk_e = criterion(outputs, labels.long())
                risk += risk_e
            risk = risk / len(envs)

            optimizer.zero_grad()
            if args.method == 'ERM':
                tot_loss = risk
            tot_loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print('epoch', epoch, '########################', flush=True)
            test_perform = testing(net, args)
            train_perform = training_eval(net, new_envs)
            test_r.append([np.mean(test_perform), np.std(test_perform)])
            train_r.append([np.mean(train_perform), np.std(train_perform)])
            iter_r.append(epoch)
            result.append(np.mean(test_perform))
    result = np.array(result)
    np.savetxt('test_acc.txt', result)
    test_r = np.array(test_r)[:, 0]
    train_r = np.array(train_r)[:, 0]

    pickle.dump([iter_r, train_r, test_r], open(os.path.join(path, 'gmm_' + str(args.seed) + '.pkl'), 'wb'))