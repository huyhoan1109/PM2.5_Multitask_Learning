from process import *
from models import model1, model2
import os

def get_name(net):
    vnames = [name for name in globals() if globals()[name] is net]
    return vnames[0]

def train(ae_net, multi_net, loader_ae, loader_multi, save=True):
    if not os.path.exists('./savemodel/latent.pth'):
    # Train AutoEncoder
        optim_1 = optim.Adam(ae_net.parameters(), lr=lr,weight_decay=w_decay)
        criter_1 = nn.MSELoss()
        run_1_loss = 0.0
        ae_net.train()
        ae_net.to(device)
        print('Train AE: ...')
        for epoch in range(epochs//2):
            loop_1 = tqdm(enumerate(loader_ae), total=len(loader_ae), leave=False)
            for _, meteo_data in loop_1:
                input_meo, _ = meteo_data
                input_meo = input_meo.to(device)
                optim_1.zero_grad()
                target = ae_net(input_meo)
                loss_ae = criter_1(target, input_meo)
                loss_1 = loss_ae.item()
                run_1_loss += loss_1
                loss_ae.backward() 
                optim_1.step()
                loop_1.set_description(f'Epoch [{epoch+1}/{epochs//2}]')
                loop_1.set_postfix(loss=loss_1)
        print('Done')
        latent = Variable(ae_net.latent).to(device)
        if save == True:
            torch.save(latent, './savemodel/latent.pth')
            torch.save(ae_net.state_dict(), f'./savemodel/{get_name(ae_net)}.pth')
    else:
        latent = torch.load('./savemodel/latent.pth')
    # Train Multitasks
    optim_2 = optim.Adam(multi_net.parameters(), lr=lr,weight_decay=w_decay)
    criter_2 = nn.MSELoss()
    scheduler_2 = optim.lr_scheduler.StepLR(optim_2, step_size=10, gamma=0.1, last_epoch=-1, verbose=False)
    run_2_loss = 0.0
    multi_net.train()
    multi_net.to(device)
    print('Train Multitask Net: ...')
    for epoch in range(epochs):
        loop_2 = tqdm(enumerate(loader_multi), total=len(loader_multi), leave=False)
        for _, pm2_5_data in loop_2:
            input_pm2_5, output_pm2_5 = pm2_5_data
            input_pm2_5 = input_pm2_5.to(device)
            output_pm2_5 = output_pm2_5.to(device)
            optim_2.zero_grad()
            res = multi_net(input_pm2_5,latent)
            loss_net = criter_2(res, output_pm2_5)
            loss_2 = loss_net.item()
            run_2_loss += loss_2
            loss_net.backward() 
            optim_2.step()
            loop_2.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop_2.set_postfix(loss=loss_2)
        scheduler_2.step()
    print('Done')
    
    if save == True:
        torch.save(multi_net.state_dict(), f'./savemodel/{get_name(multi_net)}.pth')

def showParam(*nets):
    params = []
    for net in nets:
        param = sum(p.numel() for p in net.parameters())
        params.append(param)
    print(f"Total params {sum(params)} with:")
    for i in range(len(nets)):
        name = get_name(nets[i])
        print(
            f'{name} '
            f'has {params[i]} params'
        )
def draw(model, loader, latent=None, times=1000):
    in_lists = []
    out_lists = []
    model.eval()
    name = get_name(model)
    stage = 0
    if latent != None and model.__class__.__name__ == 'Multitask_Net':
        stage = 0
    elif model.__class__.__name__ == 'AutoEncoder':
        stage = 1
    else:
        ValueError("Can't process timeseries!")

    for _, data_idx in enumerate(loader, 0):
        input_idx, target_idx = data_idx
        input_idx = input_idx.to(device)
        if stage == 0:
            ip = [input_idx, latent]
        else:
            ip = [input_idx]
        output_idx = model(input_idx)
        for i in range(64):
            in_lists.append(float(target_idx[i][0][0][0]))
            out_lists.append(float(output_idx[i][0][0][0]))
        
    if times > len(in_lists):
        times = len(in_lists)
    plt.figure(figsize=(25,10))
    plt.plot(np.arange(times), in_lists[0:times], 'b', linewidth = 1.5, label = "standard")
    plt.plot(np.arange(times), out_lists[0:times], 'r', linewidth = 1.5, label = "predict")
    plt.savefig(f'./results/{name}.png')

if __name__ == '__main__':
    
    data = Process_Data('./data/', True)
    meteo_and_pm2_5 = data.split_and_to_tensor()
    
    meteo_train_samples = create_seq_samples(meteo_and_pm2_5[0], in_seq_len, out_seq_len)
    meteo_test_samples = create_seq_samples(meteo_and_pm2_5[1], in_seq_len, out_seq_len)
    pm2_5_train_samples = create_seq_samples(meteo_and_pm2_5[2], in_seq_len, out_seq_len)
    pm2_5_test_samples = create_seq_samples(meteo_and_pm2_5[3], in_seq_len, out_seq_len)

    meteo_train_dataset = AQIdatasets(meteo_train_samples[0], meteo_train_samples[1])
    meteo_test_dataset = AQIdatasets(meteo_test_samples[0], meteo_test_samples[1])
    pm2_5_train_dataset = AQIdatasets(pm2_5_train_samples[0], pm2_5_train_samples[1])
    pm2_5_test_dataset = AQIdatasets(pm2_5_test_samples[0], pm2_5_test_samples[1])

    meteo_train_loader = DataLoader(meteo_train_dataset, batch_size=batch_size, shuffle=True)
    meteo_test_loader = DataLoader(meteo_test_dataset, batch_size=batch_size, shuffle=True)
    pm2_5_train_loader = DataLoader(pm2_5_train_dataset, batch_size=batch_size, shuffle=True)
    pm2_5_test_loader = DataLoader(pm2_5_test_dataset, batch_size=batch_size, shuffle=True)
    
    AE = model1.AutoEncoder(n_tasks * meteo_size * in_seq_len)
    MultiNet1 = model1.Multitask_Net(n_tasks, in_seq_len, out_seq_len)
    showParam(AE, MultiNet1)
    train(AE, MultiNet1, meteo_train_loader, pm2_5_train_loader)
    MultiNet2 = model2.Multitask_Net(n_tasks, in_seq_len, out_seq_len)
    showParam(MultiNet2)
    train(AE, MultiNet2, meteo_train_loader, pm2_5_train_loader)