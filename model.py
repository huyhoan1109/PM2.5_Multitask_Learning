from modules import *

class AutoEncoder(nn.Module):
    def __init__(self, input_ae):
        super(AutoEncoder, self).__init__()
        self.input_ae = input_ae
        self.encoder = nn.Sequential(
            nn.Linear(input_ae, 128, True, device),
            nn.Sigmoid(),
            nn.Linear(128, 64, True, device),
            nn.Sigmoid(),
            nn.Linear(64, 32, True, device),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64, True, device),
            nn.ReLU(),
            nn.Linear(64, 128, True, device),
            nn.ReLU(),
            nn.Linear(128, input_ae, True, device),
            nn.ReLU()
        )
        self.latent = None
    def forward(self, x):
        origin_size = x.shape
        x = x.view((origin_size[0], -1))
        x = self.encoder(x)
        self.latent = x.clone()
        x = self.decoder(x)
        x = x.view(origin_size)
        return x

class Multitask_Net(nn.Module):
    def __init__(self, n_tasks, in_seq_len, out_seq_len):
        super(Multitask_Net, self).__init__()
        self.n_tasks = n_tasks
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.lstm0 = nn.LSTM(
            input_size = n_tasks,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )
        self.lstm1 = nn.LSTM(
            input_size = 64,
            hidden_size = 32,
            num_layers = 1,
            batch_first = True
        )
        self.mix_block = nn.Sequential(
            nn.Linear(64, 64, True, device),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 128, True, device),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64, True, device),
            nn.LeakyReLU(0.01)
        )
        self.predict_block = nn.ModuleList()
        for _ in range(n_tasks):
            self.predict_block.append(
                nn.Sequential(
                    nn.Linear(64, 16, True, device),
                    nn.LeakyReLU(0.01),
                    nn.Linear(16, out_seq_len, True, device)
                )
            )
    def forward(self, x, latent):
        # x(batch_size, N, thong so, len)
        origin_size = x.shape
        # x(batch_size, N x thong so, len)
        x = x.view((origin_size[0], -1, origin_size[3]))
        # x(batch_size, len, N x thong so)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(1, origin_size[0], 64, device=device)
        c0 = torch.zeros(1, origin_size[0], 64, device=device)
        h1 = torch.zeros(1, origin_size[0], 32, device=device)
        c1 = torch.zeros(1, origin_size[0], 32, device=device)

        out0, (_, _) = self.lstm0(x, (h0, c0))
        _, (h1_n, _) = self.lstm1(out0, (h1, c1))
        
        mix_features = self.mix_block(torch.cat((h1_n[0], latent), 1))

        out = torch.zeros(origin_size[0], self.n_tasks, self.out_seq_len, device=device)
        for i in range(self.n_tasks):
            out[:,i] += self.predict_block[i](mix_features)

        # out(batch_size, len, N x thong so) 
        out = out.permute(0, 2, 1)
        # out(batch_size, N x thong so, len) 
        out = out.reshape((origin_size[0], origin_size[1], origin_size[2], self.out_seq_len))
        return out