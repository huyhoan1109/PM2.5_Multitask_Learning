from modules import *

DIR = './data/'

class Process_Data:
    
    def __init__(self, dir_name=DIR, get_process=False, scaler=MinMaxScaler()) -> None:
        
        self.path = dir_name
        self.scaler = scaler
        self.get_process = get_process
        self.file_names = [self.path + name for name in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, name))]
        
        if get_process:
            self.processing()
    
    def processing(self):
        
        self.num_files = len(self.file_names)
        self.raw_data = dict()
        self.stations = list()
        for i in range(self.num_files):
    
            df_raw = pd.read_csv(self.file_names[i])
            station = df_raw['station'][0]
            self.stations.append(station)
            df_raw.drop(columns=['No'],inplace=True)
            self.raw_data[station] = df_raw

        date_time = ['year','month','day','hour']
        
        self.process_data = dict()
        self.max_vals = dict()
        self.min_vals = dict()

        for station in self.stations:
            
            iter_max = dict()
            iter_min = dict()

            df_proc = pd.DataFrame(self.raw_data[station])

            df_proc['datetime'] = pd.to_datetime(df_proc[date_time])
            df_proc.drop(columns=date_time+['station'], inplace=True)

            df_proc['wd'] = df_proc['wd'].astype('category').cat.codes

            for col in df_proc.columns:
                if col != 'datetime':
                    iter_max[col] = df_proc[col].max()
                    iter_min[col] = df_proc[col].min()
                    df_proc[col].fillna(np.mean(df_proc[col]), inplace=True) 

            self.max_vals[station] = iter_max
            self.min_vals[station] = iter_min
            
            scale = self.scaler
            df_scaled = scale.fit_transform(df_proc.loc[:, df_proc.columns != 'datetime'].to_numpy())
            df_scaled = pd.DataFrame(df_scaled, columns=[col for col in df_proc.columns if col != 'datetime'])
            df_scaled['datetime'] = df_proc['datetime']
            df_scaled.set_index(['datetime'], inplace=True)
            self.types = [col for col in df_scaled.columns if col != 'station']
            self.process_data[station] = df_scaled
        
    def split_and_to_tensor(self, tr_size=0.8):
        
        df = dict()
        for station in self.stations:
            df[station] = self.process_data[station]
        
        meteo_cols = ['TEMP','PRES','DEWP','RAIN','wd','WSPM']
        
        # X(trạm, thời gian, thông số trạm)
        # X_pm2_5 = np.array([df_i[[col for col in df_i.columns if col not in meteo_cols]] for df_i in df.values()])
        X_pm2_5 = np.array([df_i[['PM2.5']] for df_i in df.values()])
        X_meteo = np.array([df_i[meteo_cols] for df_i in df.values()])
        
        # Chuyen sang tensor
        X_meteo = Variable(torch.from_numpy(X_meteo))
        X_pm2_5 = Variable(torch.from_numpy(X_pm2_5))
        
        # Split train and test
        time_len = X_pm2_5.shape[1]
        train_size = int(time_len * 0.8)
        X_meteo_train = X_meteo[:,:train_size,:]
        X_meteo_test = X_meteo[:,train_size:,:]
        X_pm2_5_train = X_pm2_5[:,:train_size,:]
        X_pm2_5_test = X_pm2_5[:,train_size:,:]
        
        return X_meteo_train, X_meteo_test, X_pm2_5_train, X_pm2_5_test
        

def create_seq_samples(data_, in_seq_len, out_seq_len):

    # X(N, thời gian, thông số khí của trạm)
    T = data_.shape[1]
    sample_len = T - in_seq_len - out_seq_len
    in_sample = torch.zeros(size=(data_.shape[0], sample_len,  in_seq_len, data_.shape[2]))
    out_sample = torch.zeros(size=(data_.shape[0], sample_len,  out_seq_len, data_.shape[2]))
    for i in range(0, sample_len):
        in_sample[:,i,:,:] = data_[:,i:i+in_seq_len,:]
        out_sample[:,i,:,:] = data_[:,i+in_seq_len:i+in_seq_len+out_seq_len,:]

    # => Chuyen sample ve (N, các thông số trạm, số lượng time series, seq_len)
    in_sample = torch.permute(in_sample, (0,3,1,2))
    out_sample = torch.permute(out_sample, (0,3,1,2))
    return in_sample, out_sample

class AQIdatasets(Dataset):
    
    def __init__(self, x, y) -> None:
        super(AQIdatasets).__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[2]

    def __getitem__(self, index):
        return self.x[:,:,index], self.y[:,:,index]