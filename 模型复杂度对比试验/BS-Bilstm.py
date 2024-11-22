"""
第五版增加了SSA对角平均
第六版 将BO模型从TPU改成了GP
paotong-----khjfiowjk
改了r值
"""
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.transforms as transforms
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from numpy import concatenate
import pywt  # 引入pywt库
from hyperopt import fmin, tpe, hp, Trials
# 忽略可能的数值问题警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置用于训练的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import time
# 定义计时器类
class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            self.start_time = None
        return self.elapsed_time

# SSA分解函数
def ssa_decompose(time_series, window_length):
    series_length = len(time_series)
    K = series_length - window_length + 1
    X = np.zeros((window_length, K))
    for i in range(window_length):
        X[i, :] = time_series[i:i + K]
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Z = np.zeros((VT.shape))
    for n in range(len(Sigma)):
        Z[n, :] = Sigma[n] * VT[n, :]
    return U, Z, series_length, window_length, Sigma

# 对角平均函数
def diagonal_averaging(X_reconstructed, series_length, window_length):
    K = series_length - window_length + 1
    reconstructed = np.zeros(series_length)
    weight = np.zeros(series_length)

    for i in range(window_length):
        for j in range(K):
            reconstructed[i + j] += X_reconstructed[i, j]
            weight[i + j] += 1

    return reconstructed / weight

# SSA重构函数
def ssa_reconstruct(U, Z, series_length, window_length, r):
    X_reconstructed = np.dot(U[:, :r], Z[:r, :])
    reconstructed = diagonal_averaging(X_reconstructed, series_length, window_length)
    return reconstructed

# 小波降噪函数
def wavelet_denoising(data, wavelet='db4', level=1, threshold_scale=0.5):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data))) * threshold_scale
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    denoised_data = pywt.waverec(coeff, wavelet, mode='per')
    return denoised_data[:len(data)]  # 确保长度一致

# 使用Local Outlier Factor检测离群点
def knn_outlier_detection(data):
    lof = LocalOutlierFactor(n_neighbors=70, contamination=0.1)
    outliers = lof.fit_predict(data.reshape(-1, 1))
    return np.where(outliers == -1)[0]

# 使用Isolation Forest检测离群点
def isolation_forest_detection(data):
    iso = IsolationForest(contamination=0.1)
    outliers = iso.fit_predict(data.reshape(-1, 1))
    return np.where(outliers == -1)[0]
# 替换异常点
def replace_outliers(data, outlier_indices):
    replaced_indices = []
    for idx in outlier_indices:
        # 找到前一个非异常值
        prev_idx = idx - 1
        while prev_idx in outlier_indices and prev_idx > 0:
            prev_idx -= 1
        # 找到后一个非异常值
        next_idx = idx + 1
        while next_idx in outlier_indices and next_idx < len(data) - 1:
            next_idx += 1
        # 用前后非异常值的平均值替换异常点
        if prev_idx >= 0 and next_idx < len(data):
            data[idx] = (data[prev_idx] + data[next_idx]) / 2
            replaced_indices.append(idx)
    return data

# 数据准备函数
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    cols.append(df.iloc[:, :-1])  # Add current sensor data but exclude the target
    names += [('var%d(t)' % (j + 1)) for j in range(df.shape[1] - 1)]

    cols.append(df.iloc[:, -1])  # Only the target variable at time t
    names.append('target(t)')

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 读取CSV文件并获取第0列数据作为总数据
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\实验数据\\0.1ms\\Y\\处理后\\可预测\\删减后\\data-sum.csv')
values = df.values.astype('float32')
values123=values
# 小波降噪
denoised_series = []
for i in range(values.shape[1]):
    denoised = wavelet_denoising(values[:,5])
    denoised_series.append(denoised)
denoised_series = np.array(denoised_series).T

plt.figure(figsize=(10, 6))
plt.plot( values123[:,5], color='red', label='forecast')
plt.plot( denoised_series[:,5], color='blue', label='true')
plt.legend()
plt.title('SSA CNN BiLSTM Attention')
plt.xlabel('time number')
plt.ylabel('Speed value (cm/s)')
plt.show()
# 使用Local Outlier Factor检测离群点
outlier_indices_lof = knn_outlier_detection(denoised_series[:, 5])

# 使用Isolation Forest检测离群点
outlier_indices_iso = isolation_forest_detection(denoised_series[:, 5])
# 替换异常点
denoised_series[:, 5] = replace_outliers(denoised_series[:, 5], outlier_indices_iso)
denoised_series[:, 5] = replace_outliers(denoised_series[:, 5], outlier_indices_lof)


# 数据标准化
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(denoised_series)

# SSA分解和重构
window_length = int(len(values[:,0])/4)
reconstructed_series = []
for i in range(normalized_data.shape[1]):
    if (i < 4):
        time_series = normalized_data[:, i]
        U, Z, series_length, window_length, Sigma = ssa_decompose(time_series, window_length)
        r = window_length  # 可以根据需要调整
        reconstructed = ssa_reconstruct(U, Z, series_length, window_length, r)
        reconstructed_series.append(reconstructed)
    elif(i==4):
        time_series = normalized_data[:, i]
        U, Z, series_length, window_length, Sigma = ssa_decompose(time_series, window_length)
        r = window_length  # 可以根据需要调整
        reconstructed = ssa_reconstruct(U, Z, series_length, window_length, r)
        reconstructed_series.append(reconstructed)
    elif(i==5):
        time_series = normalized_data[:, i]
        U, Z, series_length, window_length, Sigma = ssa_decompose(time_series, window_length)
        r = window_length  # 可以根据需要调整
        reconstructed = ssa_reconstruct(U, Z, series_length, window_length, r)
        reconstructed_series.append(normalized_data[:,i])
reconstructed_series = np.array(reconstructed_series).T

plt.figure(figsize=(10, 6))
plt.plot( reconstructed_series[:200,5], color='blue', label='true')
plt.plot( values123[:200,5], color='red', label='forecast')
plt.legend()
plt.title('SSA CNN BiLSTM Attention')
plt.xlabel('time number')
plt.ylabel('Speed value (cm/s)')
plt.show()
# # 数据标准化
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(reconstructed_series)

# 参数设置
n_hours = 3
n_features = 6
n_obs = n_hours * n_features

# 数据转换为监督学习形式
reframed = series_to_supervised(reconstructed_series, n_hours, 1)
values = reframed.values

# 划分训练集、验证集和测试集
train_size = int(len(values[:, 0]) * 0.7)
valid_size = int(len(values[:, 0]) * 0.1)
test_size = len(values[:, 0]) - train_size - valid_size
train = values[:train_size, :]
valid = reframed.values[train_size:train_size + valid_size, :]
test = reframed.values[train_size + valid_size:, :]


# # 打乱训练数据的顺序
# np.random.shuffle(train)

train_X, train_y = train[:, :n_obs], train[:, -1]
valid_X, valid_y = valid[:, :n_obs], valid[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]

# 转换为3D输入
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
valid_X = valid_X.reshape((valid_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
timer = Timer()

# 转换为 PyTorch 张量
train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)
valid_X = torch.tensor(valid_X, dtype=torch.float32).to(device)
valid_y = torch.tensor(valid_y, dtype=torch.float32).unsqueeze(1).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1).to(device)

# 定义Attention机制类
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        a = self.attention(inputs)
        a = self.softmax(a)
        output_attention_mul = inputs * a
        return output_attention_mul

# 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2 * n_hours, 8)
        self.fc2 = nn.Linear(8, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 定义训练模型的函数
def train_model(params):
    hidden_dim = int(params['hidden_dim'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    num_epochs = 50  # 训练集
    n_layers = 1

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = BiLSTM(n_features, hidden_dim, 1, n_layers).to(device)  # 将模型移到GPU
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始计时
    timer.start()

    early_stopping_patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, targets)

            weights = torch.ones_like(targets).to(device)
            # 可选权重调整代码
            # threshold = 0.8
            # lowhold = 0.3
            # weights[targets > threshold] = 1.5  # 对峰值部分的权重设置为2.0

            # 计算加权损失
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(valid_X)
            val_loss = criterion(val_output, valid_y).item()

        print(f'Epoch {epoch + 1}: Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')

        # 检查是否是最佳验证损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 检查早停条件
        if epochs_no_improve >= early_stopping_patience:
            elapsed_time = timer.stop()  # 停止计时
            print(f'Early stopping triggered at epoch {epoch + 1}. Total training time: {elapsed_time:.2f} seconds.')
            break

    # 如果未触发早停，记录总时间
    if timer.start_time is not None:
        elapsed_time = timer.stop()
        print(f'Total training time without early stopping: {elapsed_time:.2f} seconds.')

    return val_loss

# 定义超参数空间
space = {
    'hidden_dim': hp.quniform('hidden_dim', 10, 300, 1),
    'learning_rate': hp.loguniform('learning_rate', -7, -1),
    'batch_size': hp.quniform('batch_size', 16, 128, 1)
}

trials = Trials()
best = fmin(fn=train_model, space=space, algo=tpe.suggest, max_evals=5, trials=trials)

print("最佳超参数:", best)

# 使用最佳超参数重新训练和评估模型
best_hidden_dim = int(best['hidden_dim'])
best_learning_rate = best['learning_rate']
best_batch_size = int(best['batch_size'])
num_epochs = 100  # 验证集
n_layers = 1

train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=best_batch_size, shuffle=True)
model = BiLSTM(n_features, best_hidden_dim, 1, n_layers).to(device)  # 将模型移到GPU
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

train_losses = []
val_losses = []
timer.start()

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()  # 确保模型处于训练模式
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # 评估模型的验证损失
    model.eval()  # 将模型置于评估模式
    with torch.no_grad():
        val_output = model(valid_X)
        val_loss = criterion(val_output, valid_y).item()
        val_losses.append(val_loss)

    print(f'第{epoch + 1}轮训练，训练损失: {running_loss / len(train_loader):.4f}, 验证损失: {val_loss:.4f}')
# 停止计时
elapsed_time = timer.stop()
print(f'Total training time 在后面训练的: {elapsed_time:.2f} seconds')

# 训练完成后绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='train_losses')
plt.plot(val_losses, label='val_losses')
plt.legend()
plt.title('train_losses and val_losses')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 预测并评估模型
predictions = model(test_X).detach().cpu().numpy()
test_y = test_y.cpu().numpy()
# 将test_X调整为2个维度
test_X_flattened = test_X.reshape((test_X.shape[0], n_hours * n_features)).cpu().numpy()

# 合并 test_X 和 test_y 以便进行反向转换
inv_y = concatenate((test_X_flattened[:, -5:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# 合并 test_X 和 predictions
inv_yhat = concatenate((test_X_flattened[:, -5:], predictions), axis=1)
# 对合并后的数据进行反向转换
inv_yhat = scaler.inverse_transform(inv_yhat)
# 获取预测结果
inv_yhat = inv_yhat[:, -1]

# 计算均方误差
mse = mean_squared_error(inv_y, inv_yhat)
# 计算 RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
# 计算 MAE
mae = mean_absolute_error(inv_y, inv_yhat)
# 计算 MAPE
mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100
# 计算 R2
r2 = r2_score(inv_y, inv_yhat)
print('TEST MSE: %.3f' % mse)
print('TEST RMSE: %.3f' % rmse)
print('TEST MAE: %.3f' % mae)
print('TEST MAPE: %.3f' % mape)
print('TEST R2: %.3f' % r2)
