import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------
# 显示的是最终多输入多输出模型，之前效果不好的模型在注释中
# -----------------------------------------------------------------------------------------------------------------

# excel文件的位置，有4个sheet
path = "C:\\Users\\Lenovo\\Desktop\\301.xlsx"
# 40条静态数据的测井数据
df = pd.read_excel(path, engine='openpyxl', sheet_name=2)
# 19001条测井数据
# df = pd.read_excel(path, engine='openpyxl', sheet_name=3)
# 相关性分析
hitmapTemp = df[['DEPTH', 'CALI', 'DEN', 'CNL', 'GR', 'RXO', 'RT',
                 'RI', 'AC', 'SP', 'vd', 'Ed', 'vs', 'Es']]
hitmapData = hitmapTemp.corr()
f, ax = plt.subplots(figsize=(8, 8))
plt.tick_params(labelsize=12)
plt.rc('font',family='Times New Roman')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
sns.heatmap(hitmapData, vmax=1, square=True, annot=True, annot_kws={'size':9,'weight':'bold'})
plt.show()

# 301井的测井数据，为excel文件
path = "C:\\Users\\Lenovo\\Desktop\\301.xlsx"
# sheet0数据未归一化，sheet1已经归一化
df = pd.read_excel(path, engine='openpyxl', sheet_name=1)
data = torch.tensor(df.iloc[:, :5].values, dtype=torch.float32)  # 取测井资料 DEN、depth、AC、GR、CNL
label = torch.tensor(df.iloc[:, 8:10].values, dtype=torch.float32) # 数据标签为vs和Es
data_train = data[:15000]  # 总共19001条数据，取前15000为训练集
label_train = label[:15000]
data_test = data[15000:]   # 后4001为测试集
label_test = label[15000:]
# 转换为能放入模型的数据格式
data_train = Variable(torch.from_numpy(np.array(data_train)[np.newaxis, :]))
label_train = Variable(torch.from_numpy(np.array(label_train)[np.newaxis, :]))
data_test = Variable(torch.from_numpy(np.array(data_test)[np.newaxis, :]))
label_test = Variable(torch.from_numpy(np.array(label_test)[np.newaxis, :]))

# 权重初始化的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 定义模型
net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(),nn.Linear(32, 2))
net.apply(init_weights)   # 初始化权重
loss = nn.MSELoss()   # 损失函数为MSE
trainer = torch.optim.Adam(net.parameters(), lr=0.01)
num_epochs = 500    # 训练轮数
train_loss = []  # 训练集的loss
test_loss = []   # 测试集的loss

for epoch in range(num_epochs):
    # 训练模型
    net.train()
    pred_train = net(data_train)
    l = loss(pred_train, label_train)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    train_loss.append(l.item())  # 将每一轮的损失加入列表
    # 测试模型
    net.eval()
    tl = loss(net(data_test), label_test)
    test_loss.append(tl.item())
    # 输出每一轮的具体loss值
    print('epoch:', epoch + 1, ' train_loss:', '{:.4f}'.format(l.item()),
          ' test_loss:', '{:.4f}'.format(tl.item()))

# 作出loss下降曲线图
epoch = [i for i in range(num_epochs)]
figsize = 7.5, 6
plt.rcParams['xtick.direction'] = 'in'  # 坐标轴刻度向内
plt.rcParams['ytick.direction'] = 'in'
figure, ax = plt.subplots(figsize=figsize)
plt.plot(epoch, train_loss, label='train_loss')
plt.plot(epoch, test_loss, label='test_loss')
plt.grid(linestyle="--")
plt.ylim((0, 0.01))
plt.legend(prop={'family':'Times New Roman', 'size':23})
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.show()

'''
vs = torch.tensor(df.iloc[:, 8:9].values, dtype=torch.float32)  # 标签为静态泊松比vs
Es = torch.tensor(df.iloc[:, 9:10].values, dtype=torch.float32)  # 标签为静态弹性模量Es
data_train = data[:15000]
vs_train = vs[:15000]
Es_train = Es[:15000]
data_test = data[15000:]
vs_test = vs[15000:]
Es_test = Es[15000:]
'''

'''
# lstm
class LSNN(nn.Module):
    def __init__(self):
        super(LSNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            batch_first=True,  # input&output会是以batch size为第一维度的特征集
            # e.g.(batch,time_step,input_size)
        )
        self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, 32)), \
                       torch.autograd.Variable(torch.zeros(1, 1, 32)))
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, self.hidden = self.lstm(x, self.hidden)  # hidden_state 也要作为RNN的一个输入
        self.hidden = (Variable(self.hidden[0]), Variable(self.hidden[1]))
        # 可以把这一步去掉，在loss.backward（）中加retain_graph=True，主要是Varible有记忆功能，而张量没有
        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)


LR = 0.3
num_epochs = 50
lstmNN = LSNN()
optimizer = torch.optim.Adam(lstmNN.parameters(), lr=LR)
# optimize all rnn parameters
loss_func = nn.MSELoss()
data_train = Variable(torch.from_numpy(np.array(data_train)[np.newaxis, :]))
vs_train = Variable(torch.from_numpy(np.array(vs_train)[np.newaxis, :]))
Es_train = Variable(torch.from_numpy(np.array(Es_train)[np.newaxis, :]))
data_test = Variable(torch.from_numpy(np.array(data_test)[np.newaxis, :]))
vs_test = Variable(torch.from_numpy(np.array(vs_test)[np.newaxis, :]))
Es_test = Variable(torch.from_numpy(np.array(Es_test)[np.newaxis, :]))

vs_loss_train = []
Es_loss_train = []
vs_loss_test = []
Es_loss_test = []

for epoch in range(num_epochs):
    lstmNN.train()
    # shape (batch, time_step, input_size)
    pred_train_vd = lstmNN(data_train)  # rnn 对于每个epoch的prediction
    pred_train_Ed = lstmNN(data_train)
    pred_train_vs = lstmNN(data_train)
    pred_train_Es = lstmNN(data_train)

    l_vs = loss_func(pred_train_vs, vs_train)
    l_Es = loss_func(pred_train_Es, Es_train)

    vs_loss_train.append(l_vs.detach().numpy())
    Es_loss_train.append(l_Es.detach().numpy())

    optimizer.zero_grad()  # clear gradients for this training step

    # backpropagation, compute gradients
    l_vs.backward()
    l_Es.backward()

    optimizer.step()  # apply gradients

    lstmNN.eval()
    pred_test_vs = lstmNN(data_test)
    pred_test_Es = lstmNN(data_test)

    tl_vs = loss_func(pred_test_vs, vs_test)
    vs_loss_test.append(tl_vs.detach().numpy())
    tl_Es = loss_func(pred_test_Es, Es_test)
    Es_loss_test.append(tl_Es.detach().numpy())
'''
# ANN
'''
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)
'''
'''
vs_train = Variable(torch.from_numpy(np.array(vs_train)[np.newaxis, :]))
Es_train = Variable(torch.from_numpy(np.array(Es_train)[np.newaxis, :]))
vs_test = Variable(torch.from_numpy(np.array(vs_test)[np.newaxis, :]))
Es_test = Variable(torch.from_numpy(np.array(Es_test)[np.newaxis, :]))
batch_size = 100
'''

'''
vs_train_iter = load_array((data_train, vs_train), batch_size)
Es_train_iter = load_array((data_train, Es_train), batch_size)
vs_test_iter = load_array((data_test, vs_test), batch_size)
Es_test_iter = load_array((data_test, Es_test), batch_size)
'''

'''
# ANN
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 100

vs_train_iter = load_array((data_train, vs_train), batch_size)
Es_train_iter = load_array((data_train, Es_train), batch_size)
vs_test_iter = load_array((data_test, vs_test), batch_size)
Es_test_iter = load_array((data_test, Es_test), batch_size)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 1))
net.apply(init_weights)   # 初始化权重
loss = nn.MSELoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.3)
num_epochs = 200
vs_loss_train = []
Es_loss_train = []
vs_loss_test = []
Es_loss_test = []

for epoch in range(num_epochs):
    net.train()
    for X, y in vs_train_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l_vs = loss(net(data_train), vs_train)
    vs_loss_train.append(l_vs.detach().numpy())
    for X, y in Es_train_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l_Es = loss(net(data_train), Es_train)
    Es_loss_train.append(l_Es.detach().numpy())

    net.eval()
    tl_vs = loss(net(data_test),vs_test)
    vs_loss_test.append(tl_vs.detach().numpy())
    tl_Es = loss(net(data_test),Es_test)
    Es_loss_test.append(tl_Es.detach().numpy())

    print(f'epoch {epoch + 1}, vs_loss_train {l_vs:f}, vs_loss_test {tl_vs:f}, '
          f'Es_loss_train {l_Es:f}, Es_loss_test {tl_Es:f}')

epoch = [i for i in range(num_epochs)]
figsize = 7.5, 6
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
figure, ax = plt.subplots(figsize=figsize)
plt.plot(epoch, vs_loss_train, label='vs_loss_train')
plt.plot(epoch, Es_loss_train, label='Es_loss_train')
plt.plot(epoch, vs_loss_test, label='vs_loss_test')
plt.plot(epoch, Es_loss_test, label='Es_loss_test')
plt.grid(linestyle="--")
plt.legend(prop={'family':'Times New Roman', 'size':23})
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
plt.show()
'''
