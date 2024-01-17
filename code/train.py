# ALL DATA TRAIN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from utils import ConfusionMatrix
from models import  VGG19, VGG19_118, VGG16, SwinTransformer_V2, CNN_LSTM, VGGLSTM
from VGG_14BUS import vgg3_14
from VGG_118BUS import vgg6_118


def main():
    
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    
    class MyDataset(Dataset):
        def __init__(self, df):
            xy = df.values
            self.len = xy.shape[0] # 样本数量
            self.features = torch.from_numpy(xy[:, :-1])
            self.labels = torch.from_numpy(xy[:, -1])

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.features[index], self.labels[index] # 返回元组

    
    # 读取文件 1/3
    data_df = pd.read_csv("data/balance_data/data_train.csv", header=None)
    # data_df = pd.read_csv("data/balance_data118/118data_train.csv", header=None)
    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=44)
    
    # 读取数据集
    dataset_train = MyDataset(train_df)
    dataset_val = MyDataset(val_df)
    BATCH_SIZE = 128*2
    # 加载数据集
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(dataset=dataset_val,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)
    # 选择模型
    # 2/3
    model = vgg3_14.VGG16().cuda()
    # model = vgg6_118.VGG19().cuda()
    
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    
    # 保存训练结果 3/3
    base_dir = "./logs_Cen/14BUS/14exp"
    # base_dir = "./logs_Cen/118BUS/118exp"

    # 初始化一个计数器
    counter = 1

    # 循环直到找到一个不存在的目录名
    while True:
        # 创建新目录名
        logdir = Path(f"{base_dir}{counter}")

        # 检查目录是否存在
        if not logdir.exists():
            break

        # 增加计数器以尝试下一个目录名
        counter += 1

    # 创建新目录
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)

    
    epochs = 100
    best_acc = 0.0
    best_epoch = 0
    # 保存模型
    save_path_best = logdir / 'best.pth'
    save_path_final = logdir / 'final.pth'
    # 保存训练过程数据
    csv_filename = logdir / 'training_metrics.csv'
    #打开文件
    with open(csv_filename, mode='w', newline='') as file:
        cvswriter = csv.writer(file)
        cvswriter.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1',
                                     'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1',
                                                                               'label0_Precision', 'label0_Recall', 'label0_F1',
                                                                               'label1_Precision', 'label1_Recall', 'label1_F1',
                                                                               'label2_Precision', 'label2_Recall', 'label2_F1',
                                                                               'label3_Precision', 'label3_Recall', 'label3_F1',])
        
        
        # 迭代过程
        for epoch in range(epochs):
            confusionMatrix = ConfusionMatrix(4, labels=[0, 1, 2, 3])
            # train
            model.train()
            sum_loss = 0
            Precision = 0
            Recall = 0
            F1 = 0
            PrecisionS = []
            RecallS = []
            F1S = []
            acc = 0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar, 0):  #train_loader存的是分割组合后的小批量训练样本和对应的标签
                inputs, labels = data #inputs labels都是张量
                inputs = inputs.float()
                inputs = inputs.view(inputs.size(0), 1, inputs.size(1)).cuda()
                labels = labels.long().cuda()
                y_pred = model(inputs)
                predict_y = torch.max(y_pred, dim=1)[1]
                acc += torch.eq(predict_y, labels).sum()

                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                # 进度显示
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
                confusionMatrix.update(predict_y, labels)
            Precision, Recall, F1, PrecisionS, RecallS, F1S = confusionMatrix.summary()
            

            train_accurate = acc / len(dataset_train)
            train_loss = sum_loss / len(train_loader)
            print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f' % (epoch + 1, train_loss, train_accurate))
            # 写入tensorboard
            writer.add_scalar('train_acc', train_accurate, epoch + 1)
            writer.add_scalar('train_loss', train_loss, epoch + 1)
            writer.add_scalar('train_Precision', Precision, epoch + 1)
            writer.add_scalar('train_Recall', Recall, epoch + 1)
            writer.add_scalar('train_F1', F1, epoch + 1)


            model.eval()
            sum_loss2 = 0
            Precision_v = 0
            Recall_v = 0
            F1_v = 0
            Precision_vS = []
            Recall_vS = []
            F1_vS = []
            accuracy = 0
            with torch.no_grad():
              confusionMatrix = ConfusionMatrix(4, labels=[0, 1, 2, 3])
              val_bar = tqdm(val_loader, file=sys.stdout)
              for batch_X, batch_y in val_bar:
                  batch_X = batch_X.float()
                  batch_X = batch_X.view(batch_X.size(0), 1, batch_X.size(1)).cuda()
                  batch_y = batch_y.long().cuda()
                  y_pred = model(batch_X)
                  predict_y = torch.max(y_pred, dim=1)[1]
                  accuracy += torch.eq(predict_y, batch_y).sum()

                  loss = criterion(y_pred, batch_y)

                  sum_loss2 += loss.item()
                  # 进度显示
                  val_bar.desc = "val epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
                  confusionMatrix.update(predict_y, batch_y)
              Precision_v, Recall_v, F1_v, Precision_vS, Recall_vS, F1_vS = confusionMatrix.summary()


            val_accuracy = accuracy / len(dataset_val)
            val_loss = sum_loss2 / len(val_bar)
            print('[epoch %d] val_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, val_loss, val_accuracy))
            # 写入tensorboard
            writer.add_scalar('Validata_acc', val_accuracy, epoch + 1)
            writer.add_scalar('Validata_loss', val_loss, epoch + 1)
            writer.add_scalar('Validata_Precision', Precision_v, epoch + 1)
            writer.add_scalar('Validata_Recall', Recall_v, epoch + 1)
            writer.add_scalar('Validata_F1', F1_v, epoch + 1)
            # 保存最好模型
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_epoch = epoch + 1
                print("The best model is saving\n ")
                torch.save(model.state_dict(), save_path_best)
            # 保存训练过程数据
            train_metrics = [epoch + 1, train_loss, float(train_accurate), Precision, Recall, F1]
            val_metrics = [val_loss, float(val_accuracy), Precision_v, Recall_v, F1_v]

            # 将每个类别的指标添加到列表中
            for i in range(4):
                val_metrics.extend([Precision_vS[i], Recall_vS[i], F1_vS[i]])

            # 合并训练集和验证集的指标并写入 CSV
            cvswriter.writerow(train_metrics + val_metrics)
            
            # cvswriter.writerow([epoch + 1, train_loss, float(train_accurate), Precision, Recall, F1, 
            #                                val_loss, float(val_accuracy), Precision_v, Recall_v, F1_v,
            #                                                               Precision_vS[0], Recall_vS[0], F1_vS[0],
            #                                                               Precision_vS[1], Recall_vS[1], F1_vS[1],
            #                                                               Precision_vS[2], Recall_vS[2], F1_vS[2],
            #                                                               Precision_vS[3], Recall_vS[3], F1_vS[3]])
            
            
    # 保存最后一轮模型
    torch.save(model.state_dict(), save_path_final)
    print("The best Validate epoch is {}".format(best_epoch))
    print("The best Validate accuracy is {}".format(best_acc))
    print("Finished Training, exp is {}".format(counter))
    

if __name__ == '__main__':
    main()

        

