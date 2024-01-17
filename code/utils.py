import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库: pip install prettytable
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵
        self.num_classes = num_classes
        self.labels = labels

    # 混淆矩阵更新
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # 计算并打印评价指标
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 对角线元素求和
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, F1
        table = PrettyTable()
        # sum_TP = 0
        # sum_FP = 0
        # sum_FN = 0
        # sum_TN = 0
        Ps = []
        Rs = []
        Fs = []
        Ave_Precision = 0
        Ave_Recall = 0
        Ave_F1 = 0
        table.field_names = ["", "Precision", "Recall", "F1"]  # 第一个元素是类别标签
        for i in range(self.num_classes):  # 针对每个类别进行计算
            # 整合其他行列为不属于该类的情况
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            
            # sum_TP += TP
            # sum_FP += FP
            # sum_FN += FN
            # sum_TN += TN
            # 每个类别的P, R, F指标
            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.  # 注意分母为 0 的情况
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            F1 = round(2*Precision*Recall / (Precision + Recall), 5)
            table.add_row([self.labels[i], Precision, Recall, F1])
            Ps.append(Precision)
            Rs.append(Recall)
            Fs.append(F1)
            # 平均指标
            Ave_Precision += Precision / self.num_classes
            Ave_Recall += Recall / self.num_classes
            
            
        # Precision = round(sum_TP / (sum_TP + sum_FP), 3) if sum_TP + sum_FP != 0 else 0.  # 注意分母为 0 的情况
        # Recall = round(sum_TP / (sum_TP + sum_FN), 3) if sum_TP + sum_FN != 0 else 0.
        # F1 = round(2*Precision*Recall / (Precision + Recall), 3)
        Ave_Precision = round(Ave_Precision, 5)
        Ave_Recall = round(Ave_Recall, 5)
        Ave_F1 = round(2*Ave_Precision*Ave_Recall / (Ave_Precision + Ave_Recall), 5)
        table.add_row(["Average", Ave_Precision, Ave_Recall, Ave_F1])
        print(table)
        return Ave_Precision, Ave_Recall, Ave_F1, Ps, Rs, Fs

    # 可视化混淆矩阵
    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)  # 从白色到蓝色

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)  # x 轴标签旋转 45 度方便展示
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                # 画图的时候横坐标是x，纵坐标是y
                info = int(matrix[y, x])
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()  # 图形显示更加紧凑
        plt.savefig("./ConfusionMatrix_50.png")
        plt.close('all')
        # plt.show()
