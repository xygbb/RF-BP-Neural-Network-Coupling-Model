import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
import joblib

# 读取原始Excel文件
input_excel_file = r"C:\Users\于博帆\Desktop\sci\地面塌陷训练集.xlsx"
df = pd.read_excel(input_excel_file)

# 提取特征列 (前7列) 和目标列 (第8列)
X = df.iloc[:, :7]  # 前7列为特征
y = df.iloc[:, 7]   # 第8列为目标

# 创建Random Forest分类模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 创建神经网络模型
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# 创建一个元分类器，这里使用Random Forest和神经网络作为基分类器
base_models = [('RandomForest', rf_model), ('NeuralNetwork', nn_model)]

# 创建Stacking模型，使用Random Forest作为元分类器
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# 使用交叉验证进行模型评估
kf = KFold(n_splits=5)
accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

average_accuracy = sum(accuracies) / len(accuracies)
print(f'平均准确度: {average_accuracy:.2f}')

# 将集成模型保存到文件
model_filename = r'D:\jqxx\stacked_model.pkl'
joblib.dump(stacking_model, model_filename)

print("使用Random Forest作为基模型的 Stacking 模型已训练并保存。")

import pandas as pd
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from itertools import cycle

# 读取要分析的Excel文件
test_excel_file = r"C:\Users\于博帆\Desktop\sci\塌陷测试集.xlsx"
test_df = pd.read_excel(test_excel_file)

model_filename = r'D:\jqxx\stacked_model.pkl'
stacking_model = joblib.load(model_filename)

# 提取特征列 (前7列)
X_test = test_df.iloc[:, :7]
# 根据实际测试数据创建 y_test
y_test = test_df.iloc[:, 7]

# 对 y_test 进行二值化处理，假设非零的最大值是正类标签
positive_class_label = y_test.max()
y_test_binary = label_binarize(y_test, classes=[0, positive_class_label]).ravel()

# 计算预测的概率值，用于绘制ROC曲线
y_score = stacking_model.predict_proba(X_test)

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('受试者工作特性曲线')
plt.legend(loc="lower right")

# 保存ROC曲线图
roc_curve_filename = r'D:\jqxx\roc_curve神经.png'
plt.savefig(roc_curve_filename)

# 将预测结果保存到Excel文件
test_df['预测结果'] = stacking_model.predict(X_test)
result_excel_file = r'D:\jqxx\predicted_results神经.xlsx'
test_df.to_excel(result_excel_file, index=False)

print("保存预测结果和ROC曲线。")

