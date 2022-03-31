
#在此处粘贴你的源代码：
#注意格式、字体、字号
import pandas as pd
import matplotlib.pyplot as plt
#导入决策树函数库
from sklearn.tree import DecisionTreeClassifier
#导入sklearn 中自带的数据集中的wine，txt数据集
from sklearn.datasets import load_wine
#导入随机森林函数库
from sklearn.ensemble import RandomForestClassifier
#导入划分训练集和测试集的函数库，交叉验证函数库
from sklearn.model_selection import train_test_split,cross_val_score
#导入精度评分、混淆矩阵函数库
from sklearn.metrics import accuracy_score,confusion_matrix

# 导入原数据文件
taitan = pd.read_csv(r'D:\作业\数据分析\train.csv')

# 查看一下
df = pd.read_csv(r'D:\作业\数据分析\train.csv', names=['乘客ID','是否幸存','仓位等级','姓名','性别','年龄','兄弟姐妹个数','父母子女个数','船票信息','票价','客舱','登船港口'],index_col='乘客ID',header=0)
print(df.head(5))

# 数据预处理
# 将年龄空的设为年龄中位数
taitan["Age"] = taitan["Age"].fillna(taitan["Age"].median())

# 将空缺的性别设为男
taitan["Sex"] = taitan["Sex"].fillna("male")
# 将男女字符串离散成0和1
taitan.loc[taitan["Sex"] == "male", "Sex"] = 0
taitan.loc[taitan["Sex"] == "female", "Sex"] = 1

# 将票价空缺的填为平均值
taitan['Fare'] = taitan['Fare'].fillna(taitan['Fare'].mean())

# 将空缺的登岸港口设为 数量最多的港口
taitan['Embarked'] = taitan['Embarked'].fillna( taitan['Embarked'].value_counts().index[0])
# 将登岸港口离散化为0、1、2
taitan.loc[taitan["Embarked"] == "S", "Embarked"] = 0
taitan.loc[taitan["Embarked"] == "C", "Embarked"] = 1
taitan.loc[taitan["Embarked"] == "Q", "Embarked"] = 2
print(taitan.describe())

k_range = range(1, 31)
cv_scores = []
X = taitan.iloc[:,:]

X = X.drop(["PassengerId","Survived","Name","Ticket","Cabin"], axis = 1)
# 输出检查一下
print(X)

Y = taitan.iloc[:, 1]

print(Y)

# 先根据accuracy算一下分 确定分簇多少
for n in k_range:
    rfc = RandomForestClassifier(n_estimators = n, random_state = 56, n_jobs = -1)
    scores = cross_val_score(rfc, X, Y, cv = 100, scoring = 'accuracy', n_jobs = -1)
    cv_scores.append(scores.mean())
plt.plot(k_range, cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
# print("\n")

# 图很明显 簇为12时accuracy分最高
# 给测试集跑一下分
taitan_list = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']
for s in taitan_list:
    rfc = RandomForestClassifier(n_estimators = 12, random_state = 6, n_jobs = -1)
    scores = cross_val_score(rfc, X, Y, cv = 10, scoring = s , n_jobs = -1)
    print('element = {},  score = {}'.format(s, scores.mean()))

# 导入测验集数据
data_test = pd.read_csv(r'D:\作业\数据分析\test.csv')

# 存储测验集乘客ID
ID = data_test['PassengerId']

df = data_test.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1)
# 用平均值填充空值
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
# 用数量最多项填充


# data_test = df[len(taitan):]
# 将空缺的性别设为男
df["Sex"] = df["Sex"].fillna("male")
# 将男女字符串离散成0和1
df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1
# 将空缺的登岸港口设为最多的港口
df['Embarked'] = df['Embarked'].fillna( df['Embarked'].value_counts().index[0])
# 将登岸港口离散化为0、1、2
df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2
# 输出检验一下
print(df)

# 随机森林预测数据
rfc = RandomForestClassifier(n_estimators = 12, random_state = 6, n_jobs = -1)
rfc = rfc.fit(X, Y)
pred = rfc.predict(df)
# 格式化预测数据
pred = pd.DataFrame({'PassengerId':ID.values, 'Survived':pred})
# 导出预测数据
pred.to_csv(r'D:\作业\数据分析\out.csv',index=None)

