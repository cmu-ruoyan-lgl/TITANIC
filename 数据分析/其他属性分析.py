import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series,DataFrame
taitan = pd.read_csv(r'D:\作业\数据分析\train.csv')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0)) # 在一张大图里分列几个小图
taitan.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title(u"表A 获救情况 (1为获救)") # 标题
plt.ylabel(u"人数") # Y轴标签

plt.subplot2grid((2,3),(0,1))
taitan.Pclass.value_counts().plot(kind="bar") # 柱状图显示
plt.ylabel(u"人数")
plt.title(u"表B 乘客船舱号分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(taitan.Survived, taitan.Age) #为散点图传入数据
plt.ylabel(u"年龄") # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"表C 按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
taitan.Age[taitan.Pclass == 1].plot(kind='kde') # 密度图
taitan.Age[taitan.Pclass == 2].plot(kind='kde')
taitan.Age[taitan.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"表D 各船舱号的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best')

plt.subplot2grid((2,3),(1,2))
taitan.Embarked.value_counts().plot(kind='bar')
plt.title(u"表E 各登船口岸上船人数")
plt.ylabel(u"人数")
# plt.show()

S0 = taitan.Pclass[taitan.Survived == 0].value_counts()
S1 = taitan.Pclass[taitan.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':S1,u'未获救':S0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各船舱号乘客的获救情况')
plt.xlabel(u'船舱号')
plt.ylabel(u'人数')
plt.show()

Sm = taitan.Survived[taitan.Sex == 'male'].value_counts()
Sf = taitan.Survived[taitan.Sex == 'female'].value_counts()
df = pd.DataFrame({u'未获救':Sm,u'获救':Sf})
df.plot(kind = 'bar', stacked = True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()

fig = plt.figure()
plt.title(u'根据船舱号和性别的获救情况')

ax1 = fig.add_subplot(141)
taitan.Survived[taitan.Sex == 'female'][taitan.Pclass != 3].value_counts().plot(kind = 'bar', label = 'female high class', color = '#FA2479')
ax1.set_xticklabels([u'获救',u'未获救'], rotation = 0)
ax1.legend([u'女性/高级舱'], loc = 'best')

ax2 = fig.add_subplot(142, sharey = ax1)
taitan.Survived[taitan.Sex == 'female'][taitan.Pclass == 3].value_counts().plot(kind = 'bar', label = 'female low class', color = 'pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey = ax1)
taitan.Survived[taitan.Sex == 'male'][taitan.Pclass != 3].value_counts().plot(kind = 'bar', label = 'male high class', color = 'lightblue')
ax3.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/高级舱'], loc = 'best')

ax4 = fig.add_subplot(144, sharey = ax1)
taitan.Survived[taitan.Sex == 'male'][taitan.Pclass == 3].value_counts().plot(kind = 'bar', label = 'male low class', color = 'steelblue')
ax4.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/低级舱'], loc = 'best')
plt.show()

fig = plt.figure()
fig.set(alpha = 0.2)
S0 = taitan.Embarked[taitan.Survived == 0].value_counts()
S1 = taitan.Embarked[taitan.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':S1,u'未获救':S0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各登陆港口乘客的获救情况')
plt.xlabel(u'登陆港口')
plt.ylabel(u'人数')
plt.show()
