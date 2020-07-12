import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 减少内存使用
from functions import reduce_mem_usage, log_transfer, plot_learning_curve, rf_cv

# 4.4.1 读取数据
sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))
continuous_feature_names = [x for x in sample_feature.columns if x not in ['price', 'brand', 'model', 'brand']]

# 4.4.2 线性回归 & 五折交叉验证 & 模拟真实业务情况
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']

# 4.4.2 - 1 简单建模
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
# 查看训练的线性回归模型的截距（intercept）与权重(coef)
'intercept:' + str(model.intercept_)
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True)

from matplotlib import pyplot as plt

subsample_index = np.random.randint(low=0, high=len(train_y), size=50)

# 绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布差异较大，
# 且部分预测值出现了小于0的情况，说明我们的模型存在一些问题
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price', 'Predicted Price'], loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()

# 通过作图我们发现数据的标签（price）呈现长尾分布，不利于我们的建模预测。
# 原因是很多模型都假设数据误差项符合正态分布，而长尾分布的数据违背了这一假设。
# 参考博客：https://blog.csdn.net/Noob_daniel/article/details/76087829

import seaborn as sns

print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(train_y)
plt.subplot(1, 2, 2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])

# 在这里我们对标签进行了  log(x+1)变换，使标签贴近于正态分布
train_y_ln = np.log(train_y + 1)
import seaborn as sns

print('The transformed price seems like normal distribution')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(train_y_ln)
plt.subplot(1, 2, 2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])

model = model.fit(train_X, train_y_ln)

print('intercept:' + str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True)

# 再次进行可视化，发现预测结果与真实值较为接近，且未出现异常状况
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price', 'Predicted Price'], loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()

# 4.4.2 - 2 五折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv=5,
                         scoring=make_scorer(log_transfer(mean_absolute_error)))
# 使用线性回归模型，对未处理标签的特征数据进行五折交叉验证（Error 1.36）
print('AVG:', np.mean(scores))
# 使用线性回归模型，对处理过标签的特征数据进行五折交叉验证（Error 0.19）
scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv=5, scoring=make_scorer(mean_absolute_error))
print('AVG:', np.mean(scores))
scores = pd.DataFrame(scores.reshape(1, -1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
print(scores)

# 4.4.2 - 3 模拟真实业务情况
# 但在事实上，由于我们并不具有预知未来的能力，五折交叉验证在某些与时间相关的数据集上反而反映了不真实的情况。
# 通过2018年的二手车价格预测2017年的二手车价格，这显然是不合理的，因此我们还可以采用时间顺序对数据集进行分隔。
# 在本例中，我们选用靠前时间的4/5样本当作训练集，靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大
import datetime

sample_feature = sample_feature.reset_index(drop=True)
split_point = len(sample_feature) // 5 * 4
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)
model = model.fit(train_X, train_y_ln)
mean_absolute_error(val_y_ln, model.predict(val_X))

# 4.4.2 - 4 绘制学习率曲线与验证曲线
from sklearn.model_selection import learning_curve, validation_curve

# ? learning_curve

plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5,
                    n_jobs=1)

# 4.4.3 多种模型对比
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)

# 4.4.3 - 1 线性模型 & 嵌入式特征选择
# 在过滤式和包裹式特征选择方法中，特征选择过程与学习器训练过程有明显的分别。
# 而嵌入式特征选择在学习器训练过程中自动地进行特征选择。
# 嵌入式选择最常用的是L1正则化与L2正则化。在对线性回归模型加入两种正则化方法后，他们分别变成了岭回归与Lasso回归。from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

models = [LinearRegression(),
          Ridge(),
          Lasso()]

result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
# 对三种方法的效果对比
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result

model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)

# L2正则化在拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。
# 因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。
# 可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；
# 但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』
model = Ridge().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)

# L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。如下图，我们发现power与userd_time特征非常重要。
model = Lasso().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)

# 除此之外，决策树通过信息熵或GINI指数选择分裂节点时，优先选择的分裂特征也更加重要，这同样是一种特征选择的方法。
# XGBoost与LightGBM模型中的model_importance指标正是基于此计算的

# 4.4.3 - 2 非线性模型
# 除了线性模型以外，还有许多我们常用的非线性模型如下，在此篇幅有限不再一一讲解原理。
# 我们选择了部分常用模型与线性模型进行效果比对。

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100),
          XGBRegressor(n_estimators=100, objective='reg:squarederror'),
          LGBMRegressor(n_estimators=100)]

result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')

result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)
# 可以看到随机森林模型在每一个fold中均取得了更好的效果


# 4.4.4 模型调参
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3, 5, 10, 15, 20, 40, 55]
max_depth = [3, 5, 10, 15, 20, 40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []

# 4.4.4 - 1 贪心调参
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score

best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0], num_leaves=leaves)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score

best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                          max_depth=depth)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

sns.lineplot(x=['0_initial', '1_turning_obj', '2_turning_leaves', '3_turning_depth'],
             y=[0.143, min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])

# 4.4.4 - 2 Grid Search 调参
from sklearn.model_selection import GridSearchCV

parameters = {'objective': objective, 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)

print(clf.best_params_ ) # print()

model = LGBMRegressor(objective='regression',
                      num_leaves=55,
                      max_depth=15)

np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))

# 4.4.4 - 3 贝叶斯调参
from bayes_opt import BayesianOptimization

rf_bo = BayesianOptimization(
    rf_cv,
    {
        'num_leaves': (2, 100),
        'max_depth': (2, 100),
        'subsample': (0.1, 1),
        'min_child_samples': (2, 100)
    }
)

rf_bo.maximize()

ret = 1 - rf_bo.max['target']
print(ret)