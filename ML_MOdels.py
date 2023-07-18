from sklearn import svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 定义每个模型的参数网格
param_grids = {
    'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'knn': {'n_neighbors': [3, 5, 10]},
    'dt': {'max_depth': [None, 10, 20]},
    'rf': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'xgb': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
    'et': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'lgbm': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
}

# 定义训练机器学习模型的函数
def train_ml_model(X_train, y_train, model_name):
    # 根据模型名称选择模型
    if model_name == 'svm':
        model = svm.SVC()
    elif model_name == 'knn':
        model = neighbors.KNeighborsClassifier()
    elif model_name == 'dt':
        model = tree.DecisionTreeClassifier()
    elif model_name == 'rf':
        model = RandomForestClassifier()
    elif model_name == 'xgb':
        model = XGBClassifier()
    elif model_name == 'et':
        model = ExtraTreesClassifier()
    elif model_name == 'lgbm':
        model = LGBMClassifier()
    else:
        raise ValueError(f'Unknown model name {model_name}')

    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
    grid_search.fit(X_train, y_train)

    # 返回最优的模型
    return grid_search.best_estimator_
