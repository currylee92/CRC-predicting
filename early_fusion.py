from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 定义前融合的函数
def early_fusion(ml_model, bert_model, X_train, texts, y_train, max_length=128):
    # 提取机器学习模型的特征
    ml_features = ml_model.predict_proba(X_train)

    # 提取BERT模型的特征
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_features = bert_outputs.logits.numpy()

    # 将特征进行拼接
    fused_features = np.concatenate([ml_features, bert_features], axis=1)

    # 使用网格搜索训练融合模型
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
    grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5)
    grid_search.fit(fused_features, y_train)

    return grid_search.best_estimator_
