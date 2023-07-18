import argparse
from sklearn.metrics import classification_report
from src.models.ml_models import train_ml_model
from src.models.bert_model import train_bert_model
from src.fusion.early_fusion import early_fusion
from src.fusion.late_fusion import late_fusion
from utils.data_utils import load_data

# 定义主函数
def main(args):
    # 加载数据
    X, y, texts = load_data()

    # 训练机器学习模型
    ml_model_names = ['svm', 'knn', 'dt', 'rf', 'xgb', 'et', 'lgbm']
    ml_models = [train_ml_model(X, y, name) for name in ml_model_names]

    # 训练BERT模型
    bert_model = train_bert_model(texts, y)

    # 选择融合策略
    if args.fusion_strategy == 'early':
        fusion_model = early_fusion(ml_models[4], bert_model, X, texts, y)
    elif args.fusion_strategy == 'late':
        ml_predictions = [model.predict_proba(X) for model in ml_models]
        bert_prediction = bert_model(texts)  # You need to define this function
        predictions = ml_predictions + [bert_prediction]
        weights = [1] * len(predictions)
        fusion_model = late_fusion(predictions, weights)
    else:
        raise ValueError(f'Unknown fusion strategy {args.fusion_strategy}')

    # 打印分类报告
    print(classification_report(y, fusion_model.predict(X)))

# 定义命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to predict liver metastasis')
    parser.add_argument('--fusion_strategy', choices=['early', 'late'], default='early', help='The fusion strategy to use')
    args = parser.parse_args()

    main(args)
