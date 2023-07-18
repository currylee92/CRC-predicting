import numpy as np

# 定义后融合的函数
def late_fusion(predictions, weights):
    # 按照权重对预测结果进行加权
    weighted_predictions = np.array(predictions) * np.array(weights)[:, None]
    # 计算加权后的预测结果
    final_prediction = np.sum(weighted_predictions, axis=0) / np.sum(weights)

    # 返回最终预测结果
    return np.argmax(final_prediction, axis=1)
