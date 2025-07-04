import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import toad
import json
import os
import warnings
warnings.ignore_warnings = True


def preprocess_data(df):
    """数据预处理"""
    # 编码分类变量
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes # 将分类变量转换为数值编码

    return df

def drop_missing_features(df, missing_rate: float):
    """
    删除缺失率超过阈值的特征
    :param df: 输入数据框
    :param missing_rate: 缺失率阈值
    :return: 处理后的数据框
    """
    # 计算每列的缺失率
    missing_percent = df.isnull().mean()

    # 筛选出缺失率小于阈值的列
    cols_to_keep = missing_percent[missing_percent < missing_rate].index

    return list(cols_to_keep)


def calculate_global_importance(X, y):
    """计算全局特征重要性"""
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X, y)

    importance = pd.Series(xgb.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)



def calculate_dimension_importance(X_dim, y):
    """
    计算维度内特征重要性
    :param X_dim: 当前维度的特征数据
    :param y: 目标变量
    :return: 特征重要性排序
    """
    rf_dim = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_dim.fit(X_dim, y)

    importance = pd.Series(rf_dim.feature_importances_, index=X_dim.columns)
    return importance.sort_values(ascending=False)


def visualize_dimension_results(dim_name, results):
    """可视化维度分析结果
    :param dim_name: 指定维度的变量
    :param target_name: 各指标结果
    :return: 无
    """
    plt.rcParams['font.family'] = 'SimHei' # 设置中文字体
    plt.figure(figsize=(16, 10)) # 设置图形大小

    # IV值排序
    plt.subplot(311)
    iv_df = results[['feature','iv']]
    sns.barplot(x='iv', y='feature', data=iv_df.head(10))
    plt.title(f'{dim_name} - Top 10 Features by IV')

    # 特征重要性排序
    plt.subplot(312)
    imp_df = results[['feature','importance']].head(10)
    sns.barplot(x='importance', y='feature', data=imp_df)
    plt.title(f'{dim_name} - Top 10 Features by Importance')

    # 目标变量相关性排序
    # 特征重要性排序
    plt.subplot(313)
    imp_df = results[['feature','target_corr']].head(10)
    sns.barplot(x='target_corr', y='feature', data=imp_df)
    plt.title(f'{dim_name} - Top 10 Features by Target_corr')

    plt.tight_layout()
    plt.savefig(f'./result/{dim_name}.png')
    plt.close()
    return

def automated_feature_screening(df, target, dimensions):
    """
    自动化特征筛选主函数
    :param df: 包含特征和目标的DataFrame
    :param target_name: 目标变量名
    :return: 无
    """
    results = {}

    # 数据预处理
    df = preprocess_data(df.copy())
    df = df[df[target].isin([0, 1])]

    # 1. 整体特征重要性分析（随机森林）
    # print("计算全局特征重要性...")
    # global_feature_importance = calculate_global_importance(X_train, y_train)

    # 按维度分析
    for dim_name, features in dimensions.items():
        # 选出变量名和缺失率阈值
        print(f"\n分析维度: {dim_name}")
        missing_rate = features['缺失率阈值']
        dim_features = features['变量名']
        dim_features = drop_missing_features(df[dim_features], missing_rate=missing_rate)

        # 计算IV值
        iv_scores = toad.quality(df[dim_features+[target]], target=target, iv_only=True)['iv']

        # 计算与目标变量的相关性
        target_corr = df[dim_features + [target]].corr()[[target]].sort_values(by=target, ascending=False)
        target_corr = target_corr[target_corr.index != target]

        # 计算维度内特征重要性
        dim_importance = calculate_dimension_importance(df[dim_features], df[target])

        # 保存结果
        results[dim_name] = pd.DataFrame({
            'feature': list(iv_scores.index),
            'iv': list(iv_scores.values),
            'importance': dim_importance.values,
            'target_corr': target_corr[target].values
        })

        # 可视化当前维度结果
        visualize_dimension_results(dim_name, results[dim_name])

    # # 3. 相关性分析（跨维度）
    # print("\n执行跨维度相关性分析...")
    # cross_dimension_correlation_analysis(df_processed)
    # 将字典写入 JSON 文件
    combined_df = pd.concat(results.values(), keys=results.keys(), names=["Dim"])
    combined_df.to_csv("./result/result.csv")

    return

if __name__ == "__main__":
    os.makedirs("./result", exist_ok=True)
    # 读取参数
    config = json.load(open("test.json", "r", encoding="utf-8"))
    dimensions = config['维度'] # 读取各维度及其变量
    target = config['target_name'] # 读取目标变量名
    label_path = config['path']['label_path']
    feature_path = config['path']['feature_path']
    # 读取数据
    df_label = pd.read_csv(label_path, sep=',', dtype={"appl_no": 'int64'})
    df_feature = pd.read_csv(feature_path, dtype={"appl_no": 'int64'})
    df = pd.merge(df_label, df_feature, on="serial_id", how="inner")

    # 执行自动化特征筛选
    automated_feature_screening(df, target, dimensions)




