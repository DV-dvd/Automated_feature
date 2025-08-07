import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, pearsonr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import toad
import json
import os
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
# import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optbinning import OptimalBinning

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

class AutoFeatureScreener:
    def __init__(self, config_path):
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 创建结果目录
        os.makedirs(self.config['result_path'], exist_ok=True)
        os.makedirs(self.config['result_path']+'/init', exist_ok=True)
        os.makedirs(self.config['result_path'] + '/init/fig', exist_ok=True)
        os.makedirs(self.config['result_path'] + '/trn_oot', exist_ok=True)
        os.makedirs(self.config['result_path'] + '/trn_oot/fig', exist_ok=True)
        os.makedirs(self.config['result_path'] + '/monthly', exist_ok=True)
        os.makedirs(self.config['result_path'] + '/monthly/fig', exist_ok=True)


        # 初始化参数
        self.target = self.config['target_name']
        self.missing_rate = self.config['missing_rate']
        self.iv_rate = self.config['iv_rate']
        self.psi_rate = self.config['psi_rate']
        self.importance_rate = self.config['importance_rate']
        self.target_corr_rate = self.config['target_corr']
        self.var_corr_rate = self.config['var_corr']
        self.distribution_std = self.config['distribution_std']
        self.iv_month_std = self.config['iv_month_std']
        self.psi_month_std = self.config['psi_month_std']
        self.mean_month_std = self.config['mean_month_std']
        self.if_scorecard = self.config['if_scorecard']
        self.time_column = self.config['time_column']
        self.dimensions = self.config['dimensions']
        self.train_month = self.config['train_month']
        self.oot_month = self.config['oot_month']
        self.monthly_base_month = self.config['monthly_base_month']

        # 加载数据
        self.load_data()

        # 存储结果
        self.results = {
            'initial_screening': {},
            'trn_oot_screening': {},
            'monthly_stability': {}
        }

    def load_data(self):
        """加载标签和特征数据"""
        # 读取标签数据
        if self.config['path']['merge_file_path'] == 'None':
            # 如果没有合并文件路径，则分别读取标签和特征数据
            df_label = pd.read_csv(
                self.config['path']['label_path'],
                dtype={"serial_id": 'str'}
            )
            # 读取特征数据
            df_feature = pd.read_csv(
                self.config['path']['feature_path'],
                dtype={"serial_id": 'str'}
            )
            # 合并数据
            self.df = pd.merge(df_label, df_feature, on="serial_id", how="inner")
        else:
            # 读取合并后的数据
            self.df = pd.read_csv(
                self.config['path']['merge_file_path'],
            )
        # 确保目标变量是二元分类
        self.df = self.df[self.df[self.target].isin([0, 1])]
        to_drop_list = self.config['to_drop_list']
        y_label_list = [y for y in self.config['y_label_list'] if y != self.target]
        self.df.drop(columns=to_drop_list+y_label_list, inplace=True)

        # 转换时间列
        # self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])

        print(f"数据加载完成，总样本数: {len(self.df)}")

    def preprocess_data(self, df):
        """数据预处理"""
        # 编码分类变量
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in [self.time_column, 'serial_id']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # 处理缺失值
        # for col in df.columns:
        #     if col not in [self.target, self.time_column, 'serial_id']:
        #         # 数值型用中位数填充
        #         if df[col].dtype in ['int64', 'float64']:
        #             df[col].fillna(df[col].median(), inplace=True)
        #         # 类别型用众数填充
        #         else:
        #             df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def drop_missing_features(self, df, features, missing_threshold):
        """删除缺失率超过阈值的特征"""
        missing_percent = df[features].isnull().mean()
        cols_to_keep = missing_percent[missing_percent < missing_threshold].index.tolist()
        print(f"缺失率筛选: 原始特征数 {len(features)}，保留特征数 {len(cols_to_keep)}")
        return cols_to_keep

    def calculate_iv(self, df, features):
        """计算IV值"""
        iv_df = toad.quality(df[features + [self.target]], target=self.target, iv_only=True)
        return iv_df['iv']

    def calculate_feature_importance(self, X, y):
        """计算特征重要性"""
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
        xgb.fit(X, y)
        return pd.Series(xgb.feature_importances_, index=X.columns)

    def calculate_target_corr(self, df, features):
        try:
            """计算与目标变量的相关性"""
            corr_values = df[features + [self.target]].corr()[[self.target]].sort_values(by=self.target, ascending=False)
            corr_values = corr_values[corr_values.index != self.target]
            return corr_values[self.target]
        except Exception as e:
            print(f"计算目标相关性时出错: {str(e)}")
            print(f'第一个变量为{features[0]}')

    def calculate_psi(self, df, features):
        """计算PSI值"""
        psi_values = {}
        # 划分训练集和OOT（最近一个月作为OOT）
        oot_start = sorted(df[self.time_column].unique())[-3]

        train_df = df[df[self.time_column] < oot_start]
        oot_df = df[df[self.time_column] >= oot_start]

        for feature in features:
            try:
                psi = toad.metrics.PSI(train_df[feature], oot_df[feature])
                psi_values[feature] = psi
            except:
                psi_values[feature] = np.nan

        return pd.Series(psi_values)

    def filter_high_correlation(self, df, features, iv_scores):
        """过滤高相关性变量，保留高IV值变量"""
        if len(features) > 1:
            corr_matrix = df[features].corr().abs()
            # 只保留上三角矩阵部分，减少重复计算
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = []

            for col in corr_matrix.columns:
                high_corr = corr_matrix.index[upper[col] > self.var_corr_rate].tolist()
                if high_corr:
                # 在高度相关的变量中，保留IV最高的
                    high_corr_iv = iv_scores[high_corr + [col]]
                    max_iv_feature = high_corr_iv.idxmax()
                    to_remove = [x for x in high_corr if x != max_iv_feature]
                    to_drop.extend(to_remove)

                to_drop = list(set(to_drop))
            features_to_keep = [f for f in features if f not in to_drop]
        else:
            # 如果只有一个特征，直接保留
            features_to_keep = features

        print(f"相关性筛选: 原始特征数 {len(features)}，保留特征数 {len(features_to_keep)}")
        return features_to_keep

    def initial_screening(self):
        """初筛：分模块观察整体数据情况"""
        print("\n" + "=" * 50)
        print("开始初筛阶段")
        print("=" * 50)

        # 数据预处理
        df = self.preprocess_data(self.df.copy())

        for dim_name, features_config in self.dimensions.items():
            print(f"\n分析维度: {dim_name}")
            features = features_config['变量名']
            missing_threshold = features_config.get('缺失率阈值', self.missing_rate)

            # 应用缺失率阈值
            features = self.drop_missing_features(df, features, missing_threshold)
            if not features:
                print(f"维度 {dim_name} 无满足缺失率要求的特征，跳过")
                continue

            # 计算各项指标
            iv_scores = self.calculate_iv(df, features)
            feature_importance = self.calculate_feature_importance(df[features], df[self.target])
            target_corr = self.calculate_target_corr(df, features)
            psi_scores = self.calculate_psi(df, features)

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'feature': features,
                'iv': [iv_scores.get(f, np.nan) for f in features],
                'importance': [feature_importance.get(f, np.nan) for f in features],
                'target_corr': [target_corr.get(f, np.nan) for f in features],
                'psi': [psi_scores.get(f, np.nan) for f in features]
            })

            # 应用阈值筛选
            filtered_features = results_df[
                (results_df['iv'] >= self.iv_rate) &
                (results_df['importance'] >= self.importance_rate) &
                (abs(results_df['target_corr']) >= self.target_corr_rate) &
                (results_df['psi'] <= self.psi_rate)
                ]['feature'].tolist()

            print(f"阈值筛选: 原始特征数 {len(features)}，保留特征数 {len(filtered_features)}")

            # 高相关性过滤
            # if filtered_features:
            #     filtered_features = self.filter_high_correlation(df, filtered_features, iv_scores)

            # 保存结果
            self.results['initial_screening'][dim_name] = {
                'all_features': results_df,
                'selected_features': filtered_features
            }

            # 可视化当前维度结果
            self.visualize_initial_results(dim_name, results_df, filtered_features)

        # 保存初筛结果
        self.save_initial_results()
        print("\n初筛阶段完成!")

    def visualize_initial_results(self, dim_name, results_df, selected_features):
        """可视化初筛结果"""
        plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体
        plt.figure(figsize=(18, 12))
        plt.suptitle(f"维度分析: {dim_name}", fontsize=16)

        # IV值排序
        plt.subplot(2, 2, 1)
        iv_df = results_df.sort_values('iv', ascending=False).head(20)
        sns.barplot(x='iv', y='feature', data=iv_df)
        plt.title('Top 20 Features by IV')
        # 添加阈值线
        plt.axvline(x=self.iv_rate, color='r', linestyle='--')

        # 特征重要性排序
        plt.subplot(2, 2, 2)
        imp_df = results_df.sort_values('importance', ascending=False).head(20)
        sns.barplot(x='importance', y='feature', data=imp_df)
        plt.title('Top 20 Features by Importance')
        plt.axvline(x=self.importance_rate, color='r', linestyle='--')

        # 目标变量相关性排序
        plt.subplot(2, 2, 3)
        corr_df = results_df.sort_values('target_corr', ascending=False).head(20)
        sns.barplot(x='target_corr', y='feature', data=corr_df)
        plt.title('Top 20 Features by Target Correlation')
        plt.axvline(x=self.target_corr_rate, color='r', linestyle='--')

        # PSI值排序
        plt.subplot(2, 2, 4)
        psi_df = results_df.sort_values('psi', ascending=False).head(20)
        sns.barplot(x='psi', y='feature', data=psi_df)
        plt.title('Top 20 Features by PSI')
        plt.axvline(x=self.psi_rate, color='r', linestyle='--')

        plt.tight_layout()
        plt.savefig(f"{self.config['result_path']}/init/fig/{dim_name}_initial.png")
        plt.close()

        # 保存筛选后的特征
        # if selected_features:
        #     selected_df = results_df[results_df['feature'].isin(selected_features)]
        #     selected_df.to_csv(f"{self.config['result_path']}/init/{dim_name}_selected.csv", index=False)

    def save_initial_results(self):
        """保存初筛结果"""
        all_results = []
        for dim_name, data in self.results['initial_screening'].items():
            df = data['all_features'].copy()
            df['dimension'] = dim_name
            df['selected'] = df['feature'].isin(data['selected_features'])
            all_results.append(df)

        if all_results:
            combined_df = pd.concat(all_results)
            combined_df.to_csv(f"{self.config['result_path']}/init/initial_screening_results.csv", index=False)

            # 保存所有筛选后的特征
            selected_features = []
            for data in self.results['initial_screening'].values():
                selected_features.extend(data['selected_features'])

            pd.DataFrame({'feature': selected_features}).to_csv(
                f"{self.config['result_path']}/init/selected_features_initial.csv", index=False
            )

    def trn_oot_screening(self):
        """次筛：TRN v.s OOT"""
        print("\n" + "=" * 50)
        print("开始次筛阶段 (TRN vs OOT)")
        print("=" * 50)

        # 获取初筛选出的特征
        selected_features = []
        for data in self.results['initial_screening'].values():
            selected_features.extend(data['selected_features'])

        if not selected_features:
            print("初筛未选择任何特征，跳过次筛阶段")
            return

        # 数据预处理
        df = self.preprocess_data(self.df.copy())

        # 划分TRN和OOT（最近三个月作为OOT）
        # oot_start = sorted(df[self.time_column].unique())[-3]

        train_df = df[df[self.time_column].isin(self.train_month)]
        oot_df = df[df[self.time_column].isin(self.oot_month)]

        print(f"TRN样本数: {len(train_df)}, OOT样本数: {len(oot_df)}")

        # 对每个特征进行分析
        results = []
        for feature in selected_features:
            feature_results = self.analyze_feature_trn_oot(feature, train_df, oot_df)
            if feature_results:
                results.append(feature_results)

        # 保存结果
        if results:
            results_df = pd.DataFrame(results)
            processed_data = pd.DataFrame(None)
            feature_stats_list  = results
            for feature_dict in feature_stats_list:
                feature_name = feature_dict['feature']
                monotonic_train = feature_dict['monotonic_train']
                monotonic_oot = feature_dict['monotonic_oot']
                distribution_diff = feature_dict['distribution_diff']
                # 处理训练集统计数据
                train_stat = feature_dict['train_stats']
                train_stat['feature'] = feature_name
                train_stat['data_type'] = 'train'
                train_stat['monotonic_train'] = monotonic_train
                train_stat['distribution_diff'] = distribution_diff
                train_stat = train_stat.reindex(columns=['feature', 'data_type','bin', 'total', 'bad_count',
                                                         'bad_rate','good_count', 'good_rate', 'perc', 'lift',
                                                         'monotonic_train', 'distribution_diff'])
                # 处理OOT（跨时间）统计数据
                oot_stat = feature_dict['oot_stats']
                oot_stat['data_type'] = 'oot'
                oot_stat['monotonic_oot'] = monotonic_oot
                oot_stat = oot_stat.reindex(columns=['data_type', 'bin', 'total', 'bad_count',
                                                     'bad_rate', 'good_count', 'good_rate', 'perc', 'lift',
                                                     'monotonic_oot'])

                trn_oot_stat = pd.concat([train_stat,oot_stat], axis = 1)
                processed_data= pd.concat([processed_data, trn_oot_stat], axis=0)


            processed_data.to_csv(f"{self.config['result_path']}/trn_oot/trn_oot_results.csv", index=False)

            self.results['trn_oot_screening'] = results_df

            # 筛选满足条件的特征
            passed_features = results_df[
                (results_df['monotonic_train'] == 'Yes') &
                (results_df['monotonic_oot'] == 'Yes') &
                (results_df['distribution_diff'] <= self.distribution_std)
                ]['feature'].tolist()

            pd.DataFrame({'feature': passed_features}).to_csv(
                f"{self.config['result_path']}/trn_oot/selected_features_trn_oot.csv", index=False
            )
            self.results['second_passed_features'] = passed_features
            print(f"次筛阶段完成! 原始特征数 {len(selected_features)}，保留特征数 {len(passed_features)}")
        else:
            print("次筛阶段未产生有效结果")

    def analyze_feature_trn_oot(self, feature, train_df, oot_df):
        """分析特征在TRN和OOT上的表现（分箱策略根据模型类型自适应）"""
        try:
            # ==================== 分箱策略选择 ====================
            if self.if_scorecard:
                # 评分卡模型：使用WOE分箱（toad库）
                combiner = toad.transform.Combiner()
                combiner.fit(train_df[[feature, self.target]],
                             train_df[self.target],
                             method='chi',  # 卡方分箱
                             min_samples=0.15)  # 每箱最小样本占比
                bins = combiner.export()

                # 应用分箱
                train_df['bin'] = combiner.transform(train_df[[feature]])
                oot_df['bin'] = combiner.transform(oot_df[[feature]])
            else:
                # 非评分卡模型：使用OptimalBinning（确保单调性）
                # optb = OptimalBinning(
                #     name=feature,
                #     dtype="numerical" if train_df[feature].dtype in ['float64', 'int64'] else "categorical",
                #     monotonic_trend="auto", # 自动判断单调方向
                #     max_n_bins= 5,
                #     solver="cp"  # 使用整数规划保证单调性
                # )
                # optb.fit(train_df[feature].values, train_df[self.target].values)
                #
                # # 应用分箱
                # train_df['bin'] = optb.transform(train_df[feature].values, metric="bins")
                # oot_df['bin'] = optb.transform(oot_df[feature].values, metric="bins")
                # 获取分箱边界
                # bins = optb.splits  # 数值型返回分割点，类别型返回分组

                # 直接使用等频分箱（10分箱）
                # 等频分箱（训练集）
                train_df['bin'], bins = pd.qcut(
                    train_df[feature],
                    q=10,
                    labels=False,
                    retbins=True,
                    duplicates='drop'  # 避免重复边界报错
                )

                # 应用相同的分箱边界到测试集
                oot_df['bin'] = pd.cut(
                    oot_df[feature],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )



            # 分箱统计计算
            train_stats = self.calculate_bin_stats(train_df, feature)
            oot_stats = self.calculate_bin_stats(oot_df, feature)
            # 剔除掉Missing分箱，避免后续报错
            oot_stats = [item for item in oot_stats if item['bin'] != 'Missing']
            train_stats = [item for item in train_stats if item['bin'] != 'Missing']
            # 单调性验证
            monotonic_train = self.check_monotonicity(train_stats)
            monotonic_oot = self.check_monotonicity(oot_stats)

            # 评分卡专用处理
            woe_train = woe_oot = None
            if self.if_scorecard:
                woe_train = self.calculate_woe(train_stats)
                woe_oot = self.calculate_woe(oot_stats)

            # 可视化
            # self.visualize_bin_results(feature, train_stats, oot_stats)

            return {
                'feature': feature,
                'monotonic_train': 'Yes' if monotonic_train else 'No',
                'monotonic_oot': 'Yes' if monotonic_oot else 'No',
                'distribution_diff': self.calculate_distribution_diff(train_stats, oot_stats),
                'train_stats': pd.DataFrame(train_stats),
                'oot_stats': pd.DataFrame(oot_stats),
                'woe_train': woe_train,
                'woe_oot': woe_oot,
            }

        except Exception as e:
            print(f"特征 {feature} 分析失败: {str(e)}")
            return None

    def calculate_bin_stats(self, df, feature):
        """计算分箱统计信息"""
        stats = df.groupby('bin').agg(
            total=('bin', 'count'),
            bad_count=(self.target, 'sum'),
            bad_rate=(self.target, 'mean')
        ).reset_index()

        stats['good_count'] = stats['total'] - stats['bad_count']
        stats['good_rate'] = 1 - stats['bad_rate']
        stats['perc'] = stats['total'] / stats['total'].sum()

        # 计算Lift
        overall_bad_rate = df[self.target].mean()
        stats['lift'] = stats['bad_rate'] / overall_bad_rate

        return stats.to_dict('records')

    def check_monotonicity(self, stats):
        """检查Lift单调性"""
        lift = [bin['lift'] for bin in stats]
        n = len(lift)

        # 检查单调递增
        increasing = all(lift[i] <= lift[i + 1] for i in range(n - 1))

        # 检查单调递减
        decreasing = all(lift[i] >= lift[i + 1] for i in range(n - 1))

        return increasing or decreasing

    def calculate_distribution_diff(self, train_stats, oot_stats):
        """计算分布差异"""
        train_perc = np.array([bin['perc'] for bin in train_stats])
        oot_perc = np.array([bin['perc'] for bin in oot_stats])
        return np.abs(train_perc - oot_perc).mean()

    def calculate_woe(self, stats):
        """计算WOE值"""
        for bin in stats:
            bin['woe'] = np.log((bin['good_count'] / bin['good_count'].sum()) /
                                (bin['bad_count'] / bin['bad_count'].sum()))
        return stats

    def visualize_bin_results(self, feature, train_stats, oot_stats):
        """使用Matplotlib静默保存分箱结果图（不显示）"""
        # 设置Agg后端（不显示图形）
        matplotlib.use('Agg')  # 必须在其他matplotlib导入前设置

        # 统一分箱标签（处理可能的Missing）
        def get_common_bins(stats):
            bins = set()
            for bin_data in stats:
                bin_name = 'Missing' if pd.isna(bin_data['bin']) else str(bin_data['bin'])
                bins.add(bin_name)
            return sorted(bins, key=lambda x: (x == 'Missing', x))  # Missing放最后

        # 获取统一的bin顺序
        common_bins = get_common_bins(train_stats + oot_stats)

        # 构建统一的数据结构（填充可能缺失的分箱）
        def build_consistent_data(stats, common_bins):
            data = {'perc': [], 'bad_rate': []}
            bin_map = {'Missing' if pd.isna(b['bin']) else str(b['bin']): b for b in stats}

            for bin_name in common_bins:
                if bin_name in bin_map:
                    data['perc'].append(bin_map[bin_name]['perc'])
                    data['bad_rate'].append(bin_map[bin_name]['bad_rate'])
                else:
                    data['perc'].append(0)  # 缺失分箱补零
                    data['bad_rate'].append(np.nan)  # 坏账率标记为nan
            return data

        train_data = build_consistent_data(train_stats, common_bins)
        oot_data = build_consistent_data(oot_stats, common_bins)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"特征分箱分析: {feature}", fontsize=14)

        # ========== TRN分箱结果 ==========
        # 分箱占比柱状图
        bars1 = ax1.bar(common_bins, train_data['perc'],
                        color='royalblue', alpha=0.6, label='分箱占比')
        ax1.set_ylabel('分箱占比', fontsize=10)
        ax1.set_title("TRN分箱统计", pad=10)

        # 坏账率折线图（次坐标）
        ax1_sec = ax1.twinx()
        line1 = ax1_sec.plot(common_bins, train_data['bad_rate'],
                             'r-o', linewidth=2, markersize=6, label='坏账率')
        ax1_sec.set_ylabel('坏账率', fontsize=10)

        # 处理NaN显示（虚线连接）
        if any(np.isnan(train_data['bad_rate'])):
            ax1_sec.plot(common_bins, train_data['bad_rate'], 'r--', alpha=0.3)  # 虚线辅助线

        # ========== OOT分箱结果 ==========
        # 分箱占比柱状图
        bars2 = ax2.bar(common_bins, oot_data['perc'],
                        color='forestgreen', alpha=0.6, label='分箱占比')
        ax2.set_ylabel('分箱占比', fontsize=10)
        ax2.set_title("OOT分箱统计", pad=10)

        # 坏账率折线图（次坐标）
        ax2_sec = ax2.twinx()
        line2 = ax2_sec.plot(common_bins, oot_data['bad_rate'],
                             'r-o', linewidth=2, markersize=6, label='坏账率')
        ax2_sec.set_ylabel('坏账率', fontsize=10)

        # 处理NaN显示
        if any(np.isnan(oot_data['bad_rate'])):
            ax2_sec.plot(common_bins, oot_data['bad_rate'], 'r--', alpha=0.3)

        # 统一调整图形
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)  # x轴标签旋转

        plt.tight_layout()

        # 保存图片
        output_path = f"{self.config['result_path']}/trn_oot/fig/{feature}_bin_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def monthly_stability(self):
        """终筛：by 月观察每个变量的稳定性"""
        print("\n" + "=" * 50)
        print("开始终筛阶段 (月度稳定性分析)")
        print("=" * 50)

        # 获取次筛选出的特征
        if self.results['second_passed_features']:
            selected_features = self.results['second_passed_features']
        else:
            print("无可用特征，跳过终筛阶段")
            return

        # 数据预处理
        df = self.preprocess_data(self.df.copy())

        # 提取年份和月份
        # df['year_month'] = df[self.time_column].dt.to_period('M')
        df['year_month'] = df[self.time_column]
        # 对每个特征进行分析
        results = []
        for feature in selected_features:
            feature_results = self.analyze_feature_monthly(feature, df)
            if feature_results:
                results.append(feature_results)

        # 保存结果
        if results:
            monthly_iv_df = pd.DataFrame(None)
            monthly_psi_df = pd.DataFrame(None)
            monthly_mean_df = pd.DataFrame(None)
            results_df = pd.DataFrame(results)
            for i in range(len(results_df)):
                feature = results_df.iloc[i]['feature']
                # 拼接iv
                tmp_iv_df = pd.DataFrame(pd.DataFrame(results_df.iloc[i]['monthly_stats']).set_index('year_month')['iv']).rename(
                    columns={'iv': feature}).T
                monthly_iv_df = pd.concat([monthly_iv_df, tmp_iv_df], axis=0)
                # 拼接psi
                tmp_psi_df = pd.DataFrame(
                    pd.DataFrame(results_df.iloc[i]['monthly_stats']).set_index('year_month')['psi']).rename(
                    columns={'psi':feature}).T
                monthly_psi_df = pd.concat([monthly_psi_df, tmp_psi_df], axis=0)
                # 拼接mean
                tmp_mean_df = pd.DataFrame(
                    pd.DataFrame(results_df.iloc[i]['monthly_stats']).set_index('year_month')['mean']).rename(
                    columns={'mean': feature}).T
                monthly_mean_df = pd.concat([monthly_mean_df, tmp_mean_df], axis=0)
            with pd.ExcelWriter(f"{self.config['result_path']}/monthly/monthly_stability_results.xlsx", engine="openpyxl", mode="w") as writer:
                monthly_iv_df.to_excel(writer, sheet_name="monthly_iv", index=True)
                monthly_psi_df.to_excel(writer, sheet_name="monthly_psi", index=True)
                monthly_mean_df.to_excel(writer, sheet_name="monthly_mean", index=True)

            self.results['monthly_stability'] = results_df

            # 筛选满足条件的特征
            passed_features = results_df[
                (results_df['iv_std'] <= self.iv_month_std) &
                (results_df['psi_std'] <= self.psi_month_std) &
                (results_df['mean_std'] <= self.mean_month_std)
                ]['feature'].tolist()

            pd.DataFrame({'feature': passed_features}).to_csv(
                f"{self.config['result_path']}/monthly/final_selected_features.csv", index=False
            )

            print(f"终筛阶段完成! 原始特征数 {len(selected_features)}，保留特征数 {len(passed_features)}")
        else:
            print("终筛阶段未产生有效结果")

    def analyze_feature_monthly(self, feature, df):
        """分析单个特征的月度稳定性"""
        try:
            monthly_stats = []
            months = sorted(df['year_month'].unique())

            # 基准分布（指定月份）
            base_df = df[df['year_month'].isin(self.monthly_base_month)]

            for month in months:
                month_df = df[df['year_month'] == month]
                if len(month_df) < 100:  # 样本量太小则跳过
                    continue

                # 计算IV值
                iv = toad.quality(month_df[[feature, self.target]], target=self.target, iv_only=True)['iv'].get(feature,np.nan)

                # 计算PSI（与基准分布比较）
                psi = toad.metrics.PSI(base_df[feature], month_df[feature])

                # 计算均值
                mean_val = month_df[feature].mean()

                monthly_stats.append({
                    'year_month': str(month),
                    'iv': iv,
                    'psi': psi,
                    'mean': mean_val
                })

            if len(monthly_stats) < 3:  # 至少需要3个月的数据
                return None

            # 转换为DataFrame
            stats_df = pd.DataFrame(monthly_stats)

            # 计算标准差
            iv_std = stats_df['iv'].std()
            psi_std = stats_df['psi'].std()
            mean_std = stats_df['mean'].std()

            # 可视化月度趋势
            self.visualize_monthly_trends(feature, stats_df)

            return {
                'feature': feature,
                'months': len(stats_df),
                'iv_std': iv_std,
                'psi_std': psi_std,
                'mean_std': mean_std,
                'monthly_stats': monthly_stats
            }

        except Exception as e:
            print(f"分析特征 {feature} 月度稳定性时出错: {str(e)}")
            return None

    def visualize_monthly_trends(self, feature, stats_df):
        """可视化月度趋势"""
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=('IV值趋势', 'PSI值趋势', '均值趋势'),
                            shared_xaxes=True,
                            vertical_spacing=0.1)

        # IV值趋势
        fig.add_trace(go.Scatter(
            x=stats_df['year_month'], y=stats_df['iv'],
            mode='lines+markers', name='IV值',
            line=dict(color='royalblue')
        ), row=1, col=1)

        # PSI值趋势
        fig.add_trace(go.Scatter(
            x=stats_df['year_month'], y=stats_df['psi'],
            mode='lines+markers', name='PSI值',
            line=dict(color='firebrick')
        ), row=2, col=1)

        # 均值趋势
        fig.add_trace(go.Scatter(
            x=stats_df['year_month'], y=stats_df['mean'],
            mode='lines+markers', name='均值',
            line=dict(color='forestgreen')
        ), row=3, col=1)

        # 添加阈值线
        fig.add_hline(y=self.iv_rate, line_dash="dot",
                      annotation_text=f"IV阈值: {self.iv_rate}",
                      row=1, col=1, line_color="red")

        fig.add_hline(y=self.psi_rate, line_dash="dot",
                      annotation_text=f"PSI阈值: {self.psi_rate}",
                      row=2, col=1, line_color="red")

        # 更新布局
        fig.update_layout(
            title=f"月度稳定性分析: {feature}",
            height=800,
            showlegend=False,
            xaxis3=dict(title='月份'),
            yaxis=dict(title='IV值'),
            yaxis2=dict(title='PSI值'),
            yaxis3=dict(title='均值')
        )

        fig.write_html(f"{self.config['result_path']}/monthly/fig/{feature}_monthly_stability.html")

    def run(self):
        """执行完整的特征筛选流程"""
        self.initial_screening()
        self.trn_oot_screening()
        self.monthly_stability()
        print("\n特征筛选流程完成!")


if __name__ == "__main__":
    # 初始化特征筛选器
    screener = AutoFeatureScreener("./config/config.json")

    # 执行完整流程
    screener.run()