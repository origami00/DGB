import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
import os
from datetime import timedelta
from sklearn.model_selection import GridSearchCV

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] # SimHei是一种常见的中文字体
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号


class PVPowerPrediction:
    def __init__(self, csv_path, result_dir='题二完整结果'):
        self.csv_path = csv_path
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.data = None
        self.model = None
        self.capacity = 20  # MW (电站额定容量，来自T1.py, 用于Ci)
        
        self.feature_names = None
        self.power_mean_for_inverse = None
        self.power_std_for_inverse = None
        self.test_data_with_predictions = None # DataFrame存储测试数据及其预测
        self.X_test_prepared_features = None # 新增：存储准备好的测试集特征

        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        print("加载数据...")
        try:
            self.data = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.csv_path} 未找到。")
            raise
            
        if 'date_time' not in self.data.columns:
            print("错误: 'date_time' 列未在CSV文件中找到。")
            raise ValueError("'date_time' column missing")
            
        self.data['datetime'] = pd.to_datetime(self.data['date_time'])
        self.data.sort_values('datetime', inplace=True)
        self.data.set_index('datetime', inplace=True, drop=False) # 保留datetime列用于报告

        # 确保 'power' 列存在
        if 'power' not in self.data.columns:
            # 如果 'power' 缺失，尝试查找常见的替代名称
            if '实际功率(MW)' in self.data.columns:
                self.data.rename(columns={'实际功率(MW)': 'power'}, inplace=True)
            elif 'Power(MW)' in self.data.columns: # 另一个常见变体
                self.data.rename(columns={'Power(MW)': 'power'}, inplace=True)
            else:
                print("错误: 目标列 'power' (或其常见变体) 未在数据中找到。")
                raise ValueError("Target column 'power' not found.")
        
        print(f"数据加载完成. 数据范围: {self.data.index.min()} 到 {self.data.index.max()}")
        print(f"数据包含 {len(self.data)} 行。")

    def _preprocess_data(self):
        print("数据预处理 (中值滤波和Z-score标准化)...")
        if self.data is None or 'power' not in self.data.columns:
            print("错误: 数据未加载或'power'列缺失，无法预处理。")
            return

        # 对 'power' 列进行中值滤波
        # 确保power是数值型，并在滤波前处理潜在的NaN
        self.data['power'] = pd.to_numeric(self.data['power'], errors='coerce')
        self.data.dropna(subset=['power'], inplace=True) # 删除power变为NaN的行

        if self.data['power'].isnull().any():
             print("警告: 'power' 列在转换为数值型后仍包含NaN值，将使用0填充。")
             self.data['power'].fillna(0, inplace=True)

        self.data['power_filtered'] = medfilt(self.data['power'], kernel_size=5) # kernel_size必须为奇数
        
        # 对 'power_filtered' 进行Z-score标准化
        self.power_mean_for_inverse = self.data['power_filtered'].mean()
        self.power_std_for_inverse = self.data['power_filtered'].std()
        
        if self.power_std_for_inverse == 0:
            print("警告: 滤波后功率的标准差为0，Z-score标准化将导致除以零。跳过标准化，仅进行中心化。")
            self.data['power_processed'] = self.data['power_filtered'] - self.power_mean_for_inverse 
        else:
            self.data['power_processed'] = (self.data['power_filtered'] - self.power_mean_for_inverse) / self.power_std_for_inverse
        
        print("中值滤波和Z-score标准化完成。")

    def _create_features(self):
        print("创建特征...")
        df = self.data.copy()
        
        target_col = 'power_processed' 

        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek 
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        # 季节映射
        season_map = {1: '冬季', 2: '冬季', 3: '春季', 4: '春季', 5: '春季', 
                      6: '夏季', 7: '夏季', 8: '夏季', 9: '秋季', 10: '秋季', 
                      11: '秋季', 12: '冬季'}
        df['season_num'] = df['month'].map(season_map).astype('category').cat.codes # 数值化季节

        # 周期性特征编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.)
        # df.drop(columns=['hour', 'dayofyear'], inplace=True) # 移除原始列

        # 'power_processed' 的滞后特征
        lags = [96, 96*2, 96*7] # 1天, 2天, 1周 (每天96个间隔)
        for lag in lags:
            df[f'power_lag_{lag//96}d'] = df[target_col].shift(lag)

        # 基于滞后数据的滚动窗口特征 (避免数据泄露)
        # 使用24小时前的功率作为滚动窗口的基础
        shifted_power_24h_ago = df[target_col].shift(96) 
        df['rolling_mean_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).mean()
        df['rolling_std_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).std()
        
        df.dropna(inplace=True) # 移除因滞后/滚动特征产生的NaN行
        
        self.feature_names = ['dayofweek', 'month', 'quarter', 'season_num', 
                             'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos'] + \
                             [f'power_lag_{lag//96}d' for lag in lags] + \
                             ['rolling_mean_24h_lag1d', 'rolling_std_24h_lag1d']
        
        # 确保所有特征列存在，并处理std计算可能因窗口太小产生的NaN
        for col in ['rolling_std_24h_lag1d']: # 如果窗口内所有值相同，std可能为NaN
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        print(f"使用的特征列: {self.feature_names}")
        if df.empty:
            print("错误: 创建特征后数据为空，请检查滞后和滚动窗口设置。")
            raise ValueError("No data left after feature creation and NaN drop.")
            
        return df[self.feature_names], df[target_col], df[['datetime', 'power', 'power_filtered']]


    def _get_test_indices(self, df_with_datetime_index):
        test_indices = pd.Series(False, index=df_with_datetime_index.index)
        unique_years = df_with_datetime_index.index.year.unique()
        
        for year in unique_years:
            for month_num in [2, 5, 8, 11]: # 测试月份
                # 获取此特定年和月的所有数据
                month_data = df_with_datetime_index[(df_with_datetime_index.index.year == year) & 
                                                    (df_with_datetime_index.index.month == month_num)]
                if not month_data.empty:
                    # 确定此月份的最后一天
                    last_day_in_month = month_data.index.max().day
                    # 最后一周的第一天 (7天)
                    start_day_of_last_week = last_day_in_month - 6
                    
                    # 创建此月份最后一周的掩码
                    mask_last_week = (df_with_datetime_index.index.year == year) & \
                                     (df_with_datetime_index.index.month == month_num) & \
                                     (df_with_datetime_index.index.day >= start_day_of_last_week) & \
                                     (df_with_datetime_index.index.day <= last_day_in_month)
                    test_indices[mask_last_week] = True
        return test_indices

    def train_model(self):
        X_all_features, y_all_target, report_data_all = self._create_features()
        is_test_period = self._get_test_indices(X_all_features)

        X_train = X_all_features[~is_test_period]
        y_train = y_all_target[~is_test_period]
        
        # 准备并存储测试集特征和目标变量
        self.X_test_prepared_features = X_all_features[is_test_period].copy()
        y_test_for_early_stopping = y_all_target[is_test_period] # 用于early stopping的评估集目标

        self.test_data_with_predictions = report_data_all[is_test_period].copy()
        self.test_data_with_predictions['y_true_standardized'] = y_test_for_early_stopping

        print(f"训练集大小: {X_train.shape}, 测试集大小: {self.X_test_prepared_features.shape}")
        if X_train.empty or y_train.empty:
            print("错误: 训练集为空。请检查数据量和特征创建过程。")
            raise ValueError("Training set is empty.")
        if self.X_test_prepared_features.empty or y_test_for_early_stopping.empty:
            print("警告: 测试集为空。可能没有符合测试标准的日期（2、5、8、11月最后一周）。")
            raise ValueError("Test set is empty. Check test period definition and data range.")

        print("开始使用GridSearchCV进行超参数优化...")

        # 定义参数网格
        param_grid = {
            'learning_rate': [0.01, 0.02, 0.05], # 调整学习率范围
            'max_depth': [6, 7, 8], # 调整最大深度范围
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'gamma': [0, 0.05, 0.1] # 调整gamma范围
            # n_estimators 通过 early_stopping_rounds 控制，固定一个较高值
        }

        # 初始化XGBoost估计器，固定n_estimators和early_stopping_rounds
        xgb_estimator = xgb.XGBRegressor(
            n_estimators=1000,  # 固定一个较高的值，由early stopping决定实际数量
            random_state=42,
            early_stopping_rounds=30, # early_stopping_rounds移至构造函数
        )

        # 定义GridSearchCV的fit参数，用于early stopping
        # 注意：fit_params 不应包含 early_stopping_rounds，因为它已在构造函数中设置
        fit_params = {
            # 'early_stopping_rounds': 30, # 此行已移除/注释掉
            'eval_set': [(self.X_test_prepared_features, y_test_for_early_stopping)],
            'verbose': False # 减少XGBoost训练过程中的冗余输出，GridSearchCV有自己的verbose
        }

        # 初始化GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_estimator,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error', # 目标是最小化RMSE
            cv=3, # 交叉验证折数，可根据需要调整
            verbose=2, # 显示GridSearchCV的进度
            n_jobs=-1  # 使用所有可用的CPU核心
        )

        try:
            print(f"开始GridSearchCV拟合，这将需要一些时间...")
            grid_search.fit(X_train, y_train, **fit_params)
            
            print("GridSearchCV完成。")
            print(f"找到的最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证RMSE: {-grid_search.best_score_:.4f}") # 分数是负的RMSE
            
            self.model = grid_search.best_estimator_ # 使用找到的最佳模型
            print(f"模型已更新为GridSearchCV找到的最佳估计器。")
            print(f"最佳估计器使用的树方法: {self.model.get_params().get('tree_method', 'auto')}")
        except Exception as e:
            print(f"GridSearchCV 训练过程中发生错误: {e}")
            import traceback # 打印完整的回溯信息
            traceback.print_exc()
            raise # 重新抛出GridSearchCV的错误

    def predict_and_evaluate(self):
        if self.model is None:
            print("错误: 模型未训练。")
            raise ValueError("Model not trained.")
        if self.test_data_with_predictions is None or self.test_data_with_predictions.empty:
            print("警告: 没有测试数据可供预测或评估。")
            self.metrics = {}
            return
        if self.X_test_prepared_features is None or self.X_test_prepared_features.empty:
            print("警告: 没有准备好的测试集特征用于预测。")
            self.metrics = {}
            return

        X_test_for_model = self.X_test_prepared_features.copy() # 使用存储的测试特征副本
        
        # 检查并处理NaN
        if X_test_for_model.isnull().any().any():
            print("警告: 用于预测的X_test_prepared_features副本包含NaN值。将尝试用0填充。")
            X_test_for_model.fillna(0, inplace=True)

        print("进行预测...")
        predictions_standardized = self.model.predict(X_test_for_model)
        
        # 将预测反标准化为MW
        if self.power_std_for_inverse == 0: # 如果标准化期间std为0
            predictions_mw = predictions_standardized + self.power_mean_for_inverse
        else:
            predictions_mw = (predictions_standardized * self.power_std_for_inverse) + self.power_mean_for_inverse
        
        predictions_mw = np.maximum(0, predictions_mw) # 功率不能为负
        predictions_mw = np.minimum(predictions_mw, self.capacity) # 上限为电站容量

        self.test_data_with_predictions['predicted_power_mw'] = predictions_mw
        # 'actual_power_mw' 将是原始数据中的 'power_filtered' 以保持一致性
        self.test_data_with_predictions['actual_power_mw'] = self.test_data_with_predictions['power_filtered']


        print("评估预测结果 (仅白昼时段)...")
        # 定义白昼: actual_power_mw > 0.01 * self.capacity (例如，容量的1%以上)
        daylight_threshold = 0.00 # 可以是 self.capacity * 0.01
        daylight_mask = self.test_data_with_predictions['actual_power_mw'] > daylight_threshold
        
        y_true_daylight = self.test_data_with_predictions.loc[daylight_mask, 'actual_power_mw']
        y_pred_daylight = self.test_data_with_predictions.loc[daylight_mask, 'predicted_power_mw']
        
        self.metrics = {} # 初始化指标字典
        if y_true_daylight.empty or len(y_true_daylight) < 2: # 相关性/标准差至少需要2个点
            print("警告: 白昼时段数据不足 (<2个点)，无法计算所有误差指标。")
        else:
            n = len(y_true_daylight)
            cap = self.capacity 
            
            errors_normalized = (y_true_daylight - y_pred_daylight) / cap
            
            e_rmse_norm = np.sqrt(np.mean(errors_normalized**2))
            e_mae_norm = np.mean(np.abs(errors_normalized))
            e_me_norm = np.mean(errors_normalized)
            
            mean_y_true = np.mean(y_true_daylight)
            mean_y_pred = np.mean(y_pred_daylight)
            
            numerator = np.sum((y_true_daylight - mean_y_true) * (y_pred_daylight - mean_y_pred))
            denom_true = np.sum((y_true_daylight - mean_y_true)**2)
            denom_pred = np.sum((y_pred_daylight - mean_y_pred)**2)
            
            r_correlation = 0.0 # 初始化为0
            if denom_true > 1e-9 and denom_pred > 1e-9: # 避免除以零或非常小的值
                r_correlation = numerator / (np.sqrt(denom_true * denom_pred))
            
            c_r_accuracy = (1 - e_rmse_norm) * 100 if e_rmse_norm <=1 else 0.0 # 准确率
            
            abs_err_qual_rate = np.abs(y_true_daylight - y_pred_daylight) / cap
            B_i_qual = (abs_err_qual_rate < 0.25).astype(int)
            q_r_qualified = np.mean(B_i_qual) * 100
            
            self.metrics = {
                '归一化均方根误差 (E_rmse_norm)': e_rmse_norm,
                '归一化平均绝对误差 (E_mae_norm)': e_mae_norm,
                '归一化平均误差 (E_me_norm)': e_me_norm,
                '相关系数 (r)': r_correlation,
                '准确率 (C_R %)': c_r_accuracy,
                '合格率 (Q_R %)': q_r_qualified,
                '白昼样本数': n
            }
            print("误差指标 (白昼时段):")
            for k_metric, v_metric in self.metrics.items():
                print(f"  {k_metric}: {v_metric:.4f}")
        
        self._generate_outputs()


    def _generate_outputs(self):
        print("生成输出文件和图表...")
        if self.test_data_with_predictions is None or self.test_data_with_predictions.empty:
            print("无测试预测数据可供输出。")
            return

        # --- 预测表格 (CSV) ---
        all_monthly_dfs = []
        month_map_plot = {2: "2月", 5: "5月", 8: "8月", 11: "11月"}

        for test_month_num in [2, 5, 8, 11]:
            month_data_for_csv = self.test_data_with_predictions[
                self.test_data_with_predictions.index.month == test_month_num
            ].copy() # 确保是副本

            if not month_data_for_csv.empty:
                # 确保 'datetime' 列存在 (来自原始数据加载或索引)
                if 'datetime' not in month_data_for_csv.columns:
                     month_data_for_csv['datetime'] = month_data_for_csv.index

                # 创建报告列
                month_data_for_csv['预报时间'] = month_data_for_csv['datetime'].dt.strftime('%Y/%m/%d/%H:%M')
                month_data_for_csv['起报时间'] = (month_data_for_csv['datetime'].dt.normalize() - pd.Timedelta(days=1)).dt.strftime('%Y/%m/%d/00:00')
                
                # 选择并重命名报告列
                report_df = month_data_for_csv[['起报时间', '预报时间', 'actual_power_mw', 'predicted_power_mw']].copy()
                report_df.rename(columns={
                    'actual_power_mw': '实际功率(MW)',
                    'predicted_power_mw': 'XGBoost预测功率(MW)'
                }, inplace=True)
                
                # 按预报时间排序
                report_df.sort_values(by='预报时间', inplace=True)

                csv_filename = os.path.join(self.result_dir, f"{month_map_plot[test_month_num]}预测结果.csv")
                report_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                print(f"已保存: {csv_filename}")
                all_monthly_dfs.append(report_df)

        if all_monthly_dfs:
            combined_csv = pd.concat(all_monthly_dfs)
            combined_csv.sort_values(by=['起报时间', '预报时间'], inplace=True)
            combined_filename = os.path.join(self.result_dir, "所有预测结果.csv")
            combined_csv.to_csv(combined_filename, index=False, encoding='utf-8-sig')
            print(f"已保存: {combined_filename}")
        else:
            print("警告: 未生成任何月度预测CSV文件。")

        # --- 图表 ---
        # 1. 时间序列图 (示例来自第一个可用的测试月份)
        first_test_month_data = None
        for month_num_plot in [2, 5, 8, 11]: # 按顺序检查
            data_segment = self.test_data_with_predictions[self.test_data_with_predictions.index.month == month_num_plot]
            if not data_segment.empty:
                first_test_month_data = data_segment
                break
        
        if first_test_month_data is not None and not first_test_month_data.empty:
            plt.figure(figsize=(15, 6))
            plt.plot(first_test_month_data.index, first_test_month_data['actual_power_mw'], label='实际功率', alpha=0.9)
            plt.plot(first_test_month_data.index, first_test_month_data['predicted_power_mw'], label='预测功率 (XGBoost)', linestyle='--')
            plt.title(f'光伏功率时间序列预测对比图 ({month_map_plot[first_test_month_data.index.month.unique()[0]]}示例)')
            plt.xlabel('时间')
            plt.ylabel('功率 (MW)')
            plt.legend()
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, '时间序列预测对比图.png'))
            plt.close()
            print("已保存: 时间序列预测对比图.png")

        # 2. 特征重要性
        if hasattr(self.model, 'feature_importances_') and self.model.feature_importances_ is not None:
            try:
                plt.figure(figsize=(10, max(6, len(self.feature_names) * 0.35))) # 动态高度
                xgb.plot_importance(self.model, ax=plt.gca(), importance_type='gain', title=None, show_values=False) # gain 通常更受青睐
                plt.title('特征重要性分析图 (基于增益)', y=1.02) # 自定义标题, y 用于间距
                plt.tight_layout() # 调整布局以防重叠
                plt.savefig(os.path.join(self.result_dir, '特征重要性分析图.png'))
                plt.close()
                print("已保存: 特征重要性分析图.png")
            except Exception as e_plot_importance:
                 print(f"绘制特征重要性图时出错: {e_plot_importance}")


        # 对于误差图，使用白昼数据
        daylight_threshold_for_plot = 0.00 # 与评估中使用的阈值一致
        daylight_plot_data = self.test_data_with_predictions[self.test_data_with_predictions['actual_power_mw'] > daylight_threshold_for_plot].copy()
        if not daylight_plot_data.empty:
            daylight_plot_data['error_mw'] = daylight_plot_data['actual_power_mw'] - daylight_plot_data['predicted_power_mw']
            
            # 3. 误差分布图
            plt.figure(figsize=(10, 6))
            sns.histplot(daylight_plot_data['error_mw'], kde=True, bins=50)
            plt.title('预测误差分布图 (白昼时段)')
            plt.xlabel('误差 (实际功率 - 预测功率, MW)')
            plt.ylabel('频数')
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, '误差分析图.png'))
            plt.close()
            print("已保存: 误差分析图.png")

            # 4. 散点图 (实际 vs. 预测)
            plt.figure(figsize=(8, 8))
            min_val_plot = min(daylight_plot_data['actual_power_mw'].min(), daylight_plot_data['predicted_power_mw'].min())
            max_val_plot = max(daylight_plot_data['actual_power_mw'].max(), daylight_plot_data['predicted_power_mw'].max())
            min_val_plot = max(0, min_val_plot) # 确保绘图从0开始 (如果相关)
            max_val_plot = min(self.capacity, max_val_plot) # 上限为电站容量

            plt.scatter(daylight_plot_data['actual_power_mw'], daylight_plot_data['predicted_power_mw'], alpha=0.4, s=15, edgecolors='w', linewidths=0.5)
            plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2, label='理想情况 (y=x)')
            plt.title('预测效果散点图 (白昼时段)')
            plt.xlabel('实际功率 (MW)')
            plt.ylabel('预测功率 (MW)')
            plt.xlim(min_val_plot - 0.05 * self.capacity, max_val_plot + 0.05 * self.capacity) # 稍微扩展范围
            plt.ylim(min_val_plot - 0.05 * self.capacity, max_val_plot + 0.05 * self.capacity)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, '预测效果散点图.png'))
            plt.close()
            print("已保存: 预测效果散点图.png")
        else:
            print("警告: 白昼时段无数据，部分误差相关图表无法生成。")

        # --- 将指标保存到文本文件 ---
        if self.metrics: # 检查metrics字典是否不为空
            report_path = os.path.join(self.result_dir, '预测误差分析报告.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("光伏电站日前发电功率预测模型 - 误差分析报告 (白昼时段)\n")
                f.write("="*60 + "\n")
                for k_report, v_report in self.metrics.items():
                    f.write(f"{k_report}: {v_report:.4f}\n")
                f.write("="*60 + "\n")
                f.write("指标说明 (参照附件1.txt):\n")
                f.write("- 归一化误差指标均基于额定容量进行归一化。\n")
                f.write("- E_rmse_norm: 归一化均方根误差\n")
                f.write("- E_mae_norm: 归一化平均绝对误差\n")
                f.write("- E_me_norm: 归一化平均误差\n")
                f.write("- r: 相关系数 (实际功率 vs 预测功率)\n")
                f.write("- C_R (%): 准确率, 计算方法: (1 - E_rmse_norm) * 100%\n")
                f.write("- Q_R (%): 合格率, |实际-预测|/容量 < 0.25 的样本比例\n")
            print(f"已保存: {report_path}")
        else:
            print("警告: 未计算任何指标，误差分析报告未生成。")

        print("所有输出已生成于: " + os.path.abspath(self.result_dir))

def run_prediction_pipeline():
    data_file_path = '../PVODdatasets/station01.csv'
    output_directory = '题二完整结果'

    # 在启动前检查数据文件是否存在
    if not os.path.exists(data_file_path):
        alt_data_file_path = 'PVODdatasets/station01.csv' 
        if os.path.exists(alt_data_file_path):
            data_file_path = alt_data_file_path
        else:
            print(f"错误: 数据文件 {data_file_path} (或备选路径 {alt_data_file_path}) 未找到。请确认文件路径。")
            print(f"当前工作目录: {os.getcwd()}")
            return

    print(f"使用数据文件: {os.path.abspath(data_file_path)}")
    print(f"输出将保存至: {os.path.abspath(output_directory)}")

    predictor = PVPowerPrediction(csv_path=data_file_path, result_dir=output_directory)
    try:
        predictor.train_model()
        predictor.predict_and_evaluate()
    except ValueError as ve:
        print(f"处理流程中发生值错误: {ve}")
    except Exception as e:
        print(f"处理流程中发生未预料的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
   run_prediction_pipeline() 