import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 设置全局绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

class NWPXGBoostPrediction:
    def __init__(self, csv_path, result_dir='题三完整结果'):
        self.csv_path = csv_path
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.data = None
        self.nwp_features = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                               'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
        self.target_col = 'power'
        self.time_features_engineered = []
        self.feature_names_for_model = []

        self.scaler_features = StandardScaler()
        self.power_mean_for_inverse = None
        self.power_std_for_inverse = None

        self.model_with_nwp = None
        self.model_without_nwp = None
        self.capacity = 20 # 电站装机容量 (MW)

        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        # 加载和初步处理数据
        print("加载并准备数据...")
        try:
            self.data = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.csv_path} 未找到。")
            raise
        
        if 'date_time' not in self.data.columns:
            raise ValueError("'date_time' 列缺失")
            
        self.data['datetime'] = pd.to_datetime(self.data['date_time'])
        self.data.sort_values('datetime', inplace=True)
        self.data.set_index('datetime', inplace=True, drop=False)

        # 功率列名兼容
        if self.target_col not in self.data.columns:
            if '实际功率(MW)' in self.data.columns:
                self.data.rename(columns={'实际功率(MW)': self.target_col}, inplace=True)
            elif 'Power(MW)' in self.data.columns:
                self.data.rename(columns={'Power(MW)': self.target_col}, inplace=True)
            else:
                raise ValueError(f"目标列 '{self.target_col}' (或其常见变体) 未在数据中找到。")

        self.data[self.target_col] = pd.to_numeric(self.data[self.target_col], errors='coerce')
        
        # NWP特征处理: 数值转换和插值
        for col in self.nwp_features:
            if col not in self.data.columns:
                print(f"警告: NWP特征 '{col}' 不存在，用0填充。")
                self.data[col] = 0
            else:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].interpolate(method='linear', limit_direction='both')
                self.data[col].fillna(method='bfill', inplace=True)
                self.data[col].fillna(method='ffill', inplace=True)
                self.data[col].fillna(0, inplace=True)

        self.data.dropna(subset=[self.target_col], inplace=True) # 移除功率缺失行
        print(f"数据加载完成: {self.data.index.min()} 到 {self.data.index.max()}, 共 {len(self.data)} 行。")

    def _get_season(self, month):
        # 根据月份判断季节
        if month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        elif month in [9, 10, 11]:
            return '秋季'
        else: # 12, 1, 2
            return '冬季'

    def _engineer_features(self, df_input, use_nwp_in_features=True):
        # 特征工程
        print(f"特征工程 (使用NWP: {use_nwp_in_features})...")
        df = df_input.copy() # 使用.copy()避免后续的SettingWithCopyWarning

        # 时间特征
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['minuteofday'] = df.index.hour * 60 + df.index.minute
        df['season'] = df['month'].apply(self._get_season) # 添加季节特征

        # 周期编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.)
        
        self.time_features_engineered = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 
                                         'dayofweek', 'month', 'quarter', 'minuteofday']

        # 滞后特征
        lags = [96, 96*2, 96*7] # 1天, 2天, 1周前
        lag_feature_names = []
        for lag in lags:
            feature_name = f'power_lag_{lag//96}d'
            df[feature_name] = df[self.target_col].shift(lag)
            lag_feature_names.append(feature_name)

        # 滚动统计特征 (基于1天前的功率)
        rolling_feature_names = []
        shifted_power_24h_ago = df[self.target_col].shift(96) 
        df['rolling_mean_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).mean()
        df['rolling_std_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).std()
        rolling_feature_names.extend(['rolling_mean_24h_lag1d', 'rolling_std_24h_lag1d'])

        # 构建当前模型使用的特征列表
        current_model_features = self.time_features_engineered + lag_feature_names + rolling_feature_names
        if use_nwp_in_features:
            current_model_features += self.nwp_features
        
        # 目标变量标准化
        self.power_mean_for_inverse = df[self.target_col].mean()
        self.power_std_for_inverse = df[self.target_col].std()
        if self.power_std_for_inverse == 0: # 防止除以零
            df['target_processed'] = df[self.target_col] - self.power_mean_for_inverse
        else:
            df['target_processed'] = (df[self.target_col] - self.power_mean_for_inverse) / self.power_std_for_inverse

        # 移除因特征工程产生的NaN行
        df.dropna(subset=current_model_features + ['target_processed', 'season'], inplace=True) # 确保season列也没有NaN

        X = df[current_model_features + ['season']] 
        y = df['target_processed']
        
        # 用于报告和绘图的额外信息
        report_columns_to_keep = ['datetime', self.target_col, 'season']
        if use_nwp_in_features:
            report_columns_to_keep += self.nwp_features 
        report_data_df = df.loc[:, report_columns_to_keep].copy() # 使用.loc和.copy()
        
        # 数值特征缩放
        numerical_features_to_scale = [col for col in current_model_features if col not in ['season']]
        
        X_scaled_numerical = self.scaler_features.fit_transform(X[numerical_features_to_scale])
        X_scaled_df = pd.DataFrame(X_scaled_numerical, columns=numerical_features_to_scale, index=X.index)
        
        X_final = pd.concat([X_scaled_df, X[['season']]], axis=1) # 重置season索引以匹配
        X_final.index = X_scaled_df.index # 确保最终X的索引正确
        
        self.feature_names_for_model = list(X_final.columns) 
        print(f"特征工程完成。使用特征: {self.feature_names_for_model}")
        return X_final, y, report_data_df

    def _get_test_indices_mask(self, df_with_datetime_index):
        # 定义测试集：每年2、5、8、11月的最后一周
        test_indices_mask = pd.Series(False, index=df_with_datetime_index.index)
        unique_years = df_with_datetime_index.index.year.unique()
        for year in unique_years:
            for month_num in [2, 5, 8, 11]:
                month_data = df_with_datetime_index[(df_with_datetime_index.index.year == year) & 
                                                    (df_with_datetime_index.index.month == month_num)]
                if not month_data.empty:
                    last_day_in_month = month_data.index.max().day
                    start_day_of_last_week = last_day_in_month - 6
                    mask_last_week = (df_with_datetime_index.index.year == year) & \
                                     (df_with_datetime_index.index.month == month_num) & \
                                     (df_with_datetime_index.index.day >= start_day_of_last_week) & \
                                     (df_with_datetime_index.index.day <= last_day_in_month)
                    test_indices_mask[mask_last_week] = True
        return test_indices_mask

    def train_and_evaluate_model(self, use_nwp=True):
        # 训练和评估XGBoost模型
        print(f"训练评估XGBoost模型 ({'含NWP' if use_nwp else '不含NWP'})...")
        
        X_all_features, y_all_target_processed, report_data_info = self._engineer_features(self.data, use_nwp_in_features=use_nwp)

        test_mask = self._get_test_indices_mask(X_all_features)
        train_mask = ~test_mask

        # 使用 .loc 配合 .copy() 来避免SettingWithCopyWarning
        X_train = X_all_features.loc[train_mask].copy()
        y_train = y_all_target_processed.loc[train_mask].copy()
        X_test = X_all_features.loc[test_mask].copy()
        y_test = y_all_target_processed.loc[test_mask].copy()
        
        # 转换季节为category类型
        X_train.loc[:, 'season'] = X_train['season'].astype('category')
        X_test.loc[:, 'season'] = X_test['season'].astype('category')

        test_df_report = report_data_info.loc[X_test.index].copy()
        test_df_report.loc[:, 'y_true_processed'] = y_test

        print(f"训练集: X-{X_train.shape}, y-{y_train.shape}")
        print(f"测试集: X-{X_test.shape}, y-{y_test.shape}")

        if X_train.empty or X_test.empty:
            print("错误: 训练集或测试集为空。")
            return None, {}, pd.DataFrame()

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=600,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.75,
            colsample_bytree=0.75,
            random_state=42,
            early_stopping_rounds=50,
            tree_method='hist',   # 改为hist以配合device参数
            device='cuda',        # 指定使用CUDA (GPU)
            enable_categorical=True 
        )
        
        print("训练XGBoost模型 (GPU)...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions_processed = model.predict(X_test)
        
        # 反标准化预测结果
        if self.power_std_for_inverse == 0:
            predictions_mw = predictions_processed + self.power_mean_for_inverse
        else:
            predictions_mw = (predictions_processed * self.power_std_for_inverse) + self.power_mean_for_inverse
        
        predictions_mw = np.maximum(0, predictions_mw) # 功率不为负
        predictions_mw = np.minimum(predictions_mw, self.capacity) # 不超上限

        test_df_report.loc[:, 'predicted_power_mw'] = predictions_mw
        test_df_report.loc[:, 'actual_power_mw_orig'] = test_df_report[self.target_col] 

        # 白昼时段评估
        daylight_threshold = 0.001 
        daylight_mask_report = test_df_report['actual_power_mw_orig'] > daylight_threshold
        
        y_true_daylight = test_df_report.loc[daylight_mask_report, 'actual_power_mw_orig']
        y_pred_daylight = test_df_report.loc[daylight_mask_report, 'predicted_power_mw']

        metrics = {}
        if not y_true_daylight.empty and len(y_true_daylight) > 1:
            rmse = np.sqrt(mean_squared_error(y_true_daylight, y_pred_daylight))
            mae = mean_absolute_error(y_true_daylight, y_pred_daylight)
            errors_normalized = (y_true_daylight - y_pred_daylight) / self.capacity
            e_rmse_norm = np.sqrt(np.mean(errors_normalized**2))
            e_mae_norm = np.mean(np.abs(errors_normalized))
            e_me_norm = np.mean(errors_normalized)
            if np.std(y_true_daylight) > 0 and np.std(y_pred_daylight) > 0:
                 r_correlation = np.corrcoef(y_true_daylight, y_pred_daylight)[0, 1]
            else: r_correlation = np.nan 
            c_r_accuracy = (1 - e_rmse_norm) * 100 if e_rmse_norm <=1 else 0.0
            q_r_qualified = np.mean(np.abs(errors_normalized) < 0.25) * 100
            metrics = {
                'RMSE (MW)': rmse, 'MAE (MW)': mae, 'E_rmse_norm': e_rmse_norm,
                'E_mae_norm': e_mae_norm, 'E_me_norm': e_me_norm, '相关系数r': r_correlation,
                '准确率C_R (%)': c_r_accuracy, '合格率Q_R (%)': q_r_qualified,
                '白昼样本数': len(y_true_daylight)
            }
            print(f"评估指标 ({'NWP' if use_nwp else 'No NWP'}, 白昼): {metrics}")
        else:
            print("白昼数据不足，无法计算详细指标。")

        # 特征重要性图
        if hasattr(model, 'feature_importances_') and self.feature_names_for_model:
            plt.figure(figsize=(12, max(8, len(self.feature_names_for_model) * 0.35)))
            importances = model.get_booster().get_score(importance_type='total_gain') 
            if importances: 
                sorted_features = sorted(importances.items(), key=lambda item: item[1])
                feature_names_sorted = [item[0] for item in sorted_features]
                importance_values_sorted = [item[1] for item in sorted_features]
                
                plt.barh(feature_names_sorted, importance_values_sorted)
                plt.xlabel("XGBoost 特征重要性 (Total Gain)")
                plt.title(f'特征重要性分析 ({("带NWP" if use_nwp else "不带NWP")})')
                plt.tight_layout()
                plt.savefig(os.path.join(self.result_dir, f'特征重要性分析图_{"NWP" if use_nwp else "NoNWP"}.png'))
                plt.close()
            else:
                print("未能获取特征重要性。")
        return model, metrics, test_df_report
    
    def _plot_additional_visualizations(self, test_df_report_model, model_type_suffix):
        # 绘制新增的可视化图表
        plot_dir = self.result_dir
        actual_col = 'actual_power_mw_orig'
        predicted_col = 'predicted_power_mw'

        if test_df_report_model.empty or not all(col in test_df_report_model.columns for col in [actual_col, predicted_col, 'datetime']):
            print(f"警告: {model_type_suffix} 模型的测试报告数据不足，无法生成附加图表。")
            return

        df_to_plot = test_df_report_model.copy() # 操作副本
        df_to_plot['datetime'] = pd.to_datetime(df_to_plot['datetime'])
        daylight_mask = df_to_plot[actual_col] > 0.001 

        errors = df_to_plot.loc[daylight_mask, actual_col] - df_to_plot.loc[daylight_mask, predicted_col]
        if not errors.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True, bins=50)
            plt.title(f'预测误差分布 (残差) - {model_type_suffix}')
            plt.xlabel('预测误差 (实际功率 - 预测功率) (MW)')
            plt.ylabel('频数')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'误差分析图_残差直方图_{model_type_suffix}.png'))
            plt.close()

        if 'nwp_globalirrad' in df_to_plot.columns:
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(df_to_plot.loc[daylight_mask, actual_col], 
                             df_to_plot.loc[daylight_mask, predicted_col], 
                             c=df_to_plot.loc[daylight_mask, 'nwp_globalirrad'], 
                             cmap='viridis', alpha=0.5, s=10)
            plt.plot([0, self.capacity], [0, self.capacity], 'r--', label='理想情况 (y=x)')
            plt.colorbar(sc, label='全局水平辐照度 (W/m2)') # 改为 W/m2
            plt.title(f'辐射强度下预测效果 - {model_type_suffix}')
            plt.xlabel('实际功率 (MW)')
            plt.ylabel('预测功率 (MW)')
            plt.xlim(0, self.capacity + 1)
            plt.ylim(0, self.capacity + 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'辐射强度预测效果图_{model_type_suffix}.png'))
            plt.close()
        else:
            if "NWP" in model_type_suffix: # 只在NWP模型时提示缺失
                print(f"警告: {model_type_suffix} 报告中缺少 'nwp_globalirrad'，无法生成辐射强度预测图。")

        if 'nwp_globalirrad' in df_to_plot.columns:
            irrad_median = df_to_plot['nwp_globalirrad'].median()
            sunny_threshold = irrad_median * 1.2
            cloudy_threshold = irrad_median * 0.5
            
            df_to_plot.loc[:, 'weather_type'] = '过渡' # 使用.loc避免警告
            df_to_plot.loc[df_to_plot['nwp_globalirrad'] > sunny_threshold, 'weather_type'] = '晴天'
            df_to_plot.loc[df_to_plot['nwp_globalirrad'] < cloudy_threshold, 'weather_type'] = '阴天/多云'

            plt.figure(figsize=(12, 7))
            sns.scatterplot(data=df_to_plot[daylight_mask], x=actual_col, y=predicted_col, hue='weather_type', alpha=0.6, s=15)
            plt.plot([0, self.capacity], [0, self.capacity], 'r--', label='理想情况 (y=x)')
            plt.title(f'不同天气类型下预测效果 - {model_type_suffix}')
            plt.xlabel('实际功率 (MW)')
            plt.ylabel('预测功率 (MW)')
            plt.xlim(0, self.capacity + 1)
            plt.ylim(0, self.capacity + 1)
            plt.legend(title='天气类型')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'天气类型预测效果图_{model_type_suffix}.png'))
            plt.close()
        else:
            if "NWP" in model_type_suffix: # 只在NWP模型时提示缺失
                 print(f"警告: {model_type_suffix} 报告中缺少 'nwp_globalirrad'，无法生成天气类型预测图。")

        if 'season' in df_to_plot.columns and not errors.empty:
            df_to_plot.loc[daylight_mask, 'error'] = errors 
            plt.figure(figsize=(10, 6))
            season_order = ['春季', '夏季', '秋季', '冬季']
            available_seasons = [s for s in season_order if s in df_to_plot.loc[daylight_mask, 'season'].unique()]

            if available_seasons: 
                sns.boxplot(data=df_to_plot[daylight_mask], x='season', y='error', order=available_seasons)
                plt.title(f'不同季节预测误差对比 - {model_type_suffix}')
                plt.xlabel('季节')
                plt.ylabel('预测误差 (MW)')
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'季节预测效果图_{model_type_suffix}.png'))
                plt.close()
            else:
                print(f"警告: {model_type_suffix} 报告中季节数据不足或误差数据缺失，无法生成季节预测图。")

    def analyze_scenario_performance(self, test_df_report_nwp, test_df_report_no_nwp):
        # 场景分析 (晴天 vs 阴天/多云)
        print("\n场景分析...")
        if 'nwp_globalirrad' not in test_df_report_nwp.columns:
            print("警告: NWP模型报告缺少 'nwp_globalirrad'，无法进行场景划分。")
            return {}, {} 

        irrad_median = test_df_report_nwp['nwp_globalirrad'].median()
        # 调整场景划分阈值，使其更具区分度，避免样本过少
        sunny_threshold = test_df_report_nwp['nwp_globalirrad'].quantile(0.7) # 例如，高于70%分位点为晴天
        cloudy_threshold = test_df_report_nwp['nwp_globalirrad'].quantile(0.3) # 例如，低于30%分位点为阴天/多云
        
        daylight_mask_nwp = test_df_report_nwp['actual_power_mw_orig'] > 0.001
        daylight_mask_no_nwp = test_df_report_no_nwp['actual_power_mw_orig'] > 0.001

        scenarios = {
            "晴天": test_df_report_nwp['nwp_globalirrad'] > sunny_threshold,
            "阴天或多云": test_df_report_nwp['nwp_globalirrad'] < cloudy_threshold, 
        }
        scenario_metrics_nwp = {}
        scenario_metrics_no_nwp = {}

        for scenario_name, scenario_mask_on_nwp_df in scenarios.items():
            final_mask_nwp = daylight_mask_nwp & scenario_mask_on_nwp_df
            common_indices = test_df_report_nwp.index.intersection(test_df_report_no_nwp.index)
            scenario_mask_aligned_for_no_nwp = scenario_mask_on_nwp_df.reindex(common_indices).fillna(False) # 确保对齐和处理NaN
            
            if not isinstance(test_df_report_no_nwp.index, pd.DatetimeIndex):
                 test_df_report_no_nwp.index = pd.to_datetime(test_df_report_no_nwp.index)

            final_mask_no_nwp_indices = test_df_report_no_nwp.index.isin(common_indices[scenario_mask_aligned_for_no_nwp])
            final_mask_no_nwp = daylight_mask_no_nwp & final_mask_no_nwp_indices

            def calculate_scenario_metrics(df, mask, capacity):
                # 确保df的索引是DatetimeIndex，以便正确使用.loc[mask]
                if not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('datetime', drop=False)
                
                # 筛选出真正为True的mask索引
                valid_mask_indices = mask[mask].index
                if not valid_mask_indices.empty:
                    y_true = df.loc[valid_mask_indices, 'actual_power_mw_orig']
                    y_pred = df.loc[valid_mask_indices, 'predicted_power_mw']
                    if len(y_true) > 1:
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mae = mean_absolute_error(y_true, y_pred)
                        errors_norm = (y_true - y_pred) / capacity
                        e_rmse_norm = np.sqrt(np.mean(errors_norm**2))
                        e_mae_norm = np.mean(np.abs(errors_norm))
                        if np.std(y_true) > 0 and np.std(y_pred) > 0:
                            r_corr = np.corrcoef(y_true, y_pred)[0, 1]
                        else: r_corr = np.nan
                        return {'RMSE (MW)': rmse, 'MAE (MW)': mae, '样本数': len(y_true),
                                'E_RMSE_norm': e_rmse_norm, 'E_MAE_norm': e_mae_norm, '相关系数r': r_corr}
                return None

            metrics_nwp = calculate_scenario_metrics(test_df_report_nwp.copy(), final_mask_nwp, self.capacity)
            if metrics_nwp: scenario_metrics_nwp[scenario_name] = metrics_nwp
            
            metrics_no_nwp = calculate_scenario_metrics(test_df_report_no_nwp.copy(), final_mask_no_nwp, self.capacity)
            if metrics_no_nwp: scenario_metrics_no_nwp[scenario_name] = metrics_no_nwp
        
        print("NWP模型场景性能:", scenario_metrics_nwp)
        print("无NWP模型场景性能:", scenario_metrics_no_nwp)
        return scenario_metrics_nwp, scenario_metrics_no_nwp

    def generate_report_and_save_results(self, metrics_nwp, metrics_no_nwp, scenario_metrics_nwp, scenario_metrics_no_nwp, 
                                         test_df_report_with_nwp, test_df_report_without_nwp):
        # 生成分析报告并保存所有结果文件
        print("生成分析报告和保存结果...")
        report_path = os.path.join(self.result_dir, '题三分析报告_XGBoost.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("融入NWP信息的光伏功率预测分析报告 (XGBoost模型)\n")
            f.write("="*70 + "\n\n")
            f.write("1. 模型概述\n")
            f.write("   - 使用XGBoost模型进行15分钟间隔日前光伏功率预测。\n")
            f.write("   - 特征包括时间特征、功率滞后特征、滚动统计特征、季节以及NWP气象数据。\n")
            f.write("   - XGBoost参数: n_estimators=600, learning_rate=0.03, max_depth=8, early_stopping_rounds=50, tree_method='hist', device='cuda'。\n")
            f.write("   - 对比了包含NWP特征和不包含NWP特征的两种XGBoost模型。\n\n")

            f.write("2. 整体性能评估 (白昼时段)\n")
            f.write("   2.1 模型 (包含NWP特征):\n")
            if metrics_nwp:
                for k, v_val in metrics_nwp.items(): f.write(f"      - {k}: {v_val:.4f}\n")
            else: f.write("      - 未计算指标。\n")
            
            f.write("\n   2.2 模型 (不包含NWP特征 - 基线模型):\n")
            if metrics_no_nwp:
                for k, v_val in metrics_no_nwp.items(): f.write(f"      - {k}: {v_val:.4f}\n")
            else: f.write("      - 未计算指标。\n")
            
            f.write("\n   2.3 NWP信息增益分析:\n")
            if metrics_nwp and metrics_no_nwp and 'RMSE (MW)' in metrics_nwp and 'RMSE (MW)' in metrics_no_nwp:
                rmse_gain = metrics_no_nwp['RMSE (MW)'] - metrics_nwp['RMSE (MW)']
                mae_gain = metrics_no_nwp.get('MAE (MW)', float('inf')) - metrics_nwp.get('MAE (MW)', float('inf'))
                f.write(f"      - RMSE提升 (正值表示NWP模型更优): {rmse_gain:.4f} MW\n")
                f.write(f"      - MAE提升 (正值表示NWP模型更优): {mae_gain:.4f} MW\n")
                if rmse_gain > 0.001:
                     f.write(f"      - 结论: 整体看，融入NWP信息提高了预测精度 (RMSE降低 {rmse_gain:.4f} MW)。\n")
                else:
                     f.write(f"      - 结论: 整体看，NWP信息未显著提高预测精度，或效果不佳。\n")
            else:
                f.write("      - 无法计算NWP增益 (指标缺失)。\n")

            f.write("\n3. 分场景性能评估 (白昼时段, 基于NWP模型测试集的nwp_globalirrad划分)\n")
            all_scenarios_keys = set(scenario_metrics_nwp.keys()) | set(scenario_metrics_no_nwp.keys())
            if not all_scenarios_keys:
                f.write("   - 未进行场景分析或无结果。\n")

            for scenario in sorted(list(all_scenarios_keys)):
                f.write(f"\n   场景: {scenario}\n")
                f.write("      模型 (包含NWP):\n")
                if scenario in scenario_metrics_nwp and scenario_metrics_nwp[scenario]:
                     for k,v_val in scenario_metrics_nwp[scenario].items(): f.write(f"         - {k}: {v_val:.4f}\n")
                else: f.write("         - 无此场景数据或指标 (NWP模型)。\n")

                f.write("      模型 (不包含NWP):\n")
                if scenario in scenario_metrics_no_nwp and scenario_metrics_no_nwp[scenario]:
                    for k,v_val in scenario_metrics_no_nwp[scenario].items(): f.write(f"         - {k}: {v_val:.4f}\n")
                else: f.write("         - 无此场景数据或指标 (无NWP模型)。\n")

                if scenario in scenario_metrics_nwp and scenario in scenario_metrics_no_nwp and \
                   scenario_metrics_nwp.get(scenario) and scenario_metrics_no_nwp.get(scenario) and \
                   'RMSE (MW)' in scenario_metrics_nwp[scenario] and 'RMSE (MW)' in scenario_metrics_no_nwp[scenario]:
                    s_rmse_gain = scenario_metrics_no_nwp[scenario]['RMSE (MW)'] - scenario_metrics_nwp[scenario]['RMSE (MW)']
                    f.write(f"      场景NWP增益 (RMSE降低): {s_rmse_gain:.4f} MW\n")
                    if s_rmse_gain > 0.01: 
                        f.write(f"      场景结论: NWP信息在此场景下有效提高精度。\n")
                    elif s_rmse_gain < -0.01:
                        f.write(f"      场景结论: NWP信息在此场景下反而降低精度。\n")
                    else:
                        f.write(f"      场景结论: NWP信息在此场景下影响不显著。\n")
            
            f.write("\n4. 提高预测精度的场景划分方案验证\n")
            f.write("   根据分场景评估结果：\n")
            sunny_improves_text = "不确定或无足够数据"
            cloudy_improves_text = "不确定或无足够数据"
            if scenario_metrics_nwp.get("晴天") and scenario_metrics_no_nwp.get("晴天") and \
               scenario_metrics_nwp["晴天"].get('RMSE (MW)') is not None and scenario_metrics_no_nwp["晴天"].get('RMSE (MW)') is not None:
                if (scenario_metrics_no_nwp["晴天"]['RMSE (MW)'] - scenario_metrics_nwp["晴天"]['RMSE (MW)']) > 0.01:
                    sunny_improves_text = "NWP有效提高精度"
                else:
                    sunny_improves_text = "NWP效果不显著或降低精度"
            
            if scenario_metrics_nwp.get("阴天或多云") and scenario_metrics_no_nwp.get("阴天或多云") and \
               scenario_metrics_nwp["阴天或多云"].get('RMSE (MW)') is not None and scenario_metrics_no_nwp["阴天或多云"].get('RMSE (MW)') is not None:
                 if (scenario_metrics_no_nwp["阴天或多云"]['RMSE (MW)'] - scenario_metrics_nwp["阴天或多云"]['RMSE (MW)']) > 0.01:
                    cloudy_improves_text = "NWP有效提高精度"
                 else:
                    cloudy_improves_text = "NWP效果不显著或降低精度"
            f.write(f"   - 晴天场景: {sunny_improves_text}。\n")
            f.write(f"   - 阴天或多云场景: {cloudy_improves_text}。\n")
            f.write("   - 建议方案: 优先在NWP能显著提升精度的场景使用带NWP的XGBoost模型。其他场景需进一步分析NWP质量或模型鲁棒性。\n\n")
            
            f.write("5. 结论与建议\n")
            f.write("   - XGBoost模型结合NWP特征在光伏功率预测中表现出潜力。\n")
            if metrics_nwp and metrics_no_nwp and (metrics_no_nwp.get('RMSE (MW)', float('inf')) - metrics_nwp.get('RMSE (MW)', 0)) > 0.001 :
                f.write("   - 总体而言，NWP信息的融入倾向于提高XGBoost模型的预测精度。\n")
            else:
                f.write("   - NWP信息对XGBoost整体预测精度的提升效果需结合具体场景深入分析。\n")
            f.write("   - XGBoost模型训练快速，便于调优，是光伏预测的有效方法。GPU加速能进一步提高效率。\n")
            f.write("   - 未来可探索：更精细的特征工程，如交互特征；更全面的超参数寻优；集成学习策略。\n")
        print(f"分析报告已保存至: {report_path}")

        if test_df_report_with_nwp is not None and not test_df_report_with_nwp.empty:
            output_df_main = test_df_report_with_nwp[['datetime', 'actual_power_mw_orig', 'predicted_power_mw']].copy()
            output_df_main.rename(columns={
                'datetime': '时间', 'actual_power_mw_orig': '实际功率(MW)',
                'predicted_power_mw': '预测功率_带NWP_XGB(MW)'}, inplace=True)
            
            if test_df_report_without_nwp is not None and not test_df_report_without_nwp.empty:
                # 确保datetime列用于合并，且是datetime类型
                df_no_nwp_for_merge = test_df_report_without_nwp[['datetime', 'predicted_power_mw']].copy()
                # 首先重命名列，包括 'datetime' -> '时间'
                df_no_nwp_for_merge.rename(columns={
                    'datetime': '时间',
                    'predicted_power_mw': '预测功率_无NWP_XGB(MW)'
                }, inplace=True)
                # 然后对新的 '时间' 列进行类型转换
                df_no_nwp_for_merge['时间'] = pd.to_datetime(df_no_nwp_for_merge['时间'])
                output_df_main['时间'] = pd.to_datetime(output_df_main['时间']) # 确保 output_df_main 的 '时间' 列也是 datetime

                output_df_main = pd.merge(output_df_main,
                                        df_no_nwp_for_merge, # df_no_nwp_for_merge 现在包含 '时间' 列
                                        on='时间', how='left')

            csv_filename_all = os.path.join(self.result_dir, "详细预测结果_T3_XGBoost_全时段.csv")
            output_df_main.to_csv(csv_filename_all, index=False, encoding='utf-8-sig')
            print(f"已保存全时段详细预测结果: {csv_filename_all}")

            output_df_main['时间'] = pd.to_datetime(output_df_main['时间'])
            # 使用 .dt.strftime 来确保 Period 对象被正确转换为年月字符串
            output_df_main['月份分组'] = output_df_main['时间'].dt.to_period('M')
            for month_period, group_data in output_df_main.groupby('月份分组'):
                month_str = month_period.strftime('%Y年%m月') # Period对象可以直接用strftime
                month_csv_path = os.path.join(self.result_dir, f'预测结果{month_str}.csv')
                group_data_to_save = group_data.drop(columns=['月份分组'])
                group_data_to_save.to_csv(month_csv_path, index=False, encoding='utf-8-sig')
                print(f"已保存 {month_str} 预测结果至: {month_csv_path}")

            plt.figure(figsize=(18, 7))
            sample_count_plot = min(1000, len(output_df_main))
            if sample_count_plot > 0:
                plot_df_sorted = output_df_main.sort_values(by='时间')
                sample_plot_df = plot_df_sorted.sample(n=sample_count_plot, random_state=42).sort_values(by='时间')
                
                plt.plot(sample_plot_df['时间'], sample_plot_df['实际功率(MW)'], label='实际功率', alpha=0.7, linewidth=1.5)
                plt.plot(sample_plot_df['时间'], sample_plot_df['预测功率_带NWP_XGB(MW)'], label='预测功率 (XGBoost带NWP)', linestyle='--', linewidth=1.2)
                if '预测功率_无NWP_XGB(MW)' in sample_plot_df.columns:
                    plt.plot(sample_plot_df['时间'], sample_plot_df['预测功率_无NWP_XGB(MW)'], label='预测功率 (XGBoost无NWP)', linestyle=':', linewidth=1.2)
                plt.title('光伏功率日前预测对比 (随机样本)')
                plt.xlabel('时间')
                plt.ylabel('功率 (MW)')
                plt.legend(loc='upper left')
                plt.xticks(rotation=25)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.result_dir, '时间序列预测对比图.png'))
                plt.close()
            else:
                print("警告: 输出数据为空，无法生成主预测对比图。")
        
        if test_df_report_with_nwp is not None and not test_df_report_with_nwp.empty:
            self._plot_additional_visualizations(test_df_report_with_nwp, "带NWP")
        if test_df_report_without_nwp is not None and not test_df_report_without_nwp.empty:
             self._plot_additional_visualizations(test_df_report_without_nwp, "无NWP")

    def run_pipeline(self):
        # 执行完整流水线
        self.model_with_nwp, metrics_nwp, test_df_report_nwp = self.train_and_evaluate_model(use_nwp=True)
        self.model_without_nwp, metrics_no_nwp, test_df_report_no_nwp = self.train_and_evaluate_model(use_nwp=False)
        
        scenario_metrics_nwp, scenario_metrics_no_nwp = {}, {}
        if test_df_report_nwp is not None and not test_df_report_nwp.empty and \
           test_df_report_no_nwp is not None and not test_df_report_no_nwp.empty:
            # 确保datetime列存在且为索引，以便场景分析中的reindex和筛选
            if 'datetime' not in test_df_report_nwp.columns:
                test_df_report_nwp['datetime'] = test_df_report_nwp.index.to_series()
            if 'datetime' not in test_df_report_no_nwp.columns:
                 test_df_report_no_nwp['datetime'] = test_df_report_no_nwp.index.to_series()

            # 确保 test_df_report 的索引是 datetime 类型
            if not isinstance(test_df_report_nwp.index, pd.DatetimeIndex):
                test_df_report_nwp = test_df_report_nwp.set_index(pd.to_datetime(test_df_report_nwp['datetime']), drop=False)
            if not isinstance(test_df_report_no_nwp.index, pd.DatetimeIndex):
                test_df_report_no_nwp = test_df_report_no_nwp.set_index(pd.to_datetime(test_df_report_no_nwp['datetime']), drop=False)

            scenario_metrics_nwp, scenario_metrics_no_nwp = self.analyze_scenario_performance(
                test_df_report_nwp, test_df_report_no_nwp
            )
        else:
            print("一个或两个模型的测试报告为空，跳过场景分析。")

        self.generate_report_and_save_results(metrics_nwp, metrics_no_nwp, 
                                               scenario_metrics_nwp, scenario_metrics_no_nwp,
                                               test_df_report_nwp, test_df_report_no_nwp)

def main():
    # 主函数入口
    data_file_options = ['../PVODdatasets/station01.csv', 'PVODdatasets/station01.csv']
    data_file = None
    for option in data_file_options:
        if os.path.exists(option):
            data_file = option
            break
    
    if data_file is None:
        print(f"错误: 数据文件 'station01.csv' 在预设路径中均未找到。")
        return
            
    output_dir = '题三完整结果'
    
    pipeline = NWPXGBoostPrediction(csv_path=data_file, result_dir=output_dir)
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()
