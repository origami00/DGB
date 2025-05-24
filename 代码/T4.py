import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 设置全局绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

class DownscaledNWPXGBoostPrediction:
    def __init__(self, csv_path, result_dir='题四完整结果'):
        self.csv_path = csv_path
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.data_orig = None # 存储最原始加载的数据
        self.nwp_features = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                               'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
        self.simulated_downscaled_nwp_features = [] # 存储模拟降尺度后的NWP特征列名
        self.target_col = 'power'
        self.time_features_engineered = [] # 通用时间特征名
        
        self.scaler_features_dict = {} # 为每个模型类型存储一个scaler
        self.power_mean_for_inverse = None # 假设功率的均值和标准差在不同预处理阶段保持一致或重新计算
        self.power_std_for_inverse = None

        self.model_no_nwp = None
        self.model_with_orig_nwp = None
        self.model_with_downscaled_nwp = None
        
        self.metrics_no_nwp = {}
        self.metrics_orig_nwp = {}
        self.metrics_downscaled_nwp = {}

        self.test_df_no_nwp = pd.DataFrame()
        self.test_df_orig_nwp = pd.DataFrame()
        self.test_df_downscaled_nwp = pd.DataFrame()

        self.capacity = 20 # 电站装机容量 (MW)
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print("加载并准备数据...")
        try:
            self.data_orig = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.csv_path} 未找到。")
            raise
        
        if 'date_time' not in self.data_orig.columns:
            raise ValueError("'date_time' 列缺失")
            
        self.data_orig['datetime'] = pd.to_datetime(self.data_orig['date_time'])
        self.data_orig.sort_values('datetime', inplace=True)
        self.data_orig.set_index('datetime', inplace=True, drop=False)

        if self.target_col not in self.data_orig.columns:
            common_power_cols = {'实际功率(MW)': self.target_col, 'Power(MW)': self.target_col}
            renamed = False
            for c_old, c_new in common_power_cols.items():
                if c_old in self.data_orig.columns:
                    self.data_orig.rename(columns={c_old: c_new}, inplace=True)
                    renamed = True
                    break
            if not renamed:
                raise ValueError(f"目标列 '{self.target_col}' (或其常见变体) 未在数据中找到。")

        self.data_orig[self.target_col] = pd.to_numeric(self.data_orig[self.target_col], errors='coerce')
        
        for col in self.nwp_features:
            if col not in self.data_orig.columns:
                print(f"警告: NWP特征 '{col}' 不存在，用0填充。")
                self.data_orig[col] = 0
            else:
                self.data_orig[col] = pd.to_numeric(self.data_orig[col], errors='coerce')
                self.data_orig[col] = self.data_orig[col].interpolate(method='linear', limit_direction='both')
                self.data_orig[col].fillna(method='bfill', inplace=True)
                self.data_orig[col].fillna(method='ffill', inplace=True)
                self.data_orig[col].fillna(0, inplace=True)

        self.data_orig.dropna(subset=[self.target_col], inplace=True)
        print(f"数据加载完成: {self.data_orig.index.min()} 到 {self.data_orig.index.max()}, 共 {len(self.data_orig)} 行。")

    def _get_season(self, month):
        if month in [3, 4, 5]: return '春季'
        elif month in [6, 7, 8]: return '夏季'
        elif month in [9, 10, 11]: return '秋季'
        else: return '冬季'

    def _simulate_nwp_downscaling(self, df_input):
        print("模拟NWP空间降尺度...")
        df = df_input.copy()
        self.simulated_downscaled_nwp_features = []
        for col in self.nwp_features:
            # 模拟方法: 原始值基础上增加随机扰动（表示更精细的局部变化），然后轻微平滑
            # 这是一个高度简化的模拟，真实降尺度方法会复杂得多
            noise_scale = 0.02 # 噪声幅度占标准差的百分比
            noise = np.random.normal(0, df[col].std() * noise_scale, size=len(df))
            downscaled_col_name = f'{col}_downscaled'
            # 确保模拟值在合理范围内，例如辐照度非负
            if 'irrad' in col:
                df[downscaled_col_name] = np.maximum(0, (df[col] + noise).rolling(window=3, min_periods=1, center=True).mean())
            else:
                df[downscaled_col_name] = (df[col] + noise).rolling(window=3, min_periods=1, center=True).mean()
            
            df[downscaled_col_name].fillna(df[col], inplace=True) # 填充因rolling产生的NaN
            self.simulated_downscaled_nwp_features.append(downscaled_col_name)
        print(f"模拟生成的降尺度NWP特征: {self.simulated_downscaled_nwp_features}")
        return df

    def _engineer_features(self, df_input, model_type_key, use_orig_nwp=False, use_downscaled_nwp=False):
        print(f"特征工程 (模型类型: {model_type_key})...")
        df = df_input.copy()

        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['minuteofday'] = df.index.hour * 60 + df.index.minute
        df['season'] = df['month'].apply(self._get_season)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.)
        
        self.time_features_engineered = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 
                                         'dayofweek', 'month', 'quarter', 'minuteofday']
        
        lags = [96, 96*2, 96*7] # 1天, 2天, 1周
        lag_feature_names = []
        for lag in lags:
            feature_name = f'power_lag_{lag//96}d'
            df[feature_name] = df[self.target_col].shift(lag)
            lag_feature_names.append(feature_name)

        rolling_feature_names = []
        shifted_power_24h_ago = df[self.target_col].shift(96) 
        df['rolling_mean_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).mean()
        df['rolling_std_24h_lag1d'] = shifted_power_24h_ago.rolling(window=96, min_periods=1).std()
        rolling_feature_names.extend(['rolling_mean_24h_lag1d', 'rolling_std_24h_lag1d'])

        current_model_features = self.time_features_engineered + lag_feature_names + rolling_feature_names
        
        nwp_cols_to_include = []
        if use_orig_nwp:
            nwp_cols_to_include.extend(self.nwp_features)
        if use_downscaled_nwp:
            # 如果同时使用原始和降尺度，确保不重复添加原始NWP（若降尺度数据已包含其含义）
            # 这里假设降尺度特征是全新的，或者与原始特征并行使用
            nwp_cols_to_include.extend(self.simulated_downscaled_nwp_features)
        
        # 去重，以防原始NWP和降尺度NWP有重叠（虽然模拟中列名不同）
        current_model_features.extend(list(set(nwp_cols_to_include)))
        
        # 确保 self.power_mean_for_inverse 和 std 在首次计算后能被复用
        if self.power_mean_for_inverse is None:
            self.power_mean_for_inverse = df[self.target_col].mean()
            self.power_std_for_inverse = df[self.target_col].std()

        if self.power_std_for_inverse == 0:
            df['target_processed'] = df[self.target_col] - self.power_mean_for_inverse
        else:
            df['target_processed'] = (df[self.target_col] - self.power_mean_for_inverse) / self.power_std_for_inverse

        df.dropna(subset=current_model_features + ['target_processed', 'season'], inplace=True)

        X = df[current_model_features + ['season']]
        y = df['target_processed']
        
        report_columns_to_keep = ['datetime', self.target_col, 'season'] + nwp_cols_to_include
        report_data_df = df.loc[:, df.columns.intersection(report_columns_to_keep)].copy()
        
        numerical_features_to_scale = [col for col in current_model_features if col not in ['season']]
        
        # 每个模型类型使用独立的scaler
        if model_type_key not in self.scaler_features_dict:
            self.scaler_features_dict[model_type_key] = StandardScaler()
            X_scaled_numerical = self.scaler_features_dict[model_type_key].fit_transform(X[numerical_features_to_scale])
        else:
            X_scaled_numerical = self.scaler_features_dict[model_type_key].transform(X[numerical_features_to_scale])

        X_scaled_df = pd.DataFrame(X_scaled_numerical, columns=numerical_features_to_scale, index=X.index)
        X_final = pd.concat([X_scaled_df, X[['season']]], axis=1)
        
        feature_names_for_model = list(X_final.columns)
        print(f"特征工程完成 ({model_type_key})。使用特征数: {len(feature_names_for_model)}")
        return X_final, y, report_data_df, feature_names_for_model

    def _get_test_indices_mask(self, df_with_datetime_index):
        test_indices_mask = pd.Series(False, index=df_with_datetime_index.index)
        unique_years = df_with_datetime_index.index.year.unique()
        for year in unique_years:
            for month_num in [2, 5, 8, 11]:
                month_data = df_with_datetime_index[(df_with_datetime_index.index.year == year) & 
                                                    (df_with_datetime_index.index.month == month_num)]
                if not month_data.empty:
                    last_day_in_month = month_data.index.max().day
                    start_day_of_last_week = max(1, last_day_in_month - 6) # 确保起始日不小于1
                    mask_last_week = (df_with_datetime_index.index.year == year) & \
                                     (df_with_datetime_index.index.month == month_num) & \
                                     (df_with_datetime_index.index.day >= start_day_of_last_week) & \
                                     (df_with_datetime_index.index.day <= last_day_in_month)
                    test_indices_mask[mask_last_week] = True
        return test_indices_mask

    def _train_single_model(self, data_for_features, model_type_key, features_config, common_feature_names_for_model_list):
        X_all_features, y_all_target_processed, report_data_info, feature_names = self._engineer_features(
            data_for_features, 
            model_type_key,
            use_orig_nwp=features_config.get('use_orig_nwp', False),
            use_downscaled_nwp=features_config.get('use_downscaled_nwp', False)
        )
        common_feature_names_for_model_list.clear() # 使用 clear() 然后 extend() 来修改传入的列表
        common_feature_names_for_model_list.extend(feature_names)


        test_mask = self._get_test_indices_mask(X_all_features)
        train_mask = ~test_mask

        X_train = X_all_features.loc[train_mask].copy()
        y_train = y_all_target_processed.loc[train_mask].copy()
        X_test = X_all_features.loc[test_mask].copy()
        y_test = y_all_target_processed.loc[test_mask].copy()
        
        X_train['season'] = X_train['season'].astype('category')
        X_test['season'] = X_test['season'].astype('category')

        test_df_report = report_data_info.loc[X_test.index].copy()
        test_df_report['y_true_processed'] = y_test # 保留处理后的y用于可能的分析

        print(f"训练集 ({model_type_key}): X-{X_train.shape}, y-{y_train.shape}")
        print(f"测试集 ({model_type_key}): X-{X_test.shape}, y-{y_test.shape}")

        if X_train.empty or X_test.empty:
            print(f"错误: {model_type_key} 模型的训练集或测试集为空。")
            return None, {}, pd.DataFrame()

        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
            max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=50, tree_method='hist', device='cuda', enable_categorical=True
        )
        
        print(f"训练XGBoost模型 ({model_type_key})...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions_processed = model.predict(X_test)
        
        if self.power_std_for_inverse == 0: # should have been set in _engineer_features
            predictions_mw = predictions_processed + self.power_mean_for_inverse
        else:
            predictions_mw = (predictions_processed * self.power_std_for_inverse) + self.power_mean_for_inverse
        
        predictions_mw = np.maximum(0, predictions_mw)
        predictions_mw = np.minimum(predictions_mw, self.capacity)

        test_df_report['predicted_power_mw'] = predictions_mw
        test_df_report['actual_power_mw_orig'] = test_df_report[self.target_col] 

        daylight_threshold = 0.001 
        daylight_mask_report = test_df_report['actual_power_mw_orig'] > daylight_threshold
        
        y_true_daylight = test_df_report.loc[daylight_mask_report, 'actual_power_mw_orig']
        y_pred_daylight = test_df_report.loc[daylight_mask_report, 'predicted_power_mw']

        metrics = {}
        if not y_true_daylight.empty and len(y_true_daylight) > 1:
            rmse = np.sqrt(mean_squared_error(y_true_daylight, y_pred_daylight))
            mae = mean_absolute_error(y_true_daylight, y_pred_daylight)
            if self.capacity == 0: errors_normalized = np.zeros_like(y_true_daylight)
            else: errors_normalized = (y_true_daylight - y_pred_daylight) / self.capacity
            
            e_rmse_norm = np.sqrt(np.mean(errors_normalized**2))
            e_mae_norm = np.mean(np.abs(errors_normalized))
            e_me_norm = np.mean(errors_normalized)
            r_correlation = np.nan
            if np.std(y_true_daylight) > 0 and np.std(y_pred_daylight) > 0:
                 r_correlation = np.corrcoef(y_true_daylight, y_pred_daylight)[0, 1]
            c_r_accuracy = (1 - e_rmse_norm) * 100 if e_rmse_norm <=1 else 0.0
            q_r_qualified_abs_power = np.mean(np.abs(y_true_daylight - y_pred_daylight) < (0.1 * self.capacity)) * 100 # 10% of capacity
            q_r_qualified_relative = np.mean(np.abs(errors_normalized) < 0.2) * 100 # 20% normalized error
            
            metrics = {
                'RMSE (MW)': rmse, 'MAE (MW)': mae, 'E_rmse_norm': e_rmse_norm,
                'E_mae_norm': e_mae_norm, 'E_me_norm': e_me_norm, '相关系数r': r_correlation,
                '准确率C_R (%)': c_r_accuracy, '合格率Q_R (相对20%)': q_r_qualified_relative,
                '白昼样本数': len(y_true_daylight)
            }
            print(f"评估指标 ({model_type_key}, 白昼): {metrics}")
        else:
            print(f"{model_type_key}: 白昼数据不足，无法计算详细指标。")

        if hasattr(model, 'feature_importances_') and common_feature_names_for_model_list:
            plt.figure(figsize=(12, max(8, len(common_feature_names_for_model_list) * 0.3)))
            importances = model.get_booster().get_score(importance_type='total_gain')
            if importances:
                # Filter out features not in common_feature_names_for_model_list if xgb adds some internals
                filtered_importances = {k: v for k, v in importances.items() if k in common_feature_names_for_model_list}
                if filtered_importances:
                    sorted_features = sorted(filtered_importances.items(), key=lambda item: item[1])
                    feature_names_sorted = [item[0] for item in sorted_features]
                    importance_values_sorted = [item[1] for item in sorted_features]
                    
                    plt.barh(feature_names_sorted, importance_values_sorted)
                    plt.xlabel("XGBoost 特征重要性 (Total Gain)")
                    plt.title(f'特征重要性分析 - {model_type_key}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.result_dir, f'特征重要性分析图_{model_type_key}.png'))
                    plt.close()
            else:
                print(f"{model_type_key}: 未能获取特征重要性。")
        return model, metrics, test_df_report

    def train_all_models(self):
        data_for_downscaling_sim = self.data_orig.copy()
        data_with_downscaled_features = self._simulate_nwp_downscaling(data_for_downscaling_sim)
        
        # Store feature names for each model to pass to plotting
        self.feature_names_no_nwp = []
        self.feature_names_orig_nwp = []
        self.feature_names_downscaled_nwp = []

        # 模型1: 无NWP
        self.model_no_nwp, self.metrics_no_nwp, self.test_df_no_nwp = self._train_single_model(
            self.data_orig.copy(), '无NWP', 
            {'use_orig_nwp': False, 'use_downscaled_nwp': False},
            self.feature_names_no_nwp
        )
        # 模型2: 含原始NWP
        self.model_with_orig_nwp, self.metrics_orig_nwp, self.test_df_orig_nwp = self._train_single_model(
            self.data_orig.copy(), '原始NWP', 
            {'use_orig_nwp': True, 'use_downscaled_nwp': False},
            self.feature_names_orig_nwp
        )
        # 模型3: 含模拟降尺度NWP (这里也包含原始NWP作为对比基准的一部分特征，如果需要纯降尺度，调整features_config)
        self.model_with_downscaled_nwp, self.metrics_downscaled_nwp, self.test_df_downscaled_nwp = self._train_single_model(
            data_with_downscaled_features.copy(), '降尺度NWP', 
            {'use_orig_nwp': True, 'use_downscaled_nwp': True}, # 同时使用原始和降尺度特征
            self.feature_names_downscaled_nwp
        )
        
    def _plot_additional_visualizations_q4(self, test_df_report_model, model_type_suffix):
        plot_dir = self.result_dir
        actual_col = 'actual_power_mw_orig'
        predicted_col = 'predicted_power_mw'

        if test_df_report_model.empty or not all(col in test_df_report_model.columns for col in [actual_col, predicted_col, 'datetime']):
            print(f"警告: {model_type_suffix} 模型的测试报告数据不足，无法生成附加图表。")
            return

        df_to_plot = test_df_report_model.copy()
        df_to_plot['datetime'] = pd.to_datetime(df_to_plot['datetime']) # 确保datetime是datetime对象
        daylight_mask = df_to_plot[actual_col] > 0.001
        
        # 辐射强度预测效果图
        if 'nwp_globalirrad' in df_to_plot.columns:
            plt.figure(figsize=(10, 8))
            # 使用白昼数据进行绘图
            df_daylight = df_to_plot[daylight_mask]
            if not df_daylight.empty:
                sc = plt.scatter(df_daylight[actual_col], 
                                 df_daylight[predicted_col], 
                                 c=df_daylight['nwp_globalirrad'], 
                                 cmap='viridis', alpha=0.6, s=15)
                plt.plot([0, self.capacity], [0, self.capacity], 'r--', label='理想情况 (y=x)')
                plt.colorbar(sc, label='全局水平辐照度 (W/m2)')
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
             # 仅当模型类型应包含NWP时才警告
            if "NWP" in model_type_suffix.upper():
                print(f"警告: {model_type_suffix} 报告中缺少 'nwp_globalirrad'，无法生成辐射强度预测图。")


        # 天气类型预测效果图
        # (假设使用原始NWP中的nwp_globalirrad进行天气类型划分，即使是降尺度模型也基于此统一划分)
        weather_ref_col = 'nwp_globalirrad' # 或者用降尺度的对应列
        if weather_ref_col in df_to_plot.columns:
            irrad_q75 = df_to_plot[weather_ref_col].quantile(0.75)
            irrad_q25 = df_to_plot[weather_ref_col].quantile(0.25)
            
            df_to_plot['weather_type'] = '过渡'
            df_to_plot.loc[df_to_plot[weather_ref_col] > irrad_q75, 'weather_type'] = '晴朗'
            df_to_plot.loc[df_to_plot[weather_ref_col] < irrad_q25, 'weather_type'] = '阴云'

            plt.figure(figsize=(12, 7))
            if not df_to_plot[daylight_mask].empty:
                sns.scatterplot(data=df_to_plot[daylight_mask], x=actual_col, y=predicted_col, hue='weather_type', 
                                hue_order=['晴朗', '过渡', '阴云'], palette={'晴朗':'#FFD700', '过渡':'#ADD8E6', '阴云':'#A9A9A9'}, 
                                alpha=0.6, s=15)
                plt.plot([0, self.capacity], [0, self.capacity], 'r--', label='理想情况 (y=x)')
                plt.title(f'不同天气类型下预测效果 - {model_type_suffix}')
                plt.xlabel('实际功率 (MW)')
                plt.ylabel('预测功率 (MW)')
                plt.xlim(0, self.capacity + 1); plt.ylim(0, self.capacity + 1)
                plt.legend(title='天气类型'); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'天气类型预测效果图_{model_type_suffix}.png'))
                plt.close()
        else:
            if "NWP" in model_type_suffix.upper() or "降尺度" in model_type_suffix:
                 print(f"警告: {model_type_suffix} 报告中缺少 '{weather_ref_col}'，无法生成天气类型预测图。")

        # 季节预测效果图
        if 'season' in df_to_plot.columns:
            df_daylight_errors = df_to_plot[daylight_mask].copy()
            if not df_daylight_errors.empty:
                df_daylight_errors['error'] = df_daylight_errors[actual_col] - df_daylight_errors[predicted_col]
                plt.figure(figsize=(10, 6))
                season_order = ['春季', '夏季', '秋季', '冬季']
                # Filter for available seasons in the data to avoid errors with boxplot order
                available_seasons = [s for s in season_order if s in df_daylight_errors['season'].unique()]
                if available_seasons:
                    sns.boxplot(data=df_daylight_errors, x='season', y='error', order=available_seasons)
                    plt.title(f'不同季节预测误差对比 - {model_type_suffix}')
                    plt.xlabel('季节'); plt.ylabel('预测误差 (MW)'); plt.grid(axis='y'); plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'季节预测效果图_{model_type_suffix}.png'))
                    plt.close()
                else:
                    print(f"警告: {model_type_suffix} 的季节数据不足或误差数据为空，无法生成季节误差箱线图。")
            else:
                 print(f"警告: {model_type_suffix} 的白昼数据为空，无法生成季节误差箱线图。")
                 
    def _plot_downscaling_effect_comparison_q4(self):
        metrics_list = []
        if self.metrics_no_nwp: metrics_list.append({'model': '无NWP', **self.metrics_no_nwp})
        if self.metrics_orig_nwp: metrics_list.append({'model': '原始NWP', **self.metrics_orig_nwp})
        if self.metrics_downscaled_nwp: metrics_list.append({'model': '降尺度NWP', **self.metrics_downscaled_nwp})

        if not metrics_list:
            print("无足够模型结果进行降尺度效果对比。")
            return

        df_metrics = pd.DataFrame(metrics_list)
        
        if 'RMSE (MW)' not in df_metrics.columns or 'MAE (MW)' not in df_metrics.columns:
            print("关键指标缺失，无法绘制降尺度效果对比图。")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='RMSE (MW)', hue='model', data=df_metrics, palette='viridis', legend=False)
        plt.title('空间降尺度效果分析 - RMSE对比')
        plt.ylabel('RMSE (MW)')
        plt.xlabel('模型类型')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, '空间降尺度效果分析图_RMSE.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='MAE (MW)', hue='model', data=df_metrics, palette='viridis', legend=False)
        plt.title('空间降尺度效果分析 - MAE对比')
        plt.ylabel('MAE (MW)')
        plt.xlabel('模型类型')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, '空间降尺度效果分析图_MAE.png'))
        plt.close()
        print("空间降尺度效果对比图已保存。")

    def analyze_scenario_performance_q4(self):
        # 场景分析 (平原、山地 - 通过nwp_windspeed模拟)
        # 复用T3的晴天/阴天场景分析逻辑，但应用于三个模型
        report_content = "\n\n6. 分场景性能评估\n"
        report_content += "="*60 + "\n"

        models_data = {
            "无NWP": (self.test_df_no_nwp, self.metrics_no_nwp),
            "原始NWP": (self.test_df_orig_nwp, self.metrics_orig_nwp),
            "降尺度NWP": (self.test_df_downscaled_nwp, self.metrics_downscaled_nwp)
        }
        
        # 场景1: 天气类型 (基于原始NWP的 globalirrad)
        report_content += "\n   6.1 天气类型场景 (基于原始NWP globalirrad 划分)\n"
        if not self.test_df_orig_nwp.empty and 'nwp_globalirrad' in self.test_df_orig_nwp.columns:
            irrad_q70 = self.test_df_orig_nwp['nwp_globalirrad'].quantile(0.7)
            irrad_q30 = self.test_df_orig_nwp['nwp_globalirrad'].quantile(0.3)
            
            scenarios_weather = {
                "晴朗天 (辐照度 > 70%)": self.test_df_orig_nwp['nwp_globalirrad'] > irrad_q70,
                "阴云天 (辐照度 < 30%)": self.test_df_orig_nwp['nwp_globalirrad'] < irrad_q30
            }
            
            for scenario_name, scenario_mask_orig_nwp in scenarios_weather.items():
                report_content += f"\n      场景: {scenario_name}\n"
                for model_name, (test_df, _) in models_data.items():
                    if test_df.empty: continue
                    # 将原始NWP场景的mask对齐到各个模型的测试集索引上
                    common_indices = test_df.index.intersection(self.test_df_orig_nwp.index[scenario_mask_orig_nwp])
                    if common_indices.empty:
                        report_content += f"         - {model_name} 模型: 在此场景无公共样本。\n"
                        continue
                    
                    y_true_scenario = test_df.loc[common_indices, 'actual_power_mw_orig']
                    y_pred_scenario = test_df.loc[common_indices, 'predicted_power_mw']
                    
                    if len(y_true_scenario) > 1:
                        rmse_sc = np.sqrt(mean_squared_error(y_true_scenario, y_pred_scenario))
                        mae_sc = mean_absolute_error(y_true_scenario, y_pred_scenario)
                        report_content += f"         - {model_name} 模型: RMSE={rmse_sc:.3f} MW, MAE={mae_sc:.3f} MW (样本数: {len(y_true_scenario)})\n"
                    else:
                        report_content += f"         - {model_name} 模型: 样本数不足 (<2)。\n"
        else:
            report_content += "      无法进行天气类型场景分析 (原始NWP数据或辐照度列缺失)。\n"

        # 场景2: 模拟地形 (平原/山地 - 基于原始NWP的 windspeed)
        report_content += "\n   6.2 模拟地形场景 (基于原始NWP windspeed 划分)\n"
        if not self.test_df_orig_nwp.empty and 'nwp_windspeed' in self.test_df_orig_nwp.columns:
            wind_median = self.test_df_orig_nwp['nwp_windspeed'].median()
            scenarios_terrain = {
                "模拟平原 (低风速)": self.test_df_orig_nwp['nwp_windspeed'] <= wind_median,
                "模拟山地 (高风速)": self.test_df_orig_nwp['nwp_windspeed'] > wind_median
            }
            for scenario_name, scenario_mask_orig_nwp in scenarios_terrain.items():
                report_content += f"\n      场景: {scenario_name}\n"
                for model_name, (test_df, _) in models_data.items():
                    if test_df.empty: continue
                    common_indices = test_df.index.intersection(self.test_df_orig_nwp.index[scenario_mask_orig_nwp])
                    if common_indices.empty:
                        report_content += f"         - {model_name} 模型: 在此场景无公共样本。\n"
                        continue
                        
                    y_true_scenario = test_df.loc[common_indices, 'actual_power_mw_orig']
                    y_pred_scenario = test_df.loc[common_indices, 'predicted_power_mw']
                    if len(y_true_scenario) > 1:
                        rmse_sc = np.sqrt(mean_squared_error(y_true_scenario, y_pred_scenario))
                        mae_sc = mean_absolute_error(y_true_scenario, y_pred_scenario)
                        report_content += f"         - {model_name} 模型: RMSE={rmse_sc:.3f} MW, MAE={mae_sc:.3f} MW (样本数: {len(y_true_scenario)})\n"
                    else:
                        report_content += f"         - {model_name} 模型: 样本数不足 (<2)。\n"
        else:
            report_content += "      无法进行模拟地形场景分析 (原始NWP数据或风速列缺失)。\n"
        
        return report_content.strip()


    def generate_report_and_save_results_q4(self):
        print("生成分析报告和保存结果 (题四)...")
        report_path = os.path.join(self.result_dir, '题四分析报告.txt')
        
        scenario_report_text = self.analyze_scenario_performance_q4()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("NWP空间降尺度对光伏功率预测精度影响分析报告\n")
            f.write("="*70 + "\n\n")
            f.write("1. 研究背景与目的\n")
            f.write("   探讨在现有NWP数据基础上，通过模拟的空间降尺度方法得到的NWP信息，\n")
            f.write("   是否能够提高光伏电站日前发电功率的预测精度。\n")
            f.write("   对比模型: 无NWP模型、使用原始NWP的模型、使用模拟降尺度NWP的模型。\n\n")

            f.write("2. 数据与预处理\n")
            f.write(f"   数据来源: {self.csv_path}\n")
            f.write(f"   时间范围: {self.data_orig.index.min()} 到 {self.data_orig.index.max()}\n")
            f.write("   NWP特征: " + ", ".join(self.nwp_features) + "\n")
            f.write("   目标变量: 'power' (MW)\n\n")
            
            f.write("3. NWP空间降尺度方法（模拟）\n")
            f.write("   由于缺乏多点或格点化NWP数据进行真实的空间降尺度，本报告采用模拟方法：\n")
            f.write("   对原始NWP各特征值加入基于其自身标准差的随机扰动，然后进行轻微滑动平均平滑。\n")
            f.write("   例如: new_nwp = moving_avg(original_nwp + random_noise * std(original_nwp))\n")
            f.write("   这旨在模拟更高分辨率NWP数据可能带来的局部细节变化。\n")
            f.write(f"   模拟生成的降尺度特征列: {self.simulated_downscaled_nwp_features}\n\n")

            f.write("4. 预测模型构建\n")
            f.write("   - 核心模型: XGBoost回归模型。\n")
            f.write("   - GPU加速: tree_method='hist', device='cuda'。\n")
            f.write("   - 特征工程: 时间特征（周期编码）、功率滞后特征、功率滚动统计特征、季节。\n")
            f.write("     根据模型不同，分别组合（无NWP、原始NWP、原始NWP+模拟降尺度NWP）。\n")
            f.write("   - 关于CNN: 用户思路中提及CNN提取空间特征。由于当前数据为单点时间序列，\n")
            f.write("     且降尺度NWP为模拟的单点增强特征，未直接集成CNN。若未来有格点化降尺度NWP，\n")
            f.write("     可考虑使用CNN提取空间上下文特征后输入XGBoost。\n\n")

            f.write("5. 整体性能评估 (白昼时段测试集)\n")
            f.write("="*60 + "\n")
            models_metrics = [
                ("无NWP模型", self.metrics_no_nwp),
                ("原始NWP模型", self.metrics_orig_nwp),
                ("降尺度NWP模型 (模拟)", self.metrics_downscaled_nwp)
            ]
            for name, metrics in models_metrics:
                f.write(f"\n   {name}:\n")
                if metrics:
                    for k, v_val in metrics.items(): f.write(f"      - {k}: {v_val:.4f}\n")
                else: f.write("      - 未计算或无结果。\n")
            
            if self.metrics_orig_nwp and self.metrics_downscaled_nwp and \
               'RMSE (MW)' in self.metrics_orig_nwp and 'RMSE (MW)' in self.metrics_downscaled_nwp:
                rmse_diff = self.metrics_orig_nwp['RMSE (MW)'] - self.metrics_downscaled_nwp['RMSE (MW)']
                f.write(f"\n   降尺度NWP模型 vs 原始NWP模型 (RMSE改善，正值为优): {rmse_diff:.4f} MW\n")
                if rmse_diff > 0.01:
                    f.write("      结论提示: 模拟的降尺度NWP信息显示出改善预测精度的潜力。\n")
                elif rmse_diff < -0.01:
                    f.write("      结论提示: 模拟的降尺度NWP信息可能未带来改善，甚至略差，模拟方法或特征组合需审视。\n")
                else:
                    f.write("      结论提示: 模拟的降尺度NWP信息带来的影响不显著。\n")
            
            f.write(scenario_report_text) # 添加场景分析结果

            f.write("\n\n7. 结论与讨论\n")
            f.write("="*60 + "\n")
            f.write("   - 本研究通过模拟NWP空间降尺度数据，初步探讨了其对光伏功率预测的影响。\n")
            # 基于metrics_downscaled_nwp 和 metrics_orig_nwp的对比给出结论
            if self.metrics_downscaled_nwp and self.metrics_orig_nwp and \
               self.metrics_downscaled_nwp.get('RMSE (MW)', float('inf')) < self.metrics_orig_nwp.get('RMSE (MW)', float('inf')):
                f.write("   - 从模拟结果看，使用降尺度NWP特征的模型在总体RMSE/MAE上表现出一定的优势或相似性能，\n")
                f.write("     表明更高分辨率或经过优化的NWP信息可能有助于提升预测精度。\n")
            else:
                f.write("   - 从模拟结果看，当前模拟的降尺度NWP特征并未显著优于原始NWP特征，甚至可能表现略差。\n")
                f.write("     这可能意味着模拟方法过于简单，未能捕捉到真实降尺度的有效信息，或者特征组合方式需要优化。\n")
            f.write("   - 场景分析显示，不同模型在特定气象或模拟地形条件下的表现可能存在差异。\n")
            f.write("   - 未来研究方向:\n")
            f.write("     - 获取真实的、多点或格点化的高分辨率NWP降尺度数据进行验证。\n")
            f.write("     - 探索更先进的降尺度模型（如基于物理的、统计的或深度学习的真实降尺度方法）。\n")
            f.write("     - 若有格点化数据，可研究CNN等模型提取空间特征并与XGBoost等模型融合的策略。\n")
            f.write("     - 进一步优化特征工程和模型超参数。\n\n")
            f.write('注意: 本报告中的"降尺度NWP"是基于简化模拟，其结论的普适性需结合真实降尺度数据进一步验证。\n')

        print(f"分析报告已保存至: {report_path}")

        # 保存详细预测结果CSV
        # 合并三个模型的预测结果到一个DataFrame
        final_output_df = pd.DataFrame()

        # 第一个模型 (无NWP)
        if not self.test_df_no_nwp.empty:
            df_ = self.test_df_no_nwp.copy()
            df_.index.name = None # 解除索引名与列名的潜在冲突
            df_ = df_.reset_index(drop=True) # 使用简单整数索引，保留'datetime'列
            # 现在 df_ 中的 'datetime' 明确是一个列
            df_ = df_[['datetime', 'actual_power_mw_orig', 'predicted_power_mw']].copy()
            df_['datetime'] = pd.to_datetime(df_['datetime'])
            df_.rename(columns={'predicted_power_mw': '预测功率_无NWP(MW)', 
                                'actual_power_mw_orig': '实际功率(MW)'}, inplace=True)
            final_output_df = df_
        
        # 第二个模型 (原始NWP)
        if not self.test_df_orig_nwp.empty:
            df_ = self.test_df_orig_nwp.copy()
            df_.index.name = None
            df_ = df_.reset_index(drop=True)
            df_ = df_[['datetime', 'actual_power_mw_orig', 'predicted_power_mw']].copy()
            df_['datetime'] = pd.to_datetime(df_['datetime'])
            df_.rename(columns={'predicted_power_mw': '预测功率_原始NWP(MW)'}, inplace=True)

            cols_to_merge = ['datetime', '预测功率_原始NWP(MW)']
            if '实际功率(MW)' not in final_output_df.columns and 'actual_power_mw_orig' in df_.columns :
                 df_.rename(columns={'actual_power_mw_orig': '实际功率(MW)'}, inplace=True)
                 if '实际功率(MW)' in df_.columns: cols_to_merge.append('实际功率(MW)')
            
            df_merge_subset = df_[[col for col in cols_to_merge if col in df_.columns]]

            if not final_output_df.empty:
                final_output_df = pd.merge(final_output_df, df_merge_subset, on='datetime', how='outer')
            else:
                final_output_df = df_merge_subset

        # 第三个模型 (降尺度NWP)
        if not self.test_df_downscaled_nwp.empty:
            df_ = self.test_df_downscaled_nwp.copy()
            df_.index.name = None
            df_ = df_.reset_index(drop=True)
            df_ = df_[['datetime', 'actual_power_mw_orig', 'predicted_power_mw']].copy()
            df_['datetime'] = pd.to_datetime(df_['datetime'])
            df_.rename(columns={'predicted_power_mw': '预测功率_降尺度NWP(MW)'}, inplace=True)

            cols_to_merge = ['datetime', '预测功率_降尺度NWP(MW)']
            if '实际功率(MW)' not in final_output_df.columns and 'actual_power_mw_orig' in df_.columns:
                 df_.rename(columns={'actual_power_mw_orig': '实际功率(MW)'}, inplace=True)
                 if '实际功率(MW)' in df_.columns: cols_to_merge.append('实际功率(MW)')

            df_merge_subset = df_[[col for col in cols_to_merge if col in df_.columns]]
            
            if not final_output_df.empty:
                final_output_df = pd.merge(final_output_df, df_merge_subset, on='datetime', how='outer')
            else:
                final_output_df = df_merge_subset


        if not final_output_df.empty:
            final_output_df.sort_values('datetime', inplace=True)
            csv_all_time = os.path.join(self.result_dir, "详细预测结果_题四_全时段.csv")
            final_output_df.to_csv(csv_all_time, index=False, encoding='utf-8-sig')
            print(f"已保存全时段详细预测结果: {csv_all_time}")

            # 按月份保存
            final_output_df['datetime'] = pd.to_datetime(final_output_df['datetime']) # 再次确保datetime类型
            final_output_df['月份分组'] = final_output_df['datetime'].dt.to_period('M')
            
            for month_period, group_data in final_output_df.groupby('月份分组'):
                if pd.isna(month_period): # 检查是否为 NaT
                    print(f"警告: 检测到无效的月份分组 (NaT)，跳过保存。受影响数据行数: {len(group_data)}")
                    continue
                
                # 使用YYYY-MM格式生成文件名，更健壮
                month_filename_str = month_period.strftime('%Y-%m')
                # 尝试生成中文显示格式的月份字符串
                month_display_str = month_period.strftime('%Y年%m月')
                
                # 如果中文格式返回空字符串，则回退到YYYY-MM格式进行显示
                if not month_display_str:
                    month_display_str = month_filename_str 
                
                month_csv_path = os.path.join(self.result_dir, f'预测结果_题四_{month_filename_str}.csv')
                group_data_to_save = group_data.drop(columns=['月份分组'])
                group_data_to_save.to_csv(month_csv_path, index=False, encoding='utf-8-sig')
                print(f"已保存 {month_display_str} 预测结果至: {month_csv_path}")
            
            # 时间序列预测对比图
            plt.figure(figsize=(18, 9))
            sample_count_plot = min(1000, len(final_output_df))
            if sample_count_plot > 0:
                plot_df_sorted = final_output_df.sort_values(by='datetime')
                if 'datetime' in plot_df_sorted.columns:
                    sample_plot_df = plot_df_sorted.sample(n=sample_count_plot, random_state=42).sort_values(by='datetime')
                    
                    plt.plot(sample_plot_df['datetime'], sample_plot_df['实际功率(MW)'], label='实际功率', alpha=0.7, linewidth=1.5, color='black')
                    if '预测功率_无NWP(MW)' in sample_plot_df.columns:
                        plt.plot(sample_plot_df['datetime'], sample_plot_df['预测功率_无NWP(MW)'], label='预测功率 (无NWP)', linestyle=':', linewidth=1.2)
                    if '预测功率_原始NWP(MW)' in sample_plot_df.columns:
                        plt.plot(sample_plot_df['datetime'], sample_plot_df['预测功率_原始NWP(MW)'], label='预测功率 (原始NWP)', linestyle='--', linewidth=1.2)
                    if '预测功率_降尺度NWP(MW)' in sample_plot_df.columns:
                        plt.plot(sample_plot_df['datetime'], sample_plot_df['预测功率_降尺度NWP(MW)'], label='预测功率 (降尺度NWP)', linestyle='-.', linewidth=1.2)
                    
                    plt.title('光伏功率日前预测对比 (随机样本) - 题四')
                    plt.xlabel('时间'); plt.ylabel('功率 (MW)'); plt.legend(loc='upper left')
                    plt.xticks(rotation=30); plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.result_dir, '时间序列预测对比图_题四.png'))
                    plt.close()
            else:
                print("警告: 详细预测结果为空，无法生成主预测对比图。")

        # 生成各模型的附加可视化图
        if not self.test_df_no_nwp.empty:
            self._plot_additional_visualizations_q4(self.test_df_no_nwp, "无NWP")
        if not self.test_df_orig_nwp.empty:
            self._plot_additional_visualizations_q4(self.test_df_orig_nwp, "原始NWP")
        if not self.test_df_downscaled_nwp.empty:
            self._plot_additional_visualizations_q4(self.test_df_downscaled_nwp, "降尺度NWP")

        # 生成降尺度效果对比图
        self._plot_downscaling_effect_comparison_q4()


    def run_pipeline(self):
        self.train_all_models()
        self.generate_report_and_save_results_q4()
        print("题四处理流水线完成。")

def main():
    data_file_options = ['../PVODdatasets/station01.csv', 'PVODdatasets/station01.csv']
    data_file = None
    for option in data_file_options:
        if os.path.exists(option):
            data_file = option
            break
    
    if data_file is None:
        print(f"错误: 数据文件 'station01.csv' 在预设路径中均未找到。程序将退出。")
        return # 退出
            
    output_dir_q4 = '题四完整结果' 
    
    pipeline_q4 = DownscaledNWPXGBoostPrediction(csv_path=data_file, result_dir=output_dir_q4)
    pipeline_q4.run_pipeline()

if __name__ == '__main__':
    main() 