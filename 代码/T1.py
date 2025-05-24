import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.fft import fft, fftfreq
import os
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PVStationAnalysis:
    """
    光伏电站发电特性分析类
    """
    def __init__(self, csv_path, result_dir='题一完整结果'):
        # 检查数据文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到数据文件: {csv_path}")
        self.csv_path = csv_path
        self.result_dir = result_dir
        # 创建结果文件夹
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # 电站参数
        self.latitude = 38.18306
        self.longitude = 117.45722
        self.capacity = 20  # MW
        self.panel_power = 270  # Wp
        self.panel_count = 74000
        self.panel_efficiency = 0.1623
        self.system_loss = 0.10
        self.tilt = 33
        self.azimuth = 180
        # 配置信息
        self.modules_per_string = 22
        self.strings_per_inverter = 128
        self.inverter_power = 500  # kW
        self.data = None
        self._load_and_prepare()

    def _load_and_prepare(self):
        """
        加载并预处理数据，提取时间特征
        """
        print('加载数据并预处理...')
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                raise ValueError("数据文件为空")
            # 时间特征提取
            df['datetime'] = pd.to_datetime(df['date_time'])
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['doy'] = df['datetime'].dt.dayofyear
            df['season'] = df['month'].map(lambda m: '冬季' if m in [12,1,2] else ('春季' if m in [3,4,5] else ('夏季' if m in [6,7,8] else '秋季')))
            df['solar_time'] = df['hour'] + df['minute']/60
            self.data = df
            print(f"数据量: {len(df)}，时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise

    def calc_solar_angles(self):
        """
        计算太阳高度角和方位角
        """
        print('计算太阳高度角和方位角...')
        lat_rad = np.radians(self.latitude)
        doy = self.data['doy']
        st = self.data['solar_time']
        decl = np.radians(23.45) * np.sin(np.radians(360 * (284 + doy) / 365))
        hour_angle = np.radians(15 * (st - 12))
        elev = np.arcsin(np.sin(decl) * np.sin(lat_rad) + np.cos(decl) * np.cos(lat_rad) * np.cos(hour_angle))
        az = np.arctan2(np.sin(hour_angle), np.cos(hour_angle) * np.sin(lat_rad) - np.tan(decl) * np.cos(lat_rad))
        self.data['sun_elevation'] = np.degrees(elev)
        self.data['sun_azimuth'] = np.degrees(az)

    def calc_theoretical_power(self):
        """
        计算理论可发功率
        """
        print('计算理论可发功率...')
        self.calc_solar_angles()
        elev_rad = np.radians(self.data['sun_elevation'])
        tilt_rad = np.radians(self.tilt)
        azimuth_rad = np.radians(self.azimuth)
        sun_az_rad = np.radians(self.data['sun_azimuth'])
        # 计算入射角余弦
        cos_inc = np.sin(elev_rad) * np.cos(tilt_rad) + np.cos(elev_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - azimuth_rad)
        cos_inc = np.clip(cos_inc, 0, 1)
        # 计算大气质量系数
        air_mass = np.where(self.data['sun_elevation'] > 0, 1/np.maximum(np.sin(elev_rad), 0.01), 0)
        # 计算倾斜面辐照度
        if 'nwp_globalirrad' in self.data.columns and self.data['nwp_globalirrad'].sum() > 0:
            ghi = self.data['nwp_globalirrad']
            self.data['tilted_irr'] = np.where(self.data['sun_elevation'] > 0, ghi * cos_inc / np.sin(elev_rad), 0)
        else:
            self.data['tilted_irr'] = np.where(self.data['sun_elevation'] > 0, 900 * cos_inc * np.power(0.7, air_mass), 0)
        # 温度修正
        temp_coeff = -0.0045
        stc_temp = 25
        if 'nwp_temperature' in self.data.columns:
            temp_factor = 1 + temp_coeff * (self.data['nwp_temperature'] - stc_temp)
        else:
            temp_factor = 1
        # 理论功率计算
        panel_area = 1.64 * 0.99 * self.panel_count
        self.data['theory_power'] = self.data['tilted_irr'] * panel_area * self.panel_efficiency * temp_factor * (1 - self.system_loss) / 1e6
        self.data['theory_power'] = np.minimum(self.data['theory_power'], self.capacity)

    def analyze_long_short_cycle(self):
        """
        分析长周期（月/季节）与短周期（日内）特性
        """
        print('分析长短周期特性...')
        self.data['power_eff'] = np.where(self.data['theory_power'] > 0, self.data['power'] / self.data['theory_power'], 0)
        month_stats = self.data.groupby('month').agg({'power':['mean','max','std'],'theory_power':['mean','max','std']}).round(3)
        season_stats = self.data.groupby('season').agg({'power':['mean','max','std'],'theory_power':['mean','max','std']}).round(3)
        hour_stats = self.data.groupby('hour').agg({'power':['mean','max','std'],'theory_power':['mean','max','std']}).round(3)
        return month_stats, season_stats, hour_stats

    def deviation_analysis(self):
        """
        计算实际功率与理论功率的偏差及其统计特征
        """
        print('进行偏差分析...')
        self.data['dev'] = self.data['power'] - self.data['theory_power']
        self.data['rel_dev'] = np.where(self.data['theory_power'] > 0, self.data['dev'] / self.data['theory_power'] * 100, 0)
        dev_stats = self.data['dev'].describe()
        rel_stats = self.data['rel_dev'].describe()
        return dev_stats, rel_stats

    def fft_analysis(self):
        """
        对偏差序列进行傅里叶变换，分析周期性
        """
        print('傅里叶变换分析...')
        dev = self.data['dev'].fillna(0).values
        N = len(dev)
        T = 1  # 采样间隔（可根据实际调整）
        yf = fft(dev)
        xf = fftfreq(N, T)[:N//2]
        return xf, 2.0/N * np.abs(yf[0:N//2])

    def visualize(self):
        """
        生成主要分析图表并保存
        """
        print('生成可视化图表...')
        # 月度功率对比
        plt.figure(figsize=(10,5))
        self.data.groupby('month')[['power','theory_power']].mean().plot(ax=plt.gca())
        plt.title('月均实际与理论功率')
        plt.ylabel('功率(MW)')
        plt.savefig(f'{self.result_dir}/月均功率.png')
        plt.close()

        # 日内变化对比
        plt.figure(figsize=(10,5))
        self.data.groupby('hour')[['power','theory_power']].mean().plot(ax=plt.gca())
        plt.title('日内平均功率')
        plt.ylabel('功率(MW)')
        plt.savefig(f'{self.result_dir}/日内功率.png')
        plt.close()

        # 偏差分布直方图
        plt.figure(figsize=(8,4))
        sns.histplot(self.data['dev'], bins=60, kde=True)
        plt.title('功率偏差分布')
        plt.xlabel('偏差(MW)')
        plt.savefig(f'{self.result_dir}/偏差分布.png')
        plt.close()

        # 偏差傅里叶频谱
        xf, yf = self.fft_analysis()
        plt.figure(figsize=(10,4))
        plt.plot(xf, yf)
        plt.title('偏差傅里叶频谱')
        plt.xlabel('频率')
        plt.ylabel('幅值')
        plt.savefig(f'{self.result_dir}/偏差傅里叶谱.png')
        plt.close()
        print('图表已保存至', self.result_dir)

    def report(self):
        """
        生成完整分析报告及图表
        """
        self.calc_theoretical_power()
        month_stats, season_stats, hour_stats = self.analyze_long_short_cycle()
        dev_stats, rel_stats = self.deviation_analysis()
        self.visualize()

        # 统计量中文映射
        stat_map = {
            'count': '计数',
            'mean': '均值',
            'std': '标准差',
            'min': '最小值',
            '25%': '25分位数',
            '50%': '中位数',
            '75%': '75分位数',
            'max': '最大值'
        }
        dev_stats_cn = dev_stats.rename(index=stat_map)
        rel_stats_cn = rel_stats.rename(index=stat_map)

        with open(f'{self.result_dir}/分析报告.txt', 'w', encoding='utf-8') as f:
            f.write('光伏电站发电特性分析报告\n')
            f.write(f'数据时间范围：{self.data["datetime"].min()} ~ {self.data["datetime"].max()}\n')
            f.write(f'装机容量：{self.capacity} 兆瓦\n')
            f.write(f'组件数量：{self.panel_count}\n')
            f.write('\n【月度统计】\n')
            f.write(str(month_stats.rename_axis('月份').rename(columns={'power':'实际功率','theory_power':'理论功率'})))
            f.write('\n\n【季节统计】\n')
            f.write(str(season_stats.rename_axis('季节').rename(columns={'power':'实际功率','theory_power':'理论功率'})))
            f.write('\n\n【小时统计】\n')
            f.write(str(hour_stats.rename_axis('小时').rename(columns={'power':'实际功率','theory_power':'理论功率'})))
            f.write('\n\n【偏差统计】\n')
            f.write(str(dev_stats_cn))
            f.write('\n\n【相对偏差统计(%)】\n')
            f.write(str(rel_stats_cn))
        print('分析报告已生成。')


def main():
    data_path = '../PVODdatasets/station01.csv'
    analyzer = PVStationAnalysis(data_path)
    analyzer.report()


if __name__ == '__main__':
    main() 