"""
生成示例船舶AIS数据
用于演示和测试
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional
import platform

# 配置中文字体支持
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统使用微软雅黑或SimHei
        try:
            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        except:
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    elif system == 'Darwin':  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'STSong']
    else:  # Linux
        # 尝试多个中文字体，按优先级排列
        matplotlib.rcParams['font.sans-serif'] = [
            'Noto Sans CJK SC',  # Google Noto字体（简体中文）
            'Noto Sans CJK TC',  # Google Noto字体（繁体中文）
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',   # 文泉驿正黑
            'Droid Sans Fallback',  # Android fallback字体
            'SimHei',  # 黑体
            'DejaVu Sans'  # 默认备用字体
        ]

    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

def generate_ship_trajectory(
    num_samples: int = 1000,
    start_lon: float = 121.5,
    start_lat: float = 31.2,
    base_speed: float = 15.0,
    base_course: float = 45.0,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    生成模拟船舶轨迹数据
    
    Args:
        num_samples: 样本数量
        start_lon: 起始经度
        start_lat: 起始纬度
        base_speed: 基准航速（节）
        base_course: 基准航向（度）
        noise_level: 噪声水平
        
    Returns:
        包含船舶轨迹的DataFrame
    """
    np.random.seed(42)
    
    # 初始化数据
    data = {
        'longitude': [],
        'latitude': [],
        'speed': [],
        'course': []
    }
    
    # 当前位置和状态
    current_lon = start_lon
    current_lat = start_lat
    current_speed = base_speed
    current_course = base_course
    
    for i in range(num_samples):
        # 记录当前状态
        data['longitude'].append(current_lon)
        data['latitude'].append(current_lat)
        data['speed'].append(current_speed)
        data['course'].append(current_course)
        
        # 模拟航速变化（带一定惯性）
        speed_change = np.random.normal(0, noise_level * 2)
        current_speed = np.clip(
            current_speed + speed_change,
            base_speed - 5,
            base_speed + 5
        )
        
        # 模拟航向变化（渐变，模拟转向）
        if i % 200 == 0 and i > 0:
            # 每200步模拟一次转向
            course_change = np.random.normal(0, 30)
        else:
            # 平稳航行时的小幅波动
            course_change = np.random.normal(0, noise_level * 10)
        
        current_course = (current_course + course_change) % 360
        
        # 根据航速和航向更新位置
        # 1节 ≈ 1.852 km/h ≈ 0.01668 度/小时（纬度）
        # 时间步假设为10分钟 = 1/6 小时
        time_step = 1/6  # 小时
        
        # 计算位移（度）
        speed_deg_per_hour = current_speed * 0.01668
        distance = speed_deg_per_hour * time_step
        
        # 转换航向为弧度（航向从北开始顺时针）
        course_rad = np.radians(90 - current_course)  # 转换为数学角度
        
        # 更新位置
        current_lon += distance * np.cos(course_rad)
        current_lat += distance * np.sin(course_rad)
        
        # 添加GPS定位噪声
        current_lon += np.random.normal(0, noise_level * 0.001)
        current_lat += np.random.normal(0, noise_level * 0.001)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加时间戳（可选）
    df['timestamp'] = pd.date_range(
        start='2025-01-01 00:00:00',
        periods=num_samples,
        freq='10min'  # 10分钟间隔
    )
    
    return df


def generate_multiple_scenarios(output_dir: str = 'data/raw'):
    """
    生成多种场景的船舶轨迹数据
    
    Args:
        output_dir: 输出目录
        
    Returns:
        tuple: (合并数据, 场景数据字典)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成示例船舶AIS数据...")
    
    # 场景1: 标准航行（训练用）
    print("1. 生成标准航行数据...")
    df_standard = generate_ship_trajectory(
        num_samples=2000,
        start_lon=121.5,
        start_lat=31.2,
        base_speed=15.0,
        base_course=45.0,
        noise_level=0.1
    )
    
    # 场景2: 高速航行
    print("2. 生成高速航行数据...")
    df_high_speed = generate_ship_trajectory(
        num_samples=500,
        start_lon=122.0,
        start_lat=31.5,
        base_speed=25.0,
        base_course=135.0,
        noise_level=0.15
    )
    
    # 场景3: 低速航行（如港口区域）
    print("3. 生成低速航行数据...")
    df_low_speed = generate_ship_trajectory(
        num_samples=500,
        start_lon=121.0,
        start_lat=30.8,
        base_speed=5.0,
        base_course=270.0,
        noise_level=0.2
    )
    
    # 保存场景数据字典
    scenario_dict = {
        '标准航行': df_standard,
        '高速航行': df_high_speed,
        '低速航行': df_low_speed
    }
    
    # 合并数据
    df_combined = pd.concat([df_standard, df_high_speed, df_low_speed], ignore_index=True)
    
    # 随机打乱（可选）
    # df_combined = df_combined.sample(frac=1).reset_index(drop=True)
    
    # 保存主数据集
    output_path = os.path.join(output_dir, 'ship_data.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"   总样本数: {len(df_combined)}")
    print(f"   特征: {list(df_combined.columns)}")
    
    # 生成统计信息
    print("\n数据统计信息:")
    print("=" * 50)
    print(df_combined[['longitude', 'latitude', 'speed', 'course']].describe())
    
    # 保存小样本测试数据
    test_output_path = os.path.join(output_dir, 'ship_data_test.csv')
    df_combined.head(100).to_csv(test_output_path, index=False)
    print(f"\n✅ 测试数据已保存到: {test_output_path}")
    
    return df_combined, scenario_dict


def add_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.01) -> pd.DataFrame:
    """
    向数据中添加异常值（用于测试数据清洗）
    
    Args:
        df: 原始数据
        anomaly_rate: 异常值比例
        
    Returns:
        包含异常值的数据
    """
    df_with_anomalies = df.copy()
    num_anomalies = int(len(df) * anomaly_rate)
    
    # 随机选择异常值位置
    anomaly_indices = np.random.choice(len(df), num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['speed', 'coordinate', 'missing'])
        
        if anomaly_type == 'speed':
            # 异常航速
            df_with_anomalies.loc[idx, 'speed'] = np.random.uniform(50, 100)
        elif anomaly_type == 'coordinate':
            # 异常坐标
            df_with_anomalies.loc[idx, 'longitude'] = np.random.uniform(-180, 180)
            df_with_anomalies.loc[idx, 'latitude'] = np.random.uniform(-90, 90)
        elif anomaly_type == 'missing':
            # 缺失值
            df_with_anomalies.loc[idx, np.random.choice(['speed', 'course'])] = np.nan
    
    return df_with_anomalies


def visualize_trajectory_data(df: pd.DataFrame, 
                              save_path: Optional[str] = None,
                              show_plot: bool = True):
    """
    可视化船舶轨迹数据
    
    Args:
        df: 包含轨迹数据的DataFrame
        save_path: 保存路径（可选）
        show_plot: 是否显示图表
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 绘制轨迹图（经纬度）
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['longitude'], df['latitude'], 
                         c=df.index, cmap='viridis', 
                         s=10, alpha=0.6)
    ax1.plot(df['longitude'], df['latitude'], 
            'b-', alpha=0.3, linewidth=0.5)
    ax1.plot(df['longitude'].iloc[0], df['latitude'].iloc[0], 
            'go', markersize=12, label='起点')
    ax1.plot(df['longitude'].iloc[-1], df['latitude'].iloc[-1], 
            'rs', markersize=12, label='终点')
    ax1.set_xlabel('经度 (Longitude)', fontsize=11)
    ax1.set_ylabel('纬度 (Latitude)', fontsize=11)
    ax1.set_title('船舶轨迹图', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='时间步')
    
    # 2. 速度随时间变化
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df.index, df['speed'], 'b-', linewidth=1.5, alpha=0.8)
    ax2.fill_between(df.index, df['speed'], alpha=0.3)
    ax2.set_xlabel('时间步', fontsize=11)
    ax2.set_ylabel('速度 (节)', fontsize=11)
    ax2.set_title('速度变化曲线', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    mean_speed = df['speed'].mean()
    ax2.axhline(y=mean_speed, color='r', linestyle='--', 
               label=f'平均速度: {mean_speed:.2f}节')
    ax2.legend()
    
    # 3. 航向随时间变化
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df.index, df['course'], 'g-', linewidth=1.5, alpha=0.8)
    ax3.fill_between(df.index, df['course'], alpha=0.3, color='green')
    ax3.set_xlabel('时间步', fontsize=11)
    ax3.set_ylabel('航向 (度)', fontsize=11)
    ax3.set_title('航向变化曲线', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 360)
    
    # 4. 速度分布直方图
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df['speed'], bins=30, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax4.set_xlabel('速度 (节)', fontsize=11)
    ax4.set_ylabel('频数', fontsize=11)
    ax4.set_title('速度分布直方图', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 航向分布极坐标图
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    course_rad = np.radians(df['course'])
    ax5.hist(course_rad, bins=36, color='coral', alpha=0.7)
    ax5.set_title('航向分布（极坐标）', fontsize=13, fontweight='bold', pad=20)
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    
    # 6. 数据统计摘要
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    数据统计摘要
    {'='*40}
    
    总样本数: {len(df):,}
    时间跨度: {df['timestamp'].iloc[0]} 
              至 {df['timestamp'].iloc[-1]}
    
    经度范围: {df['longitude'].min():.4f} ~ {df['longitude'].max():.4f}
    纬度范围: {df['latitude'].min():.4f} ~ {df['latitude'].max():.4f}
    
    速度统计 (节):
      - 平均: {df['speed'].mean():.2f}
      - 标准差: {df['speed'].std():.2f}
      - 最小: {df['speed'].min():.2f}
      - 最大: {df['speed'].max():.2f}
    
    航向统计 (度):
      - 平均: {df['course'].mean():.2f}
      - 标准差: {df['course'].std():.2f}
    """
    
    ax6.text(0.1, 0.5, stats_text, 
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('船舶AIS数据可视化分析', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 可视化图表已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_multiple_scenarios(df_dict: dict, 
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True):
    """
    可视化多个场景的轨迹对比
    
    Args:
        df_dict: 场景名称到DataFrame的字典
        save_path: 保存路径（可选）
        show_plot: 是否显示图表
    """
    fig = plt.figure(figsize=(16, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. 所有场景轨迹对比
    ax1 = plt.subplot(2, 2, 1)
    for idx, (name, df) in enumerate(df_dict.items()):
        color = colors[idx % len(colors)]
        ax1.plot(df['longitude'], df['latitude'], 
                'o-', color=color, label=name, 
                alpha=0.6, markersize=2, linewidth=1)
        ax1.plot(df['longitude'].iloc[0], df['latitude'].iloc[0], 
                'o', color=color, markersize=10)
    
    ax1.set_xlabel('经度 (Longitude)', fontsize=11)
    ax1.set_ylabel('纬度 (Latitude)', fontsize=11)
    ax1.set_title('多场景轨迹对比', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 速度对比
    ax2 = plt.subplot(2, 2, 2)
    for idx, (name, df) in enumerate(df_dict.items()):
        color = colors[idx % len(colors)]
        ax2.plot(range(len(df)), df['speed'], 
                '-', color=color, label=name, alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('时间步', fontsize=11)
    ax2.set_ylabel('速度 (节)', fontsize=11)
    ax2.set_title('速度对比', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度分布对比
    ax3 = plt.subplot(2, 2, 3)
    speed_data = [df['speed'] for df in df_dict.values()]
    ax3.boxplot(speed_data, tick_labels=list(df_dict.keys()))
    ax3.set_ylabel('速度 (节)', fontsize=11)
    ax3.set_title('速度分布箱线图', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15)
    
    # 4. 统计对比表
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = "场景统计对比\n" + "="*50 + "\n\n"
    for name, df in df_dict.items():
        stats_text += f"{name}:\n"
        stats_text += f"  样本数: {len(df):,}\n"
        stats_text += f"  平均速度: {df['speed'].mean():.2f} 节\n"
        stats_text += f"  速度范围: {df['speed'].min():.2f} ~ {df['speed'].max():.2f} 节\n"
        stats_text += f"  轨迹长度: {len(df)} 个点\n\n"
    
    ax4.text(0.1, 0.5, stats_text,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('多场景船舶轨迹对比分析', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 场景对比图已保存到: {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 生成示例数据
    print("="*60)
    print("           船舶AIS示例数据生成程序")
    print("="*60)
    
    df_combined, scenario_dict = generate_multiple_scenarios()
    
    # 可选：添加异常值（用于测试数据清洗功能）
    # df_with_anomalies = add_anomalies(df_combined, anomaly_rate=0.02)
    # df_with_anomalies.to_csv('data/raw/ship_data_with_anomalies.csv', index=False)
    
    print("\n" + "=" * 60)
    print("开始生成数据可视化...")
    print("=" * 60)
    
    # 创建可视化输出目录
    vis_output_dir = 'data/visualizations'
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 1. 可视化合并后的完整数据
    print("\n1. 生成完整数据集可视化...")
    visualize_trajectory_data(
        df_combined,
        save_path=os.path.join(vis_output_dir, 'combined_trajectory_analysis.png'),
        show_plot=False  # 不显示，只保存
    )
    
    # 2. 可视化多场景对比
    print("2. 生成多场景对比可视化...")
    visualize_multiple_scenarios(
        scenario_dict,
        save_path=os.path.join(vis_output_dir, 'scenario_comparison.png'),
        show_plot=False
    )
    
    # 3. 为每个场景单独生成可视化
    print("3. 生成各场景单独可视化...")
    for scenario_name, df_scenario in scenario_dict.items():
        safe_name = scenario_name.replace(' ', '_')
        visualize_trajectory_data(
            df_scenario,
            save_path=os.path.join(vis_output_dir, f'{safe_name}_analysis.png'),
            show_plot=False
        )
        print(f"   ✓ {scenario_name}可视化完成")
    
    print("\n" + "=" * 60)
    print("✅ 示例数据生成完成！")
    print("=" * 60)
    print("\n📁 生成的文件:")
    print(f"   - 数据文件: data/raw/ship_data.csv")
    print(f"   - 测试数据: data/raw/ship_data_test.csv")
    print(f"   - 可视化图表: {vis_output_dir}/")
    print(f"     • combined_trajectory_analysis.png (完整数据分析)")
    print(f"     • scenario_comparison.png (场景对比)")
    print(f"     • 标准航行_analysis.png")
    print(f"     • 高速航行_analysis.png")
    print(f"     • 低速航行_analysis.png")
    
    print("\n📊 数据摘要:")
    print(f"   - 总样本数: {len(df_combined):,}")
    print(f"   - 场景数量: {len(scenario_dict)}")
    print(f"   - 特征维度: {list(df_combined.columns)}")
    
    print("\n🚀 下一步:")
    print("   1. 查看可视化图表了解数据分布")
    print("   2. 查看生成的数据: data/raw/ship_data.csv")
    print("   3. 编辑配置文件: config/config.yaml")
    print("   4. 开始训练: python train.py")
    
    print("\n" + "=" * 60)

