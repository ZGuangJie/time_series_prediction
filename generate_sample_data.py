"""
生成示例船舶AIS数据
用于演示和测试
"""
import numpy as np
import pandas as pd
import os

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
    
    return df_combined


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


if __name__ == '__main__':
    # 生成示例数据
    df = generate_multiple_scenarios()
    
    # 可选：添加异常值（用于测试数据清洗功能）
    # df_with_anomalies = add_anomalies(df, anomaly_rate=0.02)
    # df_with_anomalies.to_csv('data/raw/ship_data_with_anomalies.csv', index=False)
    
    print("\n" + "=" * 50)
    print("✅ 示例数据生成完成！")
    print("=" * 50)
    print("\n下一步:")
    print("1. 查看生成的数据: data/raw/ship_data.csv")
    print("2. 编辑配置文件: config/config.yaml")
    print("3. 开始训练: python train.py")

