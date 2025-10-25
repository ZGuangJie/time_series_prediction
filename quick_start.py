"""
快速启动脚本
一键完成数据生成、模型训练和预测的完整流程
"""
import os
import sys
import subprocess

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} 完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}\n")
        return False

def main():
    """主流程"""
    print("\n" + "🚢" * 30)
    print("船舶位置预测系统 - 快速启动")
    print("🚢" * 30)
    
    # 步骤1: 生成示例数据
    print_section("步骤 1/3: 生成示例数据")
    if not run_command("python generate_sample_data.py", "生成示例数据"):
        print("❌ 数据生成失败，请检查错误信息")
        return
    
    # 步骤2: 训练模型
    print_section("步骤 2/3: 训练模型")
    print("⚠️  提示: 训练可能需要几分钟到几十分钟，取决于数据量和硬件配置")
    print("    如果使用CPU训练，建议先修改 config/config.yaml 中的 num_epochs 为较小值（如20）\n")
    
    user_input = input("是否开始训练？(y/n): ")
    if user_input.lower() != 'y':
        print("跳过训练步骤")
        return
    
    if not run_command("python train.py --config config/config.yaml", "训练模型"):
        print("❌ 模型训练失败，请检查错误信息")
        return
    
    # 步骤3: 预测
    print_section("步骤 3/3: 模型预测")
    
    # 检查是否存在测试数据
    test_data_path = "data/raw/ship_data_test.csv"
    if not os.path.exists(test_data_path):
        test_data_path = "data/raw/ship_data.csv"
    
    predict_cmd = (
        f"python predict.py "
        f"--config config/config.yaml "
        f"--model_path checkpoints/best_model.pth "
        f"--mode file "
        f"--input_file {test_data_path} "
        f"--output_file results/predictions.csv "
        f"--visualize"
    )
    
    if not run_command(predict_cmd, "执行预测"):
        print("❌ 预测失败，请检查错误信息")
        return
    
    # 完成
    print_section("✅ 全部完成！")
    print("结果文件:")
    print(f"  📊 训练历史: results/training_history.png")
    print(f"  📈 预测结果: results/predictions.csv")
    print(f"  🗺️  轨迹对比: results/trajectory_comparison.png")
    print(f"  📉 预测对比: results/predictions_comparison.png")
    print(f"  💾 最佳模型: checkpoints/best_model.pth")
    print(f"  📝 训练日志: logs/training.log")
    
    print("\n查看TensorBoard:")
    print("  tensorboard --logdir=runs/")
    
    print("\n" + "🎉" * 30)
    print("感谢使用船舶位置预测系统！")
    print("🎉" * 30 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

