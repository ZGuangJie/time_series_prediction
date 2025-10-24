"""
测试matplotlib中文字体配置
运行此脚本验证中文是否正常显示
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

print("=" * 60)
print("Matplotlib 中文字体配置测试")
print("=" * 60)

# 显示系统信息
print(f"\n操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {platform.python_version()}")
print(f"Matplotlib版本: {matplotlib.__version__}")

# 配置中文字体（与项目相同的配置）
system = platform.system()

if system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'STSong']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']

matplotlib.rcParams['axes.unicode_minus'] = False

print(f"\n当前字体配置: {matplotlib.rcParams['font.sans-serif']}")

# 查找系统中可用的中文字体
print("\n系统中可用的中文字体:")
all_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_keywords = ['黑', 'Song', 'Hei', 'YaHei', 'Kai', 'Ming', 'STHeiti', 'Arial Unicode']
chinese_fonts = [f for f in set(all_fonts) if any(keyword in f for keyword in chinese_keywords)]

if chinese_fonts:
    for font in sorted(chinese_fonts)[:10]:  # 只显示前10个
        print(f"  ✓ {font}")
else:
    print("  ⚠️  未找到常见中文字体")

# 创建测试图
print("\n正在生成测试图...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 子图1：简单测试
ax1 = axes[0]
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
ax1.plot(x, y, 'o-', linewidth=2, markersize=8, label='测试数据')
ax1.set_xlabel('横轴（时间）', fontsize=12)
ax1.set_ylabel('纵轴（数值）', fontsize=12)
ax1.set_title('中文显示测试', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：各种中文字符测试
ax2 = axes[1]
test_texts = [
    '训练损失',
    '验证损失',
    '评估指标',
    '预测结果',
    '模型性能'
]
y_pos = list(range(len(test_texts)))
ax2.barh(y_pos, [i+1 for i in range(len(test_texts))], color='steelblue')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(test_texts, fontsize=11)
ax2.set_xlabel('数值', fontsize=12)
ax2.set_title('中文标签测试', fontsize=14, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()

# 保存图片
output_file = 'test_chinese_font_result.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 测试图片已保存: {output_file}")

# 验证结果
print("\n" + "=" * 60)
print("验证步骤:")
print("=" * 60)
print(f"1. 打开图片: {output_file}")
print("2. 检查中文是否正常显示（不是方块）")
print("3. 如果显示正常，说明配置成功！")
print("4. 如果显示为方块，请参考 docs/FONT_SETUP.md")

# 检查是否有警告
print("\n如果上面没有字体警告，说明配置成功！")
print("如果有警告但图片中文正常显示，可以忽略警告。")
print("=" * 60)

