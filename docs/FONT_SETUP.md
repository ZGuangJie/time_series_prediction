# Matplotlib 中文字体配置指南

## 问题说明

如果您在运行程序时看到类似以下的警告：

```
UserWarning: Glyph 35757 (\N{CJK UNIFIED IDEOGRAPH-8BAD}) missing from font(s) DejaVu Sans.
```

这是因为 matplotlib 默认使用的字体不支持中文字符。

## 解决方案

### 自动配置（已实现）

项目已经在 `utils/visualization.py` 中自动配置了中文字体支持，会根据操作系统自动选择合适的中文字体：

- **Windows**: Microsoft YaHei（微软雅黑）、SimHei（黑体）
- **macOS**: Arial Unicode MS、STHeiti（华文黑体）
- **Linux**: WenQuanYi Micro Hei（文泉驿微米黑）

**大多数情况下不需要任何额外操作**，警告会自动消失。

---

## 如果仍有问题

### 方法1：安装缺失的字体（Windows）

如果您的系统缺少中文字体，可以安装：

1. **微软雅黑**（推荐，Windows 7+自带）
   - 已预装在 Windows 系统中

2. **下载并安装字体**
   - 下载 SimHei 字体文件
   - 双击 `.ttf` 文件安装
   - 重启 Python 程序

### 方法2：清除 matplotlib 缓存

有时需要清除 matplotlib 的字体缓存：

```bash
# Windows
del %USERPROFILE%\.matplotlib\fontlist-*.json

# Linux/macOS
rm ~/.matplotlib/fontlist-*.json
```

然后重新运行程序。

### 方法3：手动指定字体

编辑 `utils/visualization.py` 文件，在开头添加：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
```

### 方法4：查看系统可用字体

运行以下代码查看您系统上可用的中文字体：

```python
import matplotlib.font_manager as fm

# 列出所有字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(ch in f for ch in ['黑', 'Song', 'Hei', 'YaHei', 'Kai'])]

print("可用的中文字体：")
for font in chinese_fonts:
    print(f"  - {font}")
```

---

## 验证配置

运行以下测试脚本验证中文显示：

```python
import matplotlib.pyplot as plt
import matplotlib

print("当前字体设置：", matplotlib.rcParams['font.sans-serif'])

plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9], 'o-', label='测试数据')
plt.xlabel('横轴（中文）')
plt.ylabel('纵轴（中文）')
plt.title('中文显示测试')
plt.legend()
plt.savefig('test_chinese.png', dpi=150, bbox_inches='tight')
plt.close()

print("测试图片已保存: test_chinese.png")
print("请检查图片中的中文是否正常显示")
```

如果保存的图片中中文显示正常，说明配置成功！

---

## 常见问题

### Q1: 中文显示为方块

**原因**：系统缺少指定的中文字体

**解决**：
1. 检查系统是否安装了中文字体
2. 使用方法4查看可用字体
3. 手动指定一个存在的字体

### Q2: 负号显示为方块

**原因**：使用中文字体后，负号可能无法正确显示

**解决**：添加配置
```python
plt.rcParams['axes.unicode_minus'] = False
```

### Q3: 警告仍然存在但不影响使用

**说明**：
- 警告不影响程序运行
- 图片中的中文会正常显示
- 可以忽略这些警告（仅提示信息）

---

## Linux 系统额外配置

如果您使用 Linux 且没有中文字体，可以安装：

### Ubuntu/Debian
```bash
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
```

### CentOS/RHEL
```bash
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
```

### Arch Linux
```bash
sudo pacman -S wqy-microhei wqy-zenhei
```

安装后清除缓存并重启程序。

---

## 项目已完成的配置

✅ 自动检测操作系统
✅ 自动选择合适的中文字体
✅ 解决负号显示问题
✅ 跨平台支持（Windows/macOS/Linux）

**大部分用户无需额外配置！**

---

如有其他问题，请查看 [README.md](../README.md) 或提交 Issue。

