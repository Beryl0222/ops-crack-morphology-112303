# 缺测像素上的跨层形态学与缺陷连通判定

`data/record.json` 是 缺测像素上的跨层形态学与缺陷连通判定 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

窗口是矩阵切片，未提供的格点使用 ranges。lower 和 upper 同形且包含端点。工件域下标为层、行、列。结构元素列出腐蚀的相对输入偏移，膨胀偏移按其反射集合。跨层边从本层下标到下一层相对下标，连接可双向遍历。faces 给出起止点集合。probe 的四列是层、行、列和上游梯度。闭运算输出的可训练区域要求所有参与判定的实际输入支撑已确定。alternate_conflict 是可独立添加的一份窗口记录。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m crack_morphology_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
