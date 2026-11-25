# 缺测像素上的跨层形态学与缺陷连通判定

`data/record.json` 是本工程使用的本地合成交换样本。数组按 JSON 的下标顺序排列。
坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，
对应同一输入协议的部署规格。

窗口是矩阵切片，未提供的格点使用 `ranges`。`lower` 和 `upper` 同形且包含端点。
工件域下标为层、行、列。结构元素列出**腐蚀**的相对输入偏移（输出取输入 `r+offset`），
膨胀偏移按其反射集合。跨层边从本层下标到下一层相对下标，连接可双向遍历。
`faces` 给出起止点集合。`probe` 的四列是层、行、列和上游梯度。
闭运算输出的可训练区域要求所有参与判定的实际输入支撑已确定。
`alternate_conflict` 是可独立添加的一份窗口记录。

## 工程包 `cracklib`

纯 NumPy 实现，面向带缺孔的区间强度图，图像判定与相机训练共用同一套算子。

| 模块 | 职责 |
| --- | --- |
| `cracklib.model` | `Cell`、`Window`、`LayerSpec`、`ProcessConfig`、`ResampleTarget` 等数据模型 |
| `cracklib.store` | 窗口装配、重复观测的约束交集、冲突局部阻断、整窗替换、分层物化与溯源 |
| `cracklib.morphology` | 区间腐蚀/膨胀/闭运算原语、halo 分块调度器与 scratch 内存计量 |
| `cracklib.graph` | 肯定/可能双连通图、四邻接+配置跨层边、贯通判定与补测见证集 |
| `cracklib.training` | 确定响应区域的可训练性诊断、确定性极值路由与上游梯度 |
| `cracklib.pipeline` | `CrackEngine` 门面：装配、重算、查询、训练的一条流水线 |
| `cracklib.io` | 交换记录 ↔ 工程对象的装配，支持换配方复用 |

### 区间形态学口径

设输入格点区间为 `[lo, hi]`，原图外取工艺声明常量 `outside`：

* 腐蚀（footprint 偏移上取 min）：`erode_lo = min(lo)`，`erode_hi = min(hi)`
* 膨胀（反射偏移上取 max）：`dil_lo = max(lo)`，`dil_hi = max(hi)`
* 闭运算 = 膨胀(腐蚀())，响应的 `lower/upper` 始终可查询且保持 `lo <= hi`

分块只在**原图外边界**填工艺常量；块与块的交界不引入任何边界——
闭运算条带按完整支撑半径取 halo，裁掉 halo 后与整域计算逐格点一致
（见 `tests/test_chunks.py` 的多 band 尺寸对拍）。机台额外内存由
`scratch_bytes` 约束，`AllocationTracker` 报告峰值；极小预算下可切两遍调度，
调用者始终不需要先拼出整个张量。

### 贯通判定与补测

闭运算后按层阈值生成两张图：

* **肯定图**：响应下界已达阈值（`lower >= threshold`）
* **可能图**：响应上界仍达阈值（`upper >= threshold`，包含肯定图）

对用户选择的起止工艺面，`CrackEngine.penetration()` 返回四种结论：

* `definite`：肯定图连通，可放行
* `impossible`：可能图也不连通，肯定不贯通
* `needs_measurement`：可能图连通但肯定图不连通，返回一条可能路径，
  以及该路径上不确定响应格点的闭运算支撑中、仍未实测确定的**少数**原始格点
  （不会把全部缺测像素塞回给操作员）
* `blocked`：查询面格点被观测冲突阻断；冲突只影响局部，独立区域仍可查询

窗口替换推进观测版本号，响应与贯通缓存自动失效重算，不会拿旧判定继续放行。

### 训练梯度

`CrackEngine.training_plan(probe)` 只对所有实际输入支撑均已实测确定
（区间宽度为 0、无冲突、不含原图外常量）的响应格点开放梯度：

* 原始依赖仍有区间不确定性时，返回 `trainable=False` 与具体原因格点；
* 极值按值选择，完全相同的实测强度按全局格点 C 序裁决，前向选择与反向路由一致；
* 同一物理格点在重叠窗口中的贡献只交给唯一输入身份，梯度按全局格点累积；
* 边界常量没有输入身份，因而没有输入梯度。

## 读取样本与运行测试

使用 Python 3.11 或 3.12，在虚拟环境安装锁定依赖。

```sh
python -m pip install -r requirements.txt
python -m crack_morphology_input
python -m unittest discover -s tests -v
```

测试包含：形态学原语的暴力参照对拍、分块/两遍调度等价性、观测交集与冲突隔离、
三种贯通判定叙事、补测见证集的有界性、训练梯度的有限差分验证，以及 scratch 上限。

### 快速上手

```python
from cracklib.pipeline import CrackEngine
from crack_morphology_input import SAMPLE

engine = CrackEngine.from_record(SAMPLE)
result = engine.penetration()
print(result.verdict)                       # needs_measurement / definite / ...
print(result.to_dict()["resample"])         # 值得补采的原始格点与溯源
plan = engine.training_plan(engine._probe)  # 不可训练时给出原因格点
```
