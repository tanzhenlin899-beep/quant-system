# 多因子选股与波动率目标择时系统

课程：量化交易（周三上午）　组号：（待填）　提交日期：2025-12

一个可复现的 A 股量化投资系统，包含数据接入、因子计算、信号生成、回测引擎与绩效分析五个模块。策略核心与回测逻辑均为自行实现，不依赖第三方回测框架。支持离线模拟数据、baostock 与 akshare 三种数据源，提供命令行与 Web 两种使用方式。

回测结果与策略分析见[项目技术报告](docs/技术报告.md)。

---

## 1. 运行环境

Python 3.11（3.10 及以上亦可）。依赖见 `requirements.txt`，测试与静态检查依赖见 `requirements-dev.txt`，可选组件见 `requirements-web.txt` 与 `requirements-frameworks.txt`。

### 1.1 安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

PowerShell 若提示禁止运行脚本，执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 1.2 运行

```powershell
python main.py
```

首次运行按 `config.py` 中的数据源下载行情并缓存至 `data/cache/`，此后直接读取本地 parquet 文件。运行结束后，`outputs/` 生成以下文件：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 样本内、样本外、全样本指标表，附数据源、换手率、交易成本、成交笔数 |
| `nav.png` | 策略与基准的归一化净值曲线，含买卖点标注与回撤子图 |
| `ic.png` | 各因子 RankIC 逐日走势与累计 IC |
| `layered.png` | 因子分层累计净值 |
| `sensitivity.png` | 参数敏感性热力图 |
| `trades.csv` | 逐笔成交记录（日期、代码、方向、股数、价格、金额、费用） |

图表同时复制至 `docs/figures/`，该目录纳入版本控制，供技术报告引用。

### 1.3 命令行参数

| 参数 | 说明 |
|---|---|
| `--mode {offline,akshare,baostock}` | 指定数据源，覆盖 `config.py` 中的设置 |
| `--no-timing` | 关闭波动率目标择时，用于消融对比 |
| `--quick` | 跳过参数敏感性扫描，缩短运行时间 |

### 1.4 Web 界面（可选）

```powershell
pip install -r requirements-web.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。页面提供数据源、股票池规模、持仓只数、缓冲区、调仓周期、择时开关、目标波动共七项参数；提交后由后端重新加载数据、计算因子并执行撮合，返回本次请求的指标与净值曲线，不使用预计算结果。

数据源可在页面上直接切换（baostock / offline / akshare），选择时页面会显示该数据源在本地的缓存进度。akshare 首次运行需下载全部标的，耗时 1 至 3 分钟，且受 IP 限流影响，建议先在命令行执行一次 `python main.py --mode akshare` 建立缓存。

页面加载时只读取默认参数、不触发回测，参数可先调整再提交；运行期间界面保持可操作，可继续修改参数或取消本次请求。除绩效结果外，页面另设「运行明细」标签页，展示最后交易日的目标持仓、最近 60 笔成交、打分前 10 名的各因子取值，以及载入数据、计算因子、生成信号、回测撮合、计算指标各阶段耗时，便于核对策略行为。

前端为单个 HTML 文件，图表以原生 SVG 绘制，不引用外部资源。

接口：`GET /api/defaults`、`POST /api/run`、`GET /api/health`、`GET /api/docs`。

---

## 2. 目录结构

```
quant/
├── main.py                  程序入口，串联完整流程
├── config.py                全部可调参数
├── requirements*.txt        依赖清单
├── data/                    数据层
│   ├── loader.py              数据源调度、akshare 接口、缓存
│   ├── baostock_loader.py     baostock 接口，输出停牌与 ST 标记
│   ├── synthetic.py           离线模拟数据生成
│   └── universe.py            股票池过滤与可交易掩码
├── factors/                 因子层
│   ├── registry.py            因子注册表
│   ├── builtin.py             内置因子
│   ├── preprocess.py          去极值、标准化、中性化
│   └── composite.py           多因子合成
├── strategy/                策略层
│   ├── base.py                策略基类，约定信号时序
│   ├── multi_factor.py        多因子 TopN 选股
│   └── timing.py              波动率目标择时
├── backtest/                回测层
│   ├── engine.py              撮合内核
│   ├── cost.py                费用模型
│   └── metrics.py             绩效指标
├── analysis/                分析层
│   ├── ic.py                  RankIC、分层回测
│   └── plots.py               图表输出
├── adapters/                第三方引擎适配器
│   └── backtrader_engine.py   以 backtrader 复算同一信号，用于交叉验证
├── web/                     Web 界面
│   ├── app.py                 FastAPI 服务
│   └── static/index.html      单页前端
├── tools/                   辅助脚本
│   ├── check_data.py          数据源连通性检查
│   ├── compare_engines.py     双引擎净值对比
│   └── sweep_params.py        参数扫描
├── tests/                   测试
├── docs/                    技术报告与图表
└── outputs/                 运行产物，不纳入版本控制
```

---

## 3. 系统设计

### 3.1 数据流

```
config.py 参数
      │
      ▼
data/     行情面板（date × code）＋ 可交易掩码 ＋ 停牌/ST 标记
      │
      ▼
factors/  因子计算 → MAD 去极值 → 横截面标准化 → 加权合成
      │
      ▼
strategy/ TopN 选股 → 等权 → 缓冲区换手控制 → 波动率目标缩放
      │
      ▼
backtest/ t+1 开盘撮合，逐笔扣除费用，屏蔽停牌与涨跌停
      │
      ▼
analysis/ 绩效指标、净值曲线、因子检验
```

### 3.2 模块职责

| 模块 | 输入 | 输出 | 关键设计 |
|---|---|---|---|
| 数据层 | 配置 | 面板矩阵、掩码 | 统一为 date × code 的 DataFrame，上游数据源差异被隔离 |
| 因子层 | 面板 | 打分矩阵 | 因子以注册表管理，新增因子不影响其它模块 |
| 策略层 | 打分、掩码 | 目标权重矩阵 | 输出统一为权重，便于替换与对比 |
| 回测层 | 权重、面板、掩码 | 净值、成交记录 | 按份额记账，现金约束与不可交易状态真实生效 |
| 分析层 | 净值、因子 | 指标、图表 | 与回测解耦，可单独调用 |

各模块之间仅通过 DataFrame 传递数据，不传递自定义对象，因此任一模块可单独替换或单独测试。

---

## 4. 策略定义

### 4.1 因子

因子原始值经 MAD 去极值（中位数 ± 3 × 1.4826 × MAD 截断）与横截面 z-score 标准化后按权重合成。全部因子均已调整为数值越大越优。

| 因子 | 定义 | 权重 | 说明 |
|---|---|---|---|
| `mom20` | `close.pct_change(20)` | 0.35 | 20 日动量 |
| `mom60` | `close.pct_change(60)` | 0.20 | 60 日动量 |
| `rev5` | `-close.pct_change(5)` | 0.15 | 5 日反转 |
| `vol20` | `-std(ret, 20)` | 0.30 | 20 日低波动 |

新增因子时，在 `factors/builtin.py` 中定义函数并添加 `@register` 装饰器，再于 `config.py` 中赋予权重，其余模块无需改动。

### 4.2 组合构建

1. 剔除当日不可交易的股票（停牌、一字涨跌停、上市不足 60 个交易日）；
2. 按合成打分排序，选取前 10 只等权持有；
3. 缓冲区：已持仓股票只要仍位于前 15 名内即继续持有，掉出后替换，以降低换手；
4. 每 10 个交易日调仓一次；
5. 单只股票调仓金额低于总市值 0.2% 时不予执行，避免碎单。

### 4.3 风险控制

以过去 20 个交易日的组合已实现波动为估计，将仓位缩放至目标年化波动 18%，仓位上限 100%（不使用杠杆）。波动估计使用前一交易日数据，避免使用当期信息。

---

## 5. 回测假设

| 项目 | 设定 |
|---|---|
| 信号时点 | t 日收盘后生成，t+1 日开盘成交；权重整体 `shift(1)` |
| 成交价格 | 次日开盘价，另计滑点 |
| 佣金 | 万分之 2.5，双边，单笔最低 5 元 |
| 印花税 | 万分之 5，仅卖出方向 |
| 过户费 | 万分之 0.1，双边 |
| 滑点 | 千分之 1 |
| 停牌 | 不可成交，维持原仓位；停牌期间以最近已知价格估值 |
| 一字涨跌停 | 开盘价触及涨跌停即视为不可成交 |
| 涨跌停幅度 | 主板 10%，主板 ST 5%，创业板与科创板 20%，北交所 30% |
| 持仓 | 等权，整数股，现金不足时按可用资金缩减 |
| 复权 | 后复权，收益率不含除权跳空 |
| 基准 | 沪深 300 指数，价格指数，不含分红 |

停牌期间若以缺失值（NaN）直接估值，会使持仓市值归零并产生虚假回撤。引擎维护逐股最近已知价格用于估值，该行为由 `tests/test_suspension.py` 覆盖。

---

## 6. 数据源

系统支持三种数据源，通过 `config.py` 中 `data.mode` 或命令行参数切换。

| 模式 | 说明 |
|---|---|
| `offline` | 内置模拟数据，不联网，用于验证流程完整性 |
| `baostock` | 本项目默认。不限流，原生提供 `tradestatus`（停牌）与 `isST`（ST）字段 |
| `akshare` | 东方财富接口，速度快，但按 IP 限流 |

baostock 的原生字段使停牌识别与 ST 剔除均为逐日判断，无需维护静态名单。数据源连通性可通过 `python tools/check_data.py` 检查。

两处需要说明的差异：

- 复权基准不同。同一标的在 akshare 与 baostock 的后复权价格不可直接比较，两套数据分别缓存于 `data/cache/` 与 `data/cache/bs/`，不混用。
- 成交量单位不同。akshare 以手（100 股）计，baostock 以股计。对量价类因子仅相差常数倍数，不改变横截面排序。

---

## 7. 测试与验证

```powershell
python tests\test_engine.py        # 回测引擎行为测试，11 项
python tests\test_suspension.py    # 停牌估值专项测试
python tests\test_baostock.py      # 数据接入解析测试，4 项
python tests\test_ic.py            # RankIC 计算正确性测试，4 项
```

覆盖的假设包括：信号不存在前视偏差、交易费用计算正确、一字涨停不可成交、ST 股涨跌停幅度为 5%、绩效指标与手算一致、策略与基准相同时超额为零、持仓数超过股票数时权重归一、全市场不可交易时净值不变、单日数据不导致崩溃、因子预热期全为缺失值时空仓、现金不足时不成交、停牌不影响净值估值。

此外，`adapters/backtrader_engine.py` 将同一套目标权重交由 backtrader 复算，与自研引擎逐日对比，用于检验撮合逻辑。两者在估值口径上存在已知差异（自研引擎以当日开盘价估值后调仓，backtrader 以上一收盘价估值），对比结果见技术报告第 7.1 节。

```powershell
pip install -r requirements-frameworks.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python tools/compare_engines.py 20
```

静态检查：

```powershell
pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ruff check . --exclude .venv,tmp
mypy main.py analysis/plots.py web/app.py --ignore-missing-imports
```

---

## 8. AI 使用说明

本项目在开发过程中使用 AI 编程助手辅助完成工程骨架搭建、模块划分与部分工具函数编写。代码未引用、克隆或改编任何 GitHub 开源仓库；所采用的方法（多因子加权、MAD 去极值、横截面标准化、RankIC 检验、TopN 等权、缓冲区换手控制、波动率目标择时）均为量化领域的公开通用方法。

因子选择与经济逻辑、回测假设的设定、参数与样本内外划分、结果归因与局限分析由小组成员完成。全部回测数值均通过 `python main.py` 在本地实际运行产生，可复现。

---

## 9. 免责声明

本项目仅用于课程学习与研究。所有结果基于历史数据，不构成任何投资建议。
