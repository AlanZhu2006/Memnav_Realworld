# 2026-09-07：在线历史几何复用更新

## 改了什么

CEC 选中历史 anchor 后，直接读取观察该帧时保存的 LingBot depth/confidence，
不再为首次查询重放整段历史以取得深度。同一次在线几何估计服务当前 NavDP 的
单目深度输入和以后 CEC 的历史定位。

DINO、SuperPoint/LightGlue、PnP、certificate、2.5 m bearing residual 均保留。
这是运行时实现升级，不是新增训练或 learned relocalizer；也不是近目标／到达修复。

## 新配置及生效方式

`deployment/config/system.json` 的 `stack.cec` 新增：

```json
"historical_depth_source": "online_history"
```

`eager_depth_cache` 继续是 false。新解析出的 immutable config 会记录该值；
启动脚本同时传给外部 MemNav 与 CEC Hub，reset 会核对双方一致。
缺该字段的历史 resolved config 按 canonical 解释，不修改历史收据或历史结果。

新参数影响常驻模型兼容性签名。当前已加载的旧进程不热切换；下次正常启动新实验
时只按原有流程重新加载自管的 parked 模型，active／非本栈服务不会被本次同步替换。
不要为了启用这项优化在正在行走的 episode 中重置模型。

确认方式：

- resolved config：`cec.historical_depth_source`；
- MemNav reset：`certified_relocalization.default_reference_depth_source`；
- Hub `/healthz`：`cec_historical_depth_source`，只有 initialized=true 才表示已核对上游；
- 计划收据：`cec_relocalization_trace.reference_depth_source`；
- MemNav `/resident/status`：来源、历史深度帧数、CPU 缓存字节数。

这些是只读状态入口，不是运动启动命令。

## 与此前几项更新的区别

| 更新 | 消除的重复工作 | 本次是否改变 |
|---|---|---|
| 当前帧 flow-depth 复用 | 相同当前 RGB 的重复 depth head | 保留 |
| resident 模型 park/reuse | 反复加载模型权重 | 保留 |
| 本次历史深度复用 | 首次查询历史 anchor 的几何重放 | 新增 |

模型常驻和本次缓存都不取消 reset、sealed Survey 的 RGB 重放、目标 SHA 绑定和
独立 query-start FIFO 初始化。不要求重新采集 Survey，也不传感器深度进策略。

CPU depth/conf 缓存位于 RTX 主机，不放进 Jetson，不通过 SSH 传整段深度。每帧
约 2.05 MiB，随历史长度增长；episode reset/park 会清空运行时缓存，不删除原始
RGB、sealed dataset、MCAP 或任何实验记录。此版本尚未实现无限长存储或淘汰策略。

## 仿真证据及不能宣称的内容

本机 4 场景／8 queries／16 rollouts：

| 指标 | canonical replay | online history |
|---|---:|---:|
| Revisit SR | 3/4 | 3/4 |
| Novel SR | 0/4 | 0/4 |
| Revisit 平均实际 SPL | 0.66303 | 0.65403 |
| Revisit 首次 certificate 阶段中位耗时 | 20.330 s | 0.126 s |

4 条 Novel 全程拒绝，两臂轨迹相同；3 条 Revisit 成功，另 1 条两臂都 stuck。
初始在线 depth/conf CPU cache 397–815 MiB。0.126 s 不包含整套 DINO probe、NavDP、
通信和机器人运动，不能写成真机端到端延迟。旧论文和旧真机结果不是新版本结果。

**代码／配置同步不等于真机导航验证。** 这次不启动 ROS 栈，不 enable、不解除
estop，不下发速度，不改 Jetson 控制器、急停、相机参数、到达条件或用户现有修改。
当前 baseline / CEC 命名和运动授权流程继续遵守 `AGENTS.md`。

## 复现旧实现

新实验若需要旧版，先把 `stack.cec.historical_depth_source` 显式设为 `canonical`，
然后重新 resolve 为新的运行配置；不要手改旧 immutable config。研究仓库底层
MemNav CLI 默认仍是 canonical，旧仿真脚本不会因为本次更新自动换方法。

## 验证与同步记录

本次以语法／导入检查、现有配置解析工具、已有 RGB 的无运动模型接口检查和双机
文件比对验证，不新增或运行真机仓库 unit tests。完整实测和同步收据记录在
研究工作区 `.diagnostics/cec_online_history_promotion_20260907_AwHz3q/`。

已完成的无运动模型检查使用实际真机仓库 resident wrappers 与 CEC Hub，另起
21560/21561 私有服务，回放已有仿真 240 帧 RGB：

- 新公共入口返回 `online_history`，支持目标通过 certificate，`replayed_frames=0`；
- NavDP 返回有限值的 `[1,24,3]` 轨迹，未交给任何执行器；
- 该次完整 GPU 侧 Hub 规划调用约 0.931 s，certificate 约 143.5 ms；这是一个
  录制输入的单例，不能替代真机相机／SSH／ROS 端到端测量；
- 233 帧 depth/conf 共 500,155,936 bytes，episode release 后为 0；
- 私有服务已退出，原 18888/8888 常驻服务及运动状态未操作。

Jetson 新配置已用实际现存目标文件成功 resolve，并通过 `verify --site jetson`；
新 ID 为 `92374c01ed5fd56963bb2f96ba8a1b033e057fe960b48642849c4da361093894`。
这是静态配置检查，未安装为运行中的 formal goal，也未启动实验。
本机缺少示例目标 `goal_d_novel_frame_125.jpg`，所以该次新配置在 Jetson 解析；
没有替换目标或把本地仿真图假装成真机目标。旧 resolved config 的 shell 导出仍为 canonical。
