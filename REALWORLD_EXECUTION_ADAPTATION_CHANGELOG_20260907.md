# 真机执行适配变更说明（2026-09-07）

## 合并目的与范围

本次把真机的感知—规划—执行反馈方式向仿真对齐，处理 004/006 暴露的执行接口问题。
**不是更换 CEC/NavDP 模型，不是新的训练方法，也不是已经验证成功的弧形转向方案。**

代码来自 `feat/sim-aligned-receding-horizon-20260907`，基于 `95afe64`：

- `54da70e`：保留 Jetson 现场已存在并经过审计的六个执行相关文件修正。
- `e45b302`：滚动重规划、独立几何观察、低 critic 意图处理、显式近目标实验。
- 本次后续提交：静止延迟工具与结果，以及纯旋转/高度尺度问题的说明。

合并范围是 **AlanZhu2006/MemNav-RealWorld**。不修改研究仓库中的仿真协议和论文表格，
不覆盖已有正式 pair、目标、Survey 或现场未提交工作。GitHub 合并不等于双机部署，
也不会自行启动电机。原 GPU/Jetson 工作区在本次隔离验证时仍为 `95afe64` 加现场修改。

## 1. 真机完整链路

```text
第一次：手柄 Survey → 因果 RGB 历史封存
第二次：加载同一历史 + 安装独立 ImageGoal + 当前起点 RGB
    │
Jetson 实时相机、曝光时间、Go2 位姿/IMU
    ├─ RGB → 单一 HTTP worker → RTX LingBot 因果状态
    │                            ├─ 规划帧：单目深度 + CEC → frozen NavDP
    │                            └─ 非规划/转身帧：只更新当前几何
    ├─ Go2 位姿 → 曝光时刻坐标补偿 → 当前位姿跟踪新局部路径
    ├─ D435i 实测深度 → 仅本地近障碍安全
    └─ 当前 RGB + ImageGoal → 独立视觉到达判定
                                  ↓
                    Jetson 限速/反馈/急停 → Unitree 执行
```

CEC 仍使用 DINO 历史检索、SuperPoint/LightGlue、LingBot 深度与 PnP 证书。
长期历史认证通过后，方向投影成默认 2.5 m PointGoal，与原 ImageGoal 一起给 NavDP；
证据不足时按原有方法回到 ImageGoal 请求。2.5 m 是局部策略条件，不是一次必须走完的距离。

查询期间新 RGB 更新当前 LingBot 几何，不扩充封存 Survey 的检索候选范围，
不把中间帧塞进 NavDP FIFO。高度先验约 0.42 m；D435i 的 metric depth 不进入策略。
Go2 位姿只用于执行坐标和反馈，不作为 CEC 定位真值。

## 2. 默认行为改动

| 环节 | 合并前或现场问题 | 本次行为 |
|---|---|---|
| 反馈周期 | 旧代码往往先走完整条 2–3 m 路径再规划；006 出现约 50.64 s 回包间隔 | 普通运动允许后台滚动规划；不要求每 0.3 m 停一次 |
| 返回路径坐标 | 相机曝光到回包期间机器人会移动 | 新路径绑定 RGB 曝光时刻的位姿，再用当前位姿跟踪 |
| 转身观察 | 原地转身期间缺少中间几何帧 | 保留 IMU 连续转身，但新增 observation-only 更新；期间不反复抽样策略 |
| 状态写入 | 定时器/worker 超时和重复帧可能破坏采样语义 | 单 worker 串行写入；一次源 RGB 只消费一次 |
| 停止边界 | 旧回包可能晚于停止/换目标 | 运行边界变化后拒绝旧动作，速度发布前再次检查 enabled/estop |
| 规划寿命 | 活跃轨迹曾绕过 age 检查 | 活跃轨迹也检查过期；1.5 s 输入准入、5 s 规划寿命上限保留 |
| 低 critic | 搜索占位 `[0,±1]` 与横向位置目标混用 | 认证转向优先；不把占位符直接执行为 1 m 横移 |
| 尾段停滞 | 局部路径尾部低速指令可能长期无进展 | 保留现场 0.15 m 结束容差、4 s / 0.02 m 停滞重规划；换新路径不重置真实位移停滞判断 |
| 深度安全 | 旧提交只对前进硬停，现场已修正 | 保留现场对转向同样有效的近障碍/无效深度停止；前向 ROI 仍不证明侧方净空 |
| 到达新鲜度 | 重复/过旧 RGB 可能被用于判定 | 增加去重、源时间和匹配完成时间检查，不放宽原视觉阈值 |

保留最大速度 0.3 m/s、最大角速度 0.55 rad/s、8° 转向容差、
0.35 m 深度硬停以及启动 `enabled=false / estop=true`。
没有更新模型、ROS/RealSense 依赖、相机分辨率或 Jetson 功耗模式。

## 3. 可选实验，不是默认打开的优化

### 3.1 近目标局部尺度模式

默认仍为 `terminal_approach=bearing_only`。
新增入口：`deployment/config/experiments/fullmono_local_approach.json`，
对应 `height_scaled_local`：

1. 当前图与目标图直接定位通过，且匹配当前 frame 的 first40 高度尺度有效。
2. 估计距离在 (0, 2.5 m] 时缩短 PointGoal，不将远距离估计直接变成大跨度动作。
3. 进入约 0.60 m 邻域、有目标相机朝向时先对准；8° 内保持静止并等待视觉判定。
4. 预测距离、局部路径结束、零轨迹都没有成功/STOP 授权；视觉检测仍可拒绝。

这专门处理 006 的“估计只剩 0.318 m，控制仍请求 2.5 m”问题。
离线重判会将该条改为约 54.659° 的目标相机朝向对齐，但还未证明现场能到达。
0.60 m 是待验证的末端切换值，不是误差上界或新成功标签。

比较时两臂必须使用同一末端模式；启用后的 comparator 是
**Mono-native + shared terminal adapter**，不能冒充未改动的旧 Mono-native。
新收据使用 `cec_local_approach_handoff_v3_20260907`，客户端和 Hub 必须匹配版本。

### 3.2 有界低 critic 搜索

`low_critic_search_enabled=false` 为默认。
可选搜索使用固定左右意图、最长约 0.30 m / 半径 0.40 m 的短前进圆弧，
期间只更新几何，完成后重新规划。它不是完整 U-turn，也不是绕障规划器，
尚未做物理验证。未确认扫掠净空时不要启用。

## 4. 纯旋转风险与仿真差异

**当前大角度转身仍是原地纯 yaw，并未改成仿真的弧形 U-turn。**

仿真普通跟踪在可通行时同时前进与转向，受阻时也可能只转不走。
旧专门 U-turn 是有平移的 Dubins 圆弧，但论文主表默认关闭，且依赖仿真
NavMesh 检查整条路径。它不能不经适配直接搬到前向深度安全的 Go2 上。

新代码的中间 RGB 更新减轻跨过约 160° 的视图跳变，却不补充平移视差：

- IMU 只控制执行转角，没有作为 LingBot 位姿先验。
- 没有纯旋转平移约束、内部状态校正或转后重定位恢复机制。
- first40 高度估计只换算尺度；乘一个米制比例不能修正虚假位移或方向漂移。
- HTTP 成功、几何帧计数增加、封存历史不变，都不是定位准确性证据。

因此保留纯旋转是沿用已知执行方式的接口选择，不是已证明更适合单目几何的选择。
后续应先比较转身前后 LingBot 平移/bearing、实际机体位移和相机外参影响。
若移植弧形转弯，必须同时验证真实平移、连续观测与机身扫掠净空；
不能只给 `vx > 0` 就宣称有了有效视差。本 PR 没有宣称解决此风险。

## 5. 已有验证和剩余边界

### 已完成

- 004/006/009 共 **43 次真实规划收据**离线重判，默认模式命令字段 43/43 一致；
  不等于 43 条 episode，也不验证动态轨迹。
- 真实 Jetson 相机 + 隔离 RTX 服务，155 s 静止测量：
  137 次规划（稳态 123）、70 次几何观察，207/207 HTTP 200。
- 稳态 RGB → 规划返回：中位数 **0.765 s**，P95 **0.894 s**；
  实际约 **1 Hz**，而不是配置的 1.5 Hz。
- 几何专用 20 s 窗口：58 次，即 **2.9 Hz**；不是同时承诺 1 Hz 规划加 3 Hz 几何。
- 到达匹配 562 次：中位数 80.4 ms，P95 106.4 ms，实际约 **9.37 Hz**。
- 3,094 次控制回调均保持禁用、急停、零速度；warmup 一个 2.811 s 回包被过期检查丢弃。
- Python 语法、导入、配置和改动 shell 的语法检查；遵照 AGENTS，不运行单元测试。
- 临时 GPU 服务和 SSH 隧道已关闭，原生产服务状态未 reset，原始实验数据未删除。

### 未完成

- 运动中滚动换轨的平滑性、实际端到端执行误差。
- 原地转身期间 LingBot 平移漂移，或新弧形转弯的有效性。
- 近目标接近/对准/视觉确认/停车的完整物理闭环。
- 新配置的正式 SR：**本次没有新增导航成功率**。

静止测量全部使用同一 anchor 23，直接目标证据未通过，近目标动作未触发；
不能用它宣布 004/006 已被救回，也不能估计所有场景的最坏延迟。
原始计数、时序和启动错误详见 [延迟实测报告](REALWORLD_LOCKED_LATENCY_RESULT_20260907.md)。

## 6. 文件位置与接入顺序

| 文件/目录 | 职责 |
|---|---|
| `deployment/go2/navdp_ros_node.py` | 调度、几何观察、运行边界、转向优先级 |
| `deployment/go2/trajectory_execution.py` | 曝光坐标路径、滚动替换和实际位移反馈 |
| `deployment/go2/navigation_run_agent.py`、`latency_motion_guard.py` | 运行监督与规划时效 |
| `deployment/gpu/realworld_cec_hub.py`、`deployment/go2/navdp_client.py` | 双机几何端点和版本化返回接口 |
| `deployment/gpu/revisit_local_pose_adapter.py`、`deployment/go2/terminal_motion_override.py` | 显式近目标实验与机器人侧动作解释 |
| `deployment/go2/rgb_goal_arrival.py` | 独立视觉到达的新鲜度检查 |
| `deployment/go2/search_intent.py` | 默认关闭的有界搜索 |
| `deployment/runtime_config.py`、相关 launcher/health | 配置模式与两机版本校验 |
| `deployment/go2/audit_receding_capture.py` | 已保存真机收据的离线重判 |
| `deployment/go2/measure_locked_latency.py` | 独立端口/禁用运动的真实相机延迟测量 |

接入现有两台机器时：

1. 先保存并合并两机现有未提交工作，不要直接 reset 或覆盖现场 checkout。
2. 两机对齐同一合并后 revision，按实际目标/Survey 重新生成 resolved config；
   旧配置绑定旧 revision，不能跳过校验强行复用。
3. 先保持运动禁用，复核相机、反馈、模式、目标和服务健康。
4. 完成转向几何诊断，再在明确授权的受监督运行中验证滚动执行与末端动作。
5. 新实验使用新 run ID；不要修改 004/006/009 的历史结果或混入旧正式统计。

更多依据：[失败审计](REALWORLD_RECEDING_HORIZON_AUDIT_20260907.md)、
[收据复算](docs/evidence/realworld_receipt_reassessment_20260907.json)、
[当前状态](CURRENT_STATUS.md)。文档里的既有统计是证据，不是部署授权。
