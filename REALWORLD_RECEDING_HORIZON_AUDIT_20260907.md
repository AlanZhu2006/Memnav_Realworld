# 真机与仿真对齐：失败审计、执行适配与验证边界

日期：2026-09-07。分支：`feat/sim-aligned-receding-horizon-20260907`。

本轮完成代码适配、真实收据离线重判与配置检查，**没有部署到机器人，没有新增闭环 SR**。
本文不把“代码已经修改”写成“004/006 已被救回”。
后续同日完成隔离双机静止延迟测量，见
[实测报告](REALWORLD_LOCKED_LATENCY_RESULT_20260907.md)；它不补足运动几何或自动到达验证。
合并范围与剩余风险见 [本次变更说明](REALWORLD_EXECUTION_ADAPTATION_CHANGELOG_20260907.md)。

## 1. 核心判断

当前最明确的问题是**视觉规划、实际执行、近目标动作三个环节没有使用一致的时间和目标语义**，
不是已有证据能支持的“CEC 记不住位置”或“Go2 底层控制器没复现成功”。

- 004：低 critic 的 `[0, ±1]` 搜索占位轨迹被当作普通横向目标，与 CEC 的朝向纠正冲突。
- 006：有效的当前图—目标图定位曾估计仅剩 0.318 m，但控制目标仍为 2.5 m；目标相机朝向也未被用于末端对准。
- 两轮都存在整条局部轨迹执行完才重新规划的问题。006 相邻规划回包最长间隔约 50.64 s，远大于约 0.8 s 的推理时间。
- 原地转身期间不接收新的策略请求，也就没有中间图像进入 LingBot。下一次观察可能跨过一次约 160° 的转身。

这些是可从代码和记录定位的接口缺口。仍不能据此保证前方有可通行路线，或断言每次朝向反转都是经过了真实目标。

## 2. 本轮检查了什么

读取了 Jetson `unitree-dog:/home/unitree/MemNav-RealWorld` 的最新工作树和受保护实验注册表，
以及 GPU 对应工作树。两机原 HEAD 均为 `95afe64`，但有尚未提交的现场和 resident-policy 改动。

注册表当前指定有效完成项为 001、002、003、004、006、009；005、008、010 仍在进行中。
这只是保护和归档范围，不是本轮重新计算的正式 SR。010 的 GPU 内存阻塞属于另一条工作线，未重启其服务。

主要依据：

- `runtime/go2/experiment_pairs/index.json` 和各 pair 的 `pair.json`、manifest；
- 004、006、009 的 `cec_receipt.jsonl`、`status.jsonl`、`rgb_arrival_status.jsonl`；
- 006 当前重采集版本，而不是早先另一轮撞墙解释；
- `Nav-graph-blind/MemNavData/REALWORLD_SIM_GAP_AUDIT_20260907.md`；
- 仿真 `MemNavData/eval_2leg_habitat.py` 的 `pursuit_step`、8 帧重规划和非规划帧 memory 更新逻辑；
- 真机 executor、HeadingTurn、HTTP router、direct-local adapter、arrival detector 和运行监督器。

未上传现场图片、视频、完整原始收据、模型权重或数据集。

## 3. 失败证据与不能推出的结论

| 记录 | 可核对事实 | 解释边界 |
|---|---|---|
| 004 | 有效执行窗口约 95.03 s；19 次规划；7 次低 critic；anchor 始终为 15 | 不支持“频繁换 anchor 导致失控” |
| 004 | 低 critic 轨迹重复 `[0,-1]`；后续 CEC 再要求纠正朝向 | 这是搜索意图与位置目标混用，不代表右侧真实可通行 |
| 004 | 9 次 RGB-D 暂停，约 23.17 s；近障碍记录约 0.204 m | 不能为保持推进而取消碰撞或传感器新鲜度约束 |
| 006 | 14 次规划；无低 critic；anchor 始终为 23 | 不能套用 004 的低 critic 归因 |
| 006 | UTC 10:19:35.779876，第 37 次请求、当前帧 189，PnP 155 inliers，RMSE 0.987 px | 是直接定位证据，不是独立 GT |
| 006 | 估计距离 0.318305 m；目标朝向向右 54.659°；下发 point goal 长度仍为 2.5 m | 近目标信息没有进入动作，不应只怪底盘 |
| 006 | 后续相邻规划最长约 50.64 s；未触发视觉 arrival | 推理快不等于反馈闭环快 |
| 009 | UTC 14:23:09.631704 有 RGB arrival latch；55 matches、46 inliers | 证明既有 detector 曾工作，不证明通用自动停止可靠性 |

004 的命令时间估计采用互斥分类：约 34.74 s 仅转向、9.99 s 有前进指令、50.30 s 零指令。
这不是里程计实测运动时间，不能用指令积分替代真实位移。

## 4. 仿真与真机：对齐什么、不照搬什么

| 环节 | 主要仿真实验 | 本分支真机适配 |
|---|---|---|
| 策略输入 | RGB + LingBot 单目深度，冻结 NavDP | 保持；D435i 深度只用于本地安全 |
| 记忆 | 因果历史与几何认证的 bearing | 保持，不换模型、不改检索和证书阈值 |
| 重规划 | 每 8 个仿真控制帧更新；最多约 0.3008 m 指令前缀 | 异步滚动更新，20 Hz 执行，目标规划频率 1.5 Hz；不强制每 0.3 m 停车 |
| 中间观察 | 非规划帧仍写 memory | 查询中非规划 RGB 可独立更新几何，转身时继续观察 |
| 执行坐标 | 当前仿真状态执行局部路径 | 路径绑定 RGB 曝光时刻的 Go2 位姿，再用当前位姿跟踪 |
| 大转身 | 主要表格默认未启用专门的原子 rear-alignment；普通前进圆弧控制 | 保留已验证的 IMU 连续原地转身，禁止盲目照搬仿真小步转向 |
| 到达 | GT 距离 < 1 m 可结束；不是完整视觉自停验证 | 无 GT；相机朝向对准后仍由独立 RGB detector 判定 |
| 传感器/碰撞 | 仿真步进、理想状态接口 | 保留 RGB-D age、局部深度、IMU/位置反馈、人工 STOP |

因此这是**感知—规划—执行反馈结构对齐**，不是声称两套硬件动作逐帧相同。
Go2 里程计只做底层坐标补偿和执行反馈，不作为 CEC 的 GT 位姿或策略输入。

## 5. 新的运行结构

```text
Jetson：同步 RGB-D + 曝光时间 + Go2 位姿缓冲
  ├─ RGB → 单一 HTTP worker → GPU 因果 LingBot
  │                           ├─ 规划帧：depth + CEC → frozen NavDP
  │                           └─ 中间帧：仅几何更新，不抽样策略
  └─ depth → 本地安全（不进入策略）

新路径返回 → 检查其所属目标/运行边界、输入时间
           → 用曝光时刻位姿将局部路径放入执行坐标
           → 替换仍在执行的旧路径
           → 20 Hz 跟踪 + 当前深度安全 + 命令限速

需要大转身 → 锁定一次 IMU 朝向目标，连续执行
           → RGB 仍进入 LingBot，NavDP 不反复重采样转向
           → 完成后用停下后的新 RGB 重新规划
```

单 worker 和 Hub 互斥锁串行拥有状态写入。不会新增两个线程并发写 LingBot。
新 `/query_observation_step` 不追加 sealed Survey、不更改目标候选 ceiling，也不写 NavDP FIFO。
它只推进当前查询的因果几何状态；数据集仍是之前封存的历史。

规划频率是请求目标，不是已测得实时吞吐。若单次规划已占用约 1 s，不能同时承诺每秒 1.5 次规划和 3 次额外几何更新。
`geometry_rate_hz=3` 主要让原子转身/非规划窗口有中间观察；所有请求串行、只取新 RGB，不堆积待执行旧帧。

### 5.1 纯旋转与尺度：本次没有解决的部分

当前 `HeadingTurn` 是线速度为零的 IMU 原地转身，**不是**仿真中有前进位移的圆弧/水滴形 U-turn。
仿真普通 `pursuit_step` 在通路正常时同时前进和转向，受阻时也可能只转不走；论文主表的
专门 terminal U-turn 默认关闭，不能把旧 U-turn 机制结果归入主表验证。

连续送图只减少“转前与转后相差约 160°、缺少中间画面”的视角跳变，不补充平移视差。
`query_observation_step` 仍向普通 `/memory_step` 写入 RGB；没有向 LingBot 输入 IMU 姿态先验，
也没有旋转专用的平移约束或内部状态恢复。因此可能出现底盘转准而 LingBot 当前位姿漂移。
历史 JPEG 不变、FIFO 不变、请求成功，都不能证明当前几何正确。

first40 相机高度先验只把模型尺度换算到米，不校正错误平移或旋转；原地旋转退化仍需量化。
新弧形转向尚未实现。本次保留已知 IMU 执行是控制接口选择，不是已证明几何更优的选择。
部署前应重放或受控采集转身 RGB/IMU/里程计，比较位移、bearing 和转后重定位；若移植弧形转向，
还须验证实际平移和机身扫掠净空，不能用前向深度 ROI 代替完整转弯空间检查。

## 6. 默认执行修复

1. 去掉“active trajectory / heading 一律禁止推理”的规则。普通运动可重规划；原子转身只做几何更新。
2. 修复 worker 的 `Event.wait()` 超时后仍继续执行的问题，避免计时器频率被空轮询绕开。
3. 每个 RGB 源时间戳最多消费一次，不把同一张图当作多个新状态。
4. 新轨迹按曝光时间绑定位置；不把服务器回包时的机体坐标错当成采图坐标。
5. 增加 motion epoch，停止/重新授权/目标切换边界前发出的旧动作不能覆盖新状态。
6. 发布速度的最后一刻再次检查 enabled/estop，人工停止优先。
7. 正在执行不再豁免 plan age。沿用 1.5 s 输入准入、5 s 已接受规划寿命；后者是故障上限，不是目标重规划周期。
8. 原子转身期间几何更新也受同一 5 s 新鲜度上限约束；几何失败不能被当成普通 CEC reject。
9. 新路径替换不重置“持续要求前进但实际没移动”的 watchdog；使用当前世界坐标位移而不是重置后的新路径进度。
10. 保留现场已有 0.15 m 局部路径结束容差、4 s / 0.02 m 停滞重规划；局部结束不等于 ImageGoal 到达。

保留：最大速度 0.3 m/s、最大角速度 0.55 rad/s、8° 朝向容差、原有底盘门槛、0.35 m 深度硬停、
启动 `enabled=false / estop=true`。没有提高底盘速度去掩盖末端停滞。

## 7. 004：搜索意图与认证转向分离

优先顺序为：安全停止 → 合格的终端/认证转向 → 正常 NavDP 路径或显式低 critic 搜索。
认证转向不再先被低 critic 分支屏蔽。

低 critic 的 `[0,±1]` 不再允许直接作为 1 m 横移目标执行。默认仍停止该动作并等待新观察。
另提供 `low_critic_search_enabled=true` 的**受监督实验选项**：

- 沿用低 critic 搜索的初始左右符号，不随每次抽样随机反向；
- 前进圆弧最长 0.30 m、半径 0.40 m，最多执行 3 s；
- 一次短搜索期间只更新几何，不反复抽样；结束后重新规划；
- 仍经过深度安全、限速、反馈和人工停止。

这个小动作不是全局路线规划器。正前方深度 ROI 不能证明转向扫掠侧方净空，所以默认关闭。
004 前方实际狭窄；本轮不能证明打开开关就能绕过去。连续低 critic 也可能仍需人为选择更合适的场景或另做路线实验。

## 8. 006：近目标适配是新的显式实验，不偷偷改旧基线

默认 `terminal_approach=bearing_only` 保持原来的 scale-free 2.5 m 请求。
新增 `height_scaled_local`：

1. 必须是当前帧的 direct-local certificate；历史 anchor 的远距离估计不能触发近目标操作。
2. 仅接受 `mdtec_first40`、40 帧、未 clamped 的相机高度尺度 receipt；raw 向量乘尺度必须与距离一致。
3. 只有估计距离在 (0, 2.5 m] 内才授予局部尺度控制权限；更远仍是原来的 2.5 m bearing。
4. 在此范围内将 PointGoal 长度缩至估计距离，不额外延长行进请求。
5. 进入 0.60 m 邻域且有目标相机朝向时，优先对齐朝向；误差 ≤ 8° 后保持静止，交给独立 RGB detector。
6. 距离估计、零轨迹、局部路径结束都不授权 STOP。视觉未通过时保持未到达状态，不捏造成功。

0.60 m 是待现场验证的末端模式切换半径，不是从仿真正式主表验证出的到达阈值，也不是 metric 误差上界。
它可能因尺度/定位误差过早对准或等待；本轮未证明朝向修正后 detector 一定通过。
尺度只用 0.42 m 相机高度先验，不读 D435i metric depth 来估计导航目标。

两臂必须使用相同 `terminal_approach`。新配置的 comparator 应写为
**Mono-native + shared terminal adapter**，不能叫原封不动的 Mono-native。
默认旧配置仍严格禁用 native 的 CEC/direct-pose 控制权限。新条件不得混入旧 pair 的统计。

配置入口：`deployment/config/experiments/fullmono_local_approach.json`。
它沿用项目私有目标图路径占位，使用前必须按当前 Episode 冻结目标解析；不能把示例路径当成真实新实验目标。
resolved config、formal receipt、Hub health 和计划收据都携带模式，Jetson health preflight 核对一致性。

## 9. 到达检测与摄像头缺口

原来的 SIFT/homography arrival detector 保留，未降低 matches/inliers/coverage 阈值去“救”006。
新增源 RGB 去重、过旧图像拒绝、控制状态新鲜度检查，并把匹配耗时计入到达证据年龄（默认最大 0.60 s）。

006 的 SuperPoint/LightGlue 155 PnP inliers 不等于 SIFT arrival 的 155 inliers；两者不能直接替代。
现有 detector 在 009 有工程正例，但非平面/重复纹理/目标图视角偏差仍可能漏检或误检。
near-goal hold 如果始终不能取得视觉支持，需要以“未到达”结束，不能用一次预测距离补签成功。

RGB-D 中断不等于重规划算法缺失。保留已经落地的并行 callback / 小同步队列等现场修正；
本轮未调整 Jetson 功耗模式、USB、RealSense 驱动、分辨率或模型精度。后续运动锁定实测已完成，
但相机在真实步行/转身中的稳定性与运动控制耗时仍未测得。

## 10. 已完成的验证

`deployment/go2/audit_receding_capture.py` 只读保存的实际证据，不连接服务器、不导入 ROS、不发送运动。
机器可读复算结果和源文件哈希见
[realworld_receipt_reassessment_20260907.json](docs/evidence/realworld_receipt_reassessment_20260907.json)。

| 验证项 | 本轮结果 |
|---|---|
| 有效窗口内规划记录 | 004: 19，006: 14，009: 10；合计 43 次，**不是 43 条 episode** |
| 默认 bearing 模式 | 43/43 与已存 disposition、PointGoal、转角、STOP 字段一致 |
| 新尺度模式 | 006 有 1 次改变为近目标相机朝向；009 有 2 次获得局部尺度资格，但仍执行原来的 rear turn |
| 006 关键记录 | 0.318305 m 不再请求 2.5 m 前进；改为向右 54.659°，仍不授权 STOP |
| 004 优先级 | 7 次低 critic 中 1 次有合格认证转身，不再被低 critic 覆盖 |
| 距离直接授权 STOP | 0 次 |
| 源文件 | 9 个 JSONL SHA-256 前后不变 |
| Python | `compileall` 语法通过；Hub CLI 和纯 Python adapter 导入通过 |
| shell | 改动的 GPU launcher / health helper `bash -n` 通过 |
| 配置 | 3 个 tracked experiment JSON 结构通过；使用实际 006 目标的独立审计配置 resolve / integrity / shell export 通过 |

离线配置没有拿来启动：部署 verifier 正确拒绝了它与原 GPU checkout 不同的 Git revision。
这不是需要删除的检查；部署前应双机切换同一版本，再解析新配置。

遵守仓库 AGENTS：**未新增、恢复或运行 unit tests**。本节离线审计没有 mock ROS、虚构机器人轨迹、在线 HTTP state reset 或实机运动验证。
后续延迟测量仅 reset 本次新建的隔离服务，未 reset 原生产服务，且全程没有实机运动。
43 次离线重判不验证异步调度的真实时延、实际位移、绕障能力或新的 SR。

复算方式（`OUTPUT` 需为不存在的新文件）：

```bash
python deployment/go2/audit_receding_capture.py \
  --capture-root /path/to/copied/realworld_sim_audit \
  --output runtime/audit_receding_new/receipt_reassessment.json
```

## 11. 修改文件与隔离方式

- `trajectory_execution.py`：支持曝光坐标、滚动替换、跨计划实际位移 watchdog。
- `navdp_ros_node.py`：异步调度、单独几何请求、停止边界、转向优先级、状态披露。
- `navigation_run_agent.py`：活动轨迹也检查规划年龄。
- `navdp_client.py` / `realworld_cec_hub.py`：几何观察端点与双机 handoff v3。
- `revisit_local_pose_adapter.py` / `terminal_motion_override.py`：默认原规则、显式近目标模式。
- `search_intent.py`：默认不启用的有界搜索动作。
- `rgb_goal_arrival.py`：新鲜度和重复帧处理，原有视觉判据不变。
- `runtime_config.py` / launcher / health helper：显式配置、收据与版本一致性。

分支从 origin/main `95afe64` 建立。先用独立提交 `54da70e` 保留 Jetson 上已审计的六个文件修正，
再追加本轮适配。没有把其他人的 resident-policy、GPU 生命周期、UI 修改冒充为本轮成果。
原 GPU 工作区和 Jetson 的未提交修改均保留；没有 reset、删除运行数据、重启或切换在线服务。

## 12. 现场接入顺序与验收

1. 先保存并合并当前 Jetson/GPU 未提交工作，再选择 cherry-pick 本分支或合并。**不要在脏工作树直接 reset/switch 强制覆盖。**
2. 双机固定同一新 revision；新的目标、sealed Survey、配置、模式另建 run ID。不要复用旧 pair 路径。
3. motion-locked prepare：确认 health handoff v3、query observation capability、`terminal_approach`、0.42 m 相机高度一致。
4. 静止传感器预检：RGB-D age、IMU/位置反馈、GPU 显存、各阶段时延；先看到可用的新计划再授权运动。
5. 先在宽敞区域、同一固定目标验证滚动重规划：边走边产生不同 RGB 规划，旧轨迹不再执行十几秒；观察实际 plan/source age，不只看 HTTP 用时。
6. 验证约 160° 转身：持续 IMU 转向且中间 `query_geometry_observations` 增加；没有新 query ceiling 或重复 FIFO 写入。
7. 再开共享近目标配置，验证朝向对准、视觉 latch、速度归零与保持。先有人工距离/图像核验，不能让预测距离自证。
8. 有侧方净空保障后才单独验证低 critic 短搜索。未通过前维持关闭，不以静态圆弧长度代替扫掠安全。
9. 以上是现场 commissioning。通过后再冻结新两臂协议重采集；独立保存成功/失败与人工停止原因。

原先的 004/006/009 都是设计来源，只能用于回归诊断。最终成效仍需新运行，尤其不能用 006 的同一条 0.318 m 证据调好后直接报告新的 SR。
