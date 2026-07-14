# EA-CFRP-VLN 执行路线图

> 本文是工程执行的单一索引。方法定义以 `2stage_plan.md` 和
> `2stage_action.md` 为准；本文不改变方法 claim，只把它们转成可执行、可验收的
> Phase 0--7。

## 1. 方法对齐

EA-CFRP-VLN 不是“用 plan 做 recovery”，也不是“让语言模型输出 risk token”。
它将连续 VLN 的恢复拆成两个顺序明确的阶段：

1. **Stage 1：自主错误感知。** 正常导航策略在没有恢复干预的情况下自由
   rollout。共享 backbone 的 normal token/action head 负责
   `progress/subgoal/action`，独立 risk MLP 从 `<MONITOR>` hidden state 预测
   “若继续当前执行，未来失败的风险”。训练时可访问 oracle；推理时不可访问。
2. **Stage 2：反事实恢复规划。** 只有 learned risk 选出的高风险状态才触发
   checkpoint。相同物理和控制状态上分别展开 `continue` 与 `replan` 短程分支，
   用组内环境相对优势学习何时、如何恢复。

Persistent plan 是可执行状态，而不是 CoT。Stage 1 首轮由模型根据 instruction
初始化一次，随后由 controller 持久化并作为只读上下文返回模型；
`progress=advance` 仅允许推进 `current -> done`、`next todo -> current`。正常执行
不允许重复 plan，也不允许 `plan_update`、`tool`、`replan` 或 `risk` language token。

## 2. 不可违反的边界

| 边界 | 规定 |
| --- | --- |
| 推理可见信息 | instruction、read-only plan、固定预算 RGB history、action history、允许动作集合。 |
| 推理不可见信息 | expert/reference path、pose、goal position/distance、oracle action/label、counterfactual simulator。 |
| Stage 1 action | risk 只被记录和评估，绝不能修改 action、提前 STOP 或触发 recovery。 |
| Stage 2 trigger | 最终推理只由 learned risk 触发；训练可补少量 oracle false-negative hard state。 |
| 分支状态 | 共享 immutable trajectory prefix；分支只保存自己的 suffix。checkpoint 必须恢复 pose 和 controller memory。 |
| 优化表述 | Phase 6 是 counterfactual group-relative advantage optimization，不宣称为严格经典 on-policy GRPO；DPO 仅为 baseline。 |
| emergency risk | 只绕过 persistence、立即**启用** continue/replan 决策，不能硬编码“直接 replan”。 |

## 3. 当前状态

当前工作位于 **Phase 0**。下面是已经完成的基础能力，而不是额外的方法 phase：

- Habitat 0.3 + R2R-CE + MP3D 真实 episode、primitive action 和 task-level STOP；
- R2R 的 3m success threshold、真实 metrics、固定历史窗口；
- Qwen3-VL-4B split runtime 与真实 RGB inference smoke；
- Stage 1 XML parser、read-only persistent plan、checkpoint/restore、shared-prefix
  counterfactual state contract；
- shortest-path oracle 轨迹采集，以及 episode 级帧存储、短动作 chunk、慢快视觉历史；
- 可分片 warm-up collector、SFT manifest validator、Qwen3-VL LoRA SFT training
  preflight。

此前 200 条 episode 只是端到端链路 smoke，不是正式训练集，也不是 Phase 1 的
on-policy risk 数据。正式 Phase 0 使用 R2R-CE `train` 的全部 10,819 个 episodes；
action 来自 Habitat shortest-path oracle，plan/progress 是 instruction 拆分和确定性
标注，仅用于 normal-policy cold start。

## 4. 总览

| Phase | 名称 | 主要产物 | 进入下一阶段的门槛 |
| --- | --- | --- | --- |
| 0 | Base Navigation / Format Warm-up | 可用的正常 VLN policy | 固定验证子集指标可重复 |
| 1 | On-policy Risk Data | 当前 policy 的完整轨迹和 oracle-only risk labels | 标签、切分、泄漏审计通过 |
| 2 | Stage 1 Risk Training | `<MONITOR>` risk head | probe/LoRA/on-policy aggregation 完成 |
| 3 | Stage 1 Evaluation Gate | error-awareness 实验与阈值 | 检测达标且不伤害导航 |
| 4 | Recovery Format Warm-up | 合法 recovery XML / plan update policy | 格式和执行合法率达标 |
| 5 | Risk-guided Counterfactual Collection | same-state continue/replan groups | 分支同起点、评分可复现 |
| 6 | Group-relative Recovery Optimization | 反事实优势优化的 recovery policy | 与 DPO 等 baseline 比较完成 |
| 7 | End-to-end Iteration | learned-risk recovery agent | 全链路和消融完成 |

## 5. Phase 0：Base Navigation / Format Warm-up

### 目标

训练或加载一个可复现的 Qwen3-VL-4B 正常导航 policy。它只输出：

```xml
<!-- 仅 episode 首轮包含一次 <plan>...</plan> -->
<progress>hold|advance</progress>
<subgoal>short local navigation instruction</subgoal>
<action>one to three comma-separated primitive actions, or STOP alone</action>
```

它必须在首轮初始化 compact plan，随后读取 controller 返回的 plan、固定视觉/动作
历史，并在统一多轮 conversation 中持续导航；没有 risk head、tool、recovery 或
structural plan update。

### 剩余执行项

1. 以完整 `train` split 的确定性 shards 采集 normal-policy warm-up 数据；保留数据
   manifest、collector commit、split、seed、episode range 和配置。
2. 将完整 episode 转为统一的 bounded multi-turn conversation windows：窗口首轮
   输入完整慢快视觉上下文，后续轮只追加新连续帧；训练前检查 XML、图像 URI、
   action 分布、窗口连续性和缺失文件。
3. 在专用 Qwen 环境中安装最小 `peft` 依赖，先在小 shard 做 LoRA 训练 smoke；
   随后扩大到正式训练规模。Habitat 环境绝不安装 torch/transformers/peft。
4. 固定 `val_seen` 和 `val_unseen` episode IDs、seed、动作设置、history budget，
   重复两次 Stage 1 rollout，保存逐步记录和汇总 metrics。

### Phase 0 验收

- Stage 1 XML valid rate >= 99%；action 与 progress valid rate >= 99.5%；
- 没有 model prompt 含 pose、goal distance 或 expert path；
- complete episode 输出 SR、SPL、navigation error、oracle success、average steps、
  invalid output rate 和 STOP correctness；
- 固定验证子集两次结果可重复；
- plan 输入不造成不可接受的 baseline 性能下降；
- Qwen runtime 和 Habitat runtime 始终分离。

**禁止提前做：** risk MLP、risk label、recovery prompt、`continue/replan` tool、
counterfactual collection。

## 6. Phase 1：On-policy Risk Data

### 目标

用 Phase 0 的当前 policy 做自由 rollout，采集失败过程而不是只收 expert 成功轨迹。
每个 step 保存 model-visible input、normal output、metrics 和 training-only oracle
metadata；collector 不因 risk 改动作。

### 工作项

1. 实现 on-policy rollout logger，并区分 model-visible 与 `oracle_only` 字段；
2. 计算 v1 oracle proxy：进度、偏离、停滞、STOP/终局等信号组成 soft risk target；
3. 对阈值附近或困难状态做 suffix-rollout future-value refinement；
4. 以 scene/episode 为单位切分 train/val/test，审计不泄漏和类别比例；
5. 采样检查成功、失败、早期偏航和恢复前的风险曲线。

### 验收与禁止

- 标签可解释为 training proxy 或 soft risk target，不能偷换成“真实校准失败概率”；
- 输入中没有 oracle 字段，risk 不控制环境 action；
- 训练/验证按 scene 或 episode 的切分无泄漏。

## 7. Phase 2：Stage 1 Supervised Risk Training

### 目标

在共享 VLN backbone 上加入 `<MONITOR>` token 和 risk MLP：

```text
h_risk = hidden_state(<MONITOR>)
q_t = sigmoid(MLP(h_risk))
```

normal token/action head 保持 Phase 0 contract；risk 不是语言输出。

### 工作项

1. 先冻结 backbone，仅训练 monitor embedding、risk projection 和 MLP probe；
2. 若不足，再做 LoRA 或 top-layer adaptation，联合 token loss、risk loss、
   trajectory ranking 和 temporal regularization；
3. 进行 2--3 轮“更新 policy -> 新 on-policy rollout -> 再训练”的 aggregation；
4. 记录 calibration、false positive/negative、temporal stability 和导航性能变化。

### 验收与禁止

- forward 同时返回 normal token logits 与 risk score；risk head 可独立冻结/解冻；
- risk 与失败程度有单调关系，history 版本优于 current-frame baseline；
- Stage 1 仍不得用 risk 干预 action 或 episode。

## 8. Phase 3：Stage 1 Evaluation Gate

### 目标

证明模型在没有 external trigger 的情况下，确实能发现执行过程中的失效风险，且
不是记住视觉场景或 expert distance。

### 必做评测

- step-level AUROC、AUPRC、F1、Brier/ECE；
- episode-level failure prediction 与提前报警步数；
- 正常 VLN SR、SPL、NE 不发生严重下降；
- current frame vs history、无 plan vs plan、binary vs soft label、expert-only vs
  on-policy failure、probe vs LoRA 等消融；
- instruction/plan shuffle counterfactual test，排查视觉场景记忆。

### 进入条件

预先冻结 threshold 和通过标准。只有检测性能超过类别先验、单帧 baseline，且导航
性能可接受时，才可以进入 Phase 4。若未通过，回到 Phase 1/2 迭代，不能靠 oracle
trigger 跳过此门槛。

## 9. Phase 4：Recovery Format Warm-up

### 目标

只教会模型如何表达和执行 recovery，不教模型何时触发 recovery。

### 工作项

1. 启用 Stage 2 XML：`tool=continue|replan`；replan 时允许受约束的 plan update、
   recovery subgoal 和一个 primitive action；
2. 用少量 supervised recovery examples 训练格式、action 和 plan update 合法性；
3. 用 parser、controller 和 Habitat 执行检查所有字段。

### 禁止与验收

- SFT recovery 不是 decision policy，不把 expert deviation 作为推理 trigger；
- 无效 XML/action/plan update 必须可拒绝；
- 目标是恢复表达和执行的合法率，不宣称已学会“何时 recovery”。

## 10. Phase 5：Risk-guided Counterfactual Collection

### 目标

从 Phase 3 的 learned risk 选择 critical state，在同一 checkpoint 上生成短程
`continue` 和 `replan` 分支及其环境评分。

### 工作项

1. selector 使用 risk persistence/hysteresis；emergency 仅跳过 persistence；
2. 训练期允许少量 oracle missed states 补充 learned-risk false negative；
3. 保存 simulator pose、plan、history、turn、cooldown、risk history 等 checkpoint；
4. 每个 normal rollout 最多一组分支，控制 branch horizon 与总 action budget；
5. forced-tool prompt 仅用于覆盖候选，不进入最终训练条件；
6. 用 progress、success、SPL、path cost、invalidity、budget 等环境结果评分；
7. 保存 shared context、两条 suffix trace、forced output、normal-prompt log-prob
   所需信息和 filter reason。

### 验收

- 两分支严格相同物理和控制起点；restore/replay 可复现；
- group 不复制整条 shared prefix；
- 评分和过滤结果可复现；推理期无 simulator counterfactual branch。

## 11. Phase 6：Counterfactual Group-relative Advantage Optimization

### 目标

利用同一 checkpoint 的 `continue` / `replan` 环境差异，训练 recovery decision、
recovery subgoal、plan update 和 primitive action。

### 工作项

1. 在 **normal prompt** 下回算两份 first-step output 的 log-prob；
2. 在 group 内归一化环境 rewards，构造 relative advantage；
3. 初版冻结 risk head，只更新 recovery policy；必要时之后小学习率校准 risk；
4. 以 DPO/pairwise preference 作为明确的比较 baseline，而非主方法替代品；
5. 评估 advantage weighting、forced generation、branch horizon、risk trigger 的消融。

### 表述边界

forced-tool prompt 产生候选，而 normal prompt 回算 log-prob，因此这不是严格的经典
on-policy GRPO estimator。统一称为 **counterfactual group-relative advantage
optimization**。

## 12. Phase 7：End-to-end Iteration

### 目标

将 Stage 1 risk 与 Stage 2 recovery 放进同一推理 controller，但保持各自语义清晰。

### 工作项

1. normal action 后读取/平滑 risk；
2. 低风险维持 NORMAL；持续中高风险进入 RECOVERY_ENABLED；
3. 在 RECOVERY_ENABLED 下由模型选择 `continue/replan`，执行 recovery 后回到
   normal loop；
4. updated policy 重新生成 on-policy risk distribution，必要时回收 Phase 1 数据；
5. 完成主表、消融、失败案例与 risk curve / branch 可视化。

### 最终验收

- 推理只使用 instruction、plan、视觉历史、动作历史和内部 risk；
- learned-risk trigger 优于 oracle trigger 仅作上限的设定，且不依赖 external hint；
- 报告 detection、navigation、recovery 三类指标，并给出失败模式。

## 13. 近期顺序

当前只执行下面三项，即完成 Phase 0：

1. 扩大 normal-policy warm-up 数据并通过 manifest preflight；
2. 训练 Qwen3-VL normal `progress/subgoal/action` LoRA policy；
3. 冻结验证子集，重复评测 normal navigation。

完成后才启动 Phase 1。任何 risk、recovery、counterfactual 的代码扩展都必须等到
对应 gate 已通过。
