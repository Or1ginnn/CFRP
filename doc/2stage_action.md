# EA-CFRP-VLN 中文实施方案 Action.md

## 1. 实施目标

本文件将 `Plan.md` 转化为可执行的工程与实验任务。

最终需要交付两个相对独立、可以分别验收的系统：

```text
Stage 1 Deliverable:
一个不干预动作的 Navigation Risk Monitor

Stage 2 Deliverable:
一个由 learned risk 触发、能够选择 continue/replan 并执行恢复的 VLN agent
```

实施原则：

1. 先证明模型能够发现问题，再训练 recovery；
2. 第一阶段使用 on-policy 监督学习，不从纯 RL 开始；
3. 第一阶段 plan 只读、无 tool、无 plan update；
4. 第二阶段才加入 tool 和 CFRP；
5. 每个模块都必须有独立 baseline 和验收标准；
6. 不让 oracle 信息泄漏进推理输入。

---

# Part I：项目目录与模块边界

## 2. 建议代码结构

```text
ea_cfrp_vln/
├── configs/
│   ├── base_nav.yaml
│   ├── stage1_collect.yaml
│   ├── stage1_train.yaml
│   ├── stage1_eval.yaml
│   ├── stage2_collect.yaml
│   ├── stage2_train.yaml
│   └── inference.yaml
├── env/
│   ├── habitat_wrapper.py
│   ├── checkpoint.py
│   └── metrics.py
├── planner/
│   ├── plan_state.py
│   ├── plan_parser.py
│   ├── prompt_builder.py
│   └── controller.py
├── model/
│   ├── vlm_backbone.py
│   ├── monitor_token.py
│   ├── risk_head.py
│   ├── token_head.py
│   └── ea_cfrp_model.py
├── data/
│   ├── rollout_collector.py
│   ├── oracle_labeler.py
│   ├── risk_dataset.py
│   ├── counterfactual_group_dataset.py
│   └── schemas.py
├── training/
│   ├── train_risk_probe.py
│   ├── train_risk_lora.py
│   ├── train_recovery_sft.py
│   ├── train_group_relative.py
│   └── losses.py
├── branching/
│   ├── critical_state_selector.py
│   ├── branch_runner.py
│   ├── branch_scorer.py
│   └── sample_filter.py
├── evaluation/
│   ├── eval_stage1.py
│   ├── eval_stage2.py
│   ├── calibration.py
│   └── plots.py
└── scripts/
    ├── collect_stage1.sh
    ├── train_stage1.sh
    ├── eval_stage1.sh
    ├── collect_stage2.sh
    ├── train_stage2.sh
    └── eval_end2end.sh
```

---

## 3. 核心接口

### 3.1 PlanState

```python
class PlanPoint:
    id: str
    status: Literal["done", "current", "todo", "abandoned"]
    text: str

class PlanState:
    global_route: str
    local_points: list[PlanPoint]
    current_index: int
```

必须实现：

```text
validate_exactly_one_current()
validate_done_immutable(old_plan, new_plan)
advance_current()
apply_compact_update(plan_update)
serialize_compact_xml()
```

`advance_current()` 是 Stage 1 允许的正常执行游标推进；
`apply_compact_update()` 只在 Stage 2 的合法 `replan` 后调用。

### 3.2 Model forward

```python
outputs = model(
    instruction=instruction,
    images=recent_images,
    actions=recent_actions,
    plan=current_plan,
    allowed_actions=allowed_actions,
)
```

第一阶段返回：

```python
{
    "token_logits": ...,
    "risk_logit": ...,
    "risk_score": ...,
}
```

第二阶段返回：

```python
{
    "generated_xml": ...,
    "risk_score": ...,
}
```

### 3.3 Risk 不控制动作的开关

```python
controller.passive_monitor = True   # Stage 1
controller.passive_monitor = False  # Stage 2
```

Stage 1 中禁止根据 risk 改动作或提前结束 episode。

---

# Part II：Phase 0——基础导航与 Plan 接口

## 4. 任务 P0.1：确定基础模型与 Benchmark

建议优先选择：

```text
Benchmark: R2R-CE / VLN-CE
Environment: Habitat
Base model: 当前可稳定复现的 VLM/VLA VLN baseline
```

验收：

- 能完整运行 train/val-unseen episode；
- 能取得可重复的 baseline SR/SPL；
- 能保存每一步 observation、action、pose 和 metrics；
- 能从任意 step checkpoint 恢复 environment。

---

## 5. 任务 P0.2：实现 Persistent Plan

先实现 read-only plan，不实现 recovery 更新。

初始 plan 来源可选：

1. 离线强模型生成并缓存；
2. 使用数据集 instruction 自动拆分；
3. 使用旧版 planner 生成。

第一版要求：

```text
每个 episode 初始化一次 plan
每一步将 compact plan 输入模型
Stage 1 中 controller 可根据 progress=advance 执行 current -> done、next todo -> current
模型不输出 plan_update
```

验收：

- plan XML 可稳定解析；
- 所有 episode 恰有一个 current point；
- 输入长度在固定预算内；
- plan 加入后 baseline 导航性能没有不可接受下降。

---

## 6. 任务 P0.3：定义 Stage 1 正常输出

第一阶段 assistant 输出格式：

```xml
<progress>hold</progress>
<subgoal>short local navigation instruction</subgoal>
<action>ONE_TO_THREE_COMMA_SEPARATED_ALLOWED_ACTIONS</action>
```

`progress` 只能为 `hold/advance`。它只推进正常 plan cursor，不允许重写、放弃或
插入计划点，因此不属于 recovery tool。

禁止：

```text
<tool>
<plan_update>
<risk>language token</risk>
free-form reasoning
more than three actions
```

`<action>` 始终只出现一次，其 payload 可以包含 1--3 个按执行顺序
排列的 primitive actions；`STOP` 必须单独出现。该序列是短时域动作预测，
不是不可打断的开环承诺。Controller 将未执行动作维护为 active queue：每执行
一个 primitive 就保存新观测和真实动作历史，并异步请求刷新；推理期间最多继续
执行一个旧队列动作，新响应到达后覆盖尚未执行的尾部。若推理期间执行的动作与
新预测前缀一致，先消去已经发生的前缀，禁止重复执行。

模型输入中的 action history 只能包含已经真实执行的动作。旧队列中尚未执行、
被覆盖或被丢弃的动作不得进入 prompt，也不得作为历史 assistant action 泄漏给
下一轮。完整 drain 一个 chunk 的行为只保留为评测消融。Risk 只能由 MLP head
返回。

验收：

- XML 有效率 ≥ 99%；
- action 合法率 ≥ 99.5%；
- progress 合法率 ≥ 99.5%；
- STOP 不会作为 tool；
- 输出不包含自由形式 CoT。

---

# Part III：Phase 1——On-policy Risk 数据

## 7. 任务 P1.1：On-policy Rollout Collector

使用当前策略自由 rollout，不进行 recovery 干预。

每个 instruction 建议先采样：

```text
2–4 stochastic rollouts
```

保存成功、失败和绕路成功 episode，不只保存失败。

每 step 数据结构：

```json
{
  "episode_id": "...",
  "step_id": 17,
  "instruction": "...",
  "plan": "<plan>...</plan>",
  "recent_frame_refs": ["..."],
  "recent_actions": ["MOVE_FORWARD", "TURN_RIGHT"],
  "model_output": "<subgoal>...</subgoal><action>...</action>",
  "agent_pose": [0.0, 0.0, 0.0],
  "goal_geodesic_distance": 5.7,
  "distance_to_expert_path": 1.4,
  "collision": false,
  "final_success": false
}
```

注意：pose、expert distance 等字段必须标记为：

```text
label_only = true
```

验收：

- 每个 step 可重建模型真实输入；
- 每个 episode 有完整结果；
- 成功/失败比例可统计；
- 能识别 loop、stuck、wrong stop；
- 收集过程不使用 risk 干预。

---

## 8. 任务 P1.2：Oracle Labeler v1

先实现便宜、可规模化的标签。

Risk 的最终语义统一为：

```text
在当前 instruction、plan 和历史条件下，不进行 recovery、继续当前策略执行时，
episode 最终失败的风险。
```

以下原始信号只作为训练期 oracle proxy，不等同于 risk 本身：

```text
d_route      = distance to expert route
d_goal_delta = recent goal-distance change
progress_gap = agent vs expert progress
loop_flag
stagnation_flag
wrong_stop_flag
plan_timeout = current plan point duration
```

统一标准化到 [0,1]，计算：

```text
risk_proxy = clip(
    w_route * d_route_norm
  + w_goal  * negative_goal_progress
  + w_gap   * progress_gap_norm
  + w_loop  * loop_flag
  + w_stuck * stagnation_flag
  + w_plan  * plan_timeout_norm,
  0, 1
)
```

生成：

```json
{
  "risk_proxy": 0.78,
  "state_label": "OFF_TRACK",
  "deviation_onset": 14,
  "label_components": {
    "route": 0.9,
    "goal": 0.5,
    "loop": 0.0
  }
}
```

初始状态阈值仅作起点：

```text
ON_TRACK  : risk < 0.30
AT_RISK   : 0.30 ≤ risk < 0.65
OFF_TRACK : risk ≥ 0.65 且持续 2–3 步
```

阈值必须在 validation set 上调，不写死为最终结论。

验收：

- 人工抽查至少 200 个 prefix；
- 标签与最终失败显著相关；
- 替代成功路线不会被全部标成 OFF_TRACK；
- 标签分布不过度集中于 ON_TRACK。

---

## 9. 任务 P1.3：困难样本与 Future-value Refinement

仅对以下样本做 suffix rollout：

```text
risk_proxy 接近阈值
几何偏离但最终成功
离 expert 很近但最终失败
人工抽查发现标签冲突
模型高置信度预测错误
```

从同一 simulator checkpoint 正常继续 \(M\) 次：

```text
M = 4（第一版）
```

得到：

```text
value_success = success_count / M
value_risk = 1 - value_success
```

对于这些困难样本，`value_risk` 是更高质量的 future-failure target。其余状态
使用便宜的 `risk_proxy`。若实验采用插值，只将结果称为 soft risk target：

```text
final_risk = α * risk_proxy + (1-α) * value_risk
```

验收：

- 边界样本标签一致性提升；
- 与 episode outcome 的 Brier/ECE 改善；
- 计算成本控制在总 step 的小比例上。

---

## 10. 任务 P1.4：数据切分与防泄漏

切分原则：

- train/val/test 按 scene 切分；
- 同一个 episode 的不同 prefix 不得跨 split；
- future suffix 结果不得进入模型输入；
- expert path/pose/goal distance 不得进入模型输入；
- 同一条 rollout 的近邻 prefix 在 batch 中避免过度重复。

建议训练采样：

```text
ON_TRACK  : 40%
AT_RISK   : 30%
OFF_TRACK : 30%
```

可使用 weighted sampler，不必破坏原始统计用于评估。

---

# Part IV：Phase 2——Risk Head

## 11. 任务 P2.1：加入 `<MONITOR>` Token

输入末尾加入：

```text
<MONITOR>
```

取其最后层 hidden state：

```python
monitor_hidden = hidden_states[:, monitor_index, :]
```

风险头：

```python
risk_logit = mlp(monitor_hidden)
risk_score = sigmoid(risk_logit)
```

建议 MLP：

```text
hidden_dim -> hidden_dim/2 -> 1
activation = GELU
dropout = 0.1
```

验收：

- 不影响原 tokenizer/embedding 保存加载；
- monitor token 能唯一定位；
- forward 同时返回 token logits 和 risk；
- risk head 可单独冻结/解冻。

---

## 12. 任务 P2.2：Risk Probe

训练设置：

```text
freeze all backbone parameters
train monitor embedding + risk MLP only
```

损失：

```text
L = BCE(risk_score, risk_target)
```

目的不是追求最终最好结果，而是回答：

> 基础模型 hidden representation 是否已经包含错误感知信息？

必须比较：

- last token hidden；
- mean pooling；
- `<MONITOR>` hidden；
- current frame only；
- full history。

验收：

- AUPRC 明显高于类别先验；
- history 版本优于 current-frame baseline；
- risk 与真实失败概率有单调关系。

---

## 13. 任务 P2.3：LoRA / Top-layer Adaptation

若 probe 不足：

```text
unfreeze top N transformer layers
or
apply LoRA to attention/MLP modules
```

联合损失初始配置：

```text
L_total = L_token
        + 1.0 * L_risk
        + 0.2 * L_rank
        + 0.05 * L_temporal
```

数值只是起始配置，需要验证集调参。

训练注意：

- token loss 防止导航能力退化；
- risk batch 需平衡状态类别；
- ranking pair 主要从同一 episode 内采样；
- temporal loss 不应抹平真实 onset。

验收：

- Stage 1 AUPRC/F1 提升；
- baseline SR/SPL 下降在可接受范围内；
- ECE/Brier 不恶化；
- risk trajectory 比 probe 更稳定。

---

## 14. 任务 P2.4：On-policy Iteration

流程：

```text
Round 0：base policy rollout
Round 1：train risk model
Round 2：updated model rollout
Round 3：merge/reweight datasets and retrain
```

第一版建议最多 2–3 轮。

每轮记录：

- 新增错误类型；
- risk false positive/negative；
- 数据分布漂移；
- 与旧轮次模型的性能差异。

---

# Part V：Phase 3——Stage 1 验收

## 15. 必做 Stage 1 实验

### E1：输入信息消融

```text
A. current frame
B. instruction + current frame
C. instruction + visual history
D. + action history
E. + read-only plan
```

### E2：输出结构消融

```text
A. external classifier
B. pooled hidden + MLP
C. <MONITOR> + MLP
D. generated state token
```

### E3：训练数据消融

```text
A. expert/success only
B. artificial perturbation
C. on-policy failures
D. on-policy + hard examples
```

### E4：标签消融

```text
A. binary distance threshold
B. combined deviation soft risk
C. + future-value refinement
```

### E5：训练方式消融

```text
A. frozen probe
B. LoRA
C. top-layer fine-tuning
```

---

## 16. Stage 1 验收门槛

进入 Stage 2 前至少满足：

```text
1. AUPRC 显著超过类别先验与单帧 baseline
2. OFF_TRACK recall 足以覆盖多数失败轨迹
3. 成功 episode 的 false alarm rate 可控
4. risk score 有合理校准
5. 平均检测延迟明显早于 episode failure
6. 加入 risk head 后导航性能没有严重下降
7. 在 val-unseen scene 上仍有效
```

建议不要仅用一个准确率门槛；必须综合 detection、false alarm、calibration 和 navigation preservation。

---

# Part VI：Phase 4——Recovery Warm-up

## 17. 任务 P4.1：启用 Tool 格式

第二阶段 assistant 输出：

正常：

```xml
<tool>continue</tool>
<subgoal>...</subgoal>
<action>...</action>
```

恢复：

```xml
<tool>replan</tool>
<plan_update>
  <abandon>...</abandon>
  <current>...</current>
  <future>...</future>
</plan_update>
<subgoal>...</subgoal>
<action>...</action>
```

验收：

- XML valid rate；
- allowed action rate；
- exactly one tool/action；
- done points immutable；
- exactly one current point；
- replan 后 future 与原 instruction 一致。

---

## 18. 任务 P4.2：Recovery SFT 数据

数据来源：

- expert-guided recovery examples；
- 人工构造典型 side-room/missed-turn/overshoot 场景；
- 较强 planner 生成的 plan update；
- 成功的历史 branch trajectories。

SFT 只训练：

- recovery 格式；
- recovery subgoal；
- plan update；
- first primitive action。

它不承担“什么时候触发”的主要学习任务。

---

# Part VII：Phase 5——Risk-guided CFRP

## 19. 任务 P5.1：Critical State Selector

输入：连续 risk 序列。

第一版规则：

```text
candidate if:
    risk_t >= τ_high
or
    mean(risk_{t-M+1:t}) >= τ_mid
```

加入：

- hysteresis；
- minimum step before trigger；
- per-rollout max trigger = 1；
- recovery cooldown；
- hard-negative logging。

训练采样中加入少量 oracle missed states，专门覆盖 risk false negative。

---

## 20. 任务 P5.2：Checkpoint / Restore

Checkpoint 分成环境状态与策略状态：

```python
class EnvironmentCheckpoint:
    simulator_state
    agent_state
    episode_id
    elapsed_steps
    metrics

class PolicyCheckpoint:
    plan_state
    recent_visual_buffer
    recent_action_buffer
    controller_state
    risk_history
    python_rng_state
    numpy_rng_state
    torch_rng_state
    cuda_rng_state
    decoder_rng_state

class CFRPCheckpoint:
    environment: EnvironmentCheckpoint
    policy: PolicyCheckpoint
```

instruction、episode metadata、expert/reference trajectory 和 checkpoint 前的完整
轻量轨迹作为 group-level shared context 保存一次。两个 branch 只保存各自 suffix，
不重复复制共享前缀。

验收：

- restore 后第一帧完全一致；
- continue/replan 两分支起点相同；
- plan/history 不串分支；
- branch 结束后能返回 normal rollout 或安全丢弃。

---

## 21. 任务 P5.3：Branch Runner

每个 critical state：

```text
Branch C: force first tool = continue
Branch R: force first tool = replan
```

第一步之后恢复 normal policy。

Branch mode：

```text
recursive branching = false
risk trigger = disabled
horizon = 30–50
```

推荐先使用 deterministic decoding 控制分支方差，再测试 stochastic sampling。

---

## 22. 任务 P5.4：Branch Scorer

第一版 score：

```text
score =
    w_success * success
  + w_goal    * goal_progress
  + w_route   * route_recovery
  + w_plan    * plan_consistency
  - w_loop    * loop
  - w_stuck   * stuck
  - w_coll    * collisions
  - w_len     * excess_path
  - w_invalid * invalid_output
```

必须记录各分量，不能只存总分。

Preference filtering：

```text
keep if:
    abs(score_replan - score_continue) > margin
    and winner output valid
    and branch has enough evaluable steps
```

丢弃：

- 两分支都好；
- 两分支都差；
- margin 太小；
- winner XML/action 无效；
- branch 因系统错误中断。

---

## 23. 任务 P5.5：Counterfactual Group Dataset

每个 group 保存：

```json
{
  "normal_prompt": "...",
  "risk_score": 0.81,
  "continue_output": "<tool>continue</tool>...",
  "replan_output": "<tool>replan</tool>...",
  "return_continue": -0.22,
  "return_replan": 0.64,
  "advantage_continue": -0.43,
  "advantage_replan": 0.43,
  "continue_score_components": {},
  "replan_score_components": {},
  "checkpoint_id": "..."
}
```

重要：

```text
continue/replan 两份输出的 log-prob 必须在 normal_prompt 下计算。
forced prompt 不进入训练条件。
```

---

# Part VIII：Phase 6——第二阶段训练

## 24. 任务 P6.1：Counterfactual Group-relative Advantage Optimization

第一版主方法：

```text
R_c = G(continue branch)
R_r = G(replan branch)
R_mean = (R_c + R_r) / 2
A_c = R_c - R_mean
A_r = R_r - R_mean

L_CFRP = -A_c * log π(continue_output | normal_prompt)
         -A_r * log π(replan_output | normal_prompt)
```

原则：

- 同组分支必须来自同一 checkpoint；
- advantage 只在组内计算；
- 两份输出都参与更新，不先压缩成 chosen/rejected；
- forced prompt 只生成候选，不用于计算最终 policy log-prob；
- 第一版每组两个分支即可，后续可在每个 tool 下增加多个 suffix sample。

DPO / pairwise preference training 仅作为实验 baseline。

方法命名边界：候选输出由 forced-tool prompt 生成，但 policy log-prob 在 normal
prompt 下回算，因此该 estimator 不属于严格的经典 on-policy GRPO。论文统一称为：

```text
Counterfactual Group-relative Advantage Optimization
```

训练 mask：

```text
System/input tokens = 0
Assistant XML output = 1
```

Risk head 第一版冻结。

---

## 25. 任务 P6.2：联合或交替训练

推荐顺序：

```text
1. freeze risk head, train recovery policy
2. evaluate risk distribution shift
3. optionally recollect on-policy risk data
4. small-LR recalibrate risk head
```

不要一开始同时大幅更新 risk 和 recovery policy，否则 critical-state 分布持续变化，难以定位问题。

---

# Part IX：Phase 7——端到端推理

## 26. Controller 状态机

建议状态：

```text
NORMAL
RISK_ACCUMULATING
RECOVERY_ENABLED
RECOVERY_COOLDOWN
TERMINATED
```

转移示例：

```text
NORMAL
  ├─ low risk → NORMAL
  └─ medium/high risk → RISK_ACCUMULATING

RISK_ACCUMULATING
  ├─ risk drops → NORMAL
  ├─ persistent high risk → RECOVERY_ENABLED
  └─ emergency risk → RECOVERY_ENABLED（跳过 persistence，但仍由模型选择 continue/replan）

RECOVERY_ENABLED
  ├─ tool=continue → NORMAL/RISK_ACCUMULATING
  └─ tool=replan → RECOVERY_COOLDOWN

RECOVERY_COOLDOWN
  ├─ cooldown active → continue execution
  └─ cooldown expires → NORMAL
```

---

## 27. 推理伪代码

```python
reset_env()
plan = initialize_plan(instruction)
history = HistoryBuffer()
mode = "NORMAL"

for t in range(max_steps):
    model_input = build_input(
        instruction=instruction,
        observation=env.observation,
        history=history,
        plan=plan,
        allowed_actions=env.allowed_actions,
        add_monitor_token=True,
    )

    output = model.generate_with_risk(model_input)
    risk = smooth_risk(output.risk_score)

    if stage1_only:
        xml = output.stage1_xml
    else:
        mode = update_risk_state_machine(mode, risk)
        xml = output.stage2_xml

        if mode not in {"RECOVERY_ENABLED"}:
            xml = force_or_validate_continue(xml)

        if xml.tool == "replan":
            plan = apply_and_validate_plan_update(plan, xml.plan_update)
            mode = "RECOVERY_COOLDOWN"

    action = validate_action(xml.action, env.allowed_actions)
    env.step(action)
    history.update(env.observation, action)

    if env.done:
        break
```

---

# Part X：实验矩阵

## 28. Stage 1 主表

建议列：

```text
Method
Input history
Read-only plan
On-policy data
Risk supervision
AUROC
AUPRC
OFF F1
Episode FPR
Detection delay
ECE
SR/SPL preservation
```

---

## 29. Stage 2 主表

建议列：

```text
Method
Trigger
Recovery training
SR
SPL
Recovery success
Unnecessary replan
Replan frequency
Recovery latency
Path efficiency
```

核心比较：

```text
No recovery
Oracle trigger + recovery
Prompt trigger + recovery
Risk trigger + heuristic recovery
Risk trigger + SFT recovery
Risk trigger + CFRP
Full EA-CFRP-VLN
```

---

## 30. 必做案例可视化

至少展示三类轨迹：

### Case A：成功提前报警

```text
risk 逐步上升
模型在最终失败前发现问题
replan 分支恢复成功
```

### Case B：困难负样本

```text
模型暂时绕路但仍可成功
risk 保持中低
没有不必要 replan
```

### Case C：失败案例

```text
risk 未检出
或 risk 检出但 recovery 失败
```

每个案例画：

- top-down trajectory；
- expert route；
- risk curve；
- plan state；
- tool decision；
- branch outcomes。

---

# Part XI：风险与应对

## 31. 风险一：标签等同于 expert distance

问题：替代成功路线被误标为错误。

应对：

- 使用组合信号；
- 加入成功绕路困难负样本；
- 对边界样本做 suffix rollout；
- 报告按 expert deviation 和 future success 两种定义的结果。

---

## 32. 风险二：Risk Head 只记住视觉场景

问题：看到某类房间就报错，没有理解指令和过程。

应对：

- input ablation；
- plan-shuffling / instruction-shuffling counterfactual test；
- 相同图像不同 instruction 的对比样本；
- 同一场景不同历史的对比样本。

---

## 33. 风险三：总是高风险

问题：类别不平衡或 false-positive penalty 不足。

应对：

- calibration；
- hard negatives；
- balanced sampler；
- focal/BCE 权重；
- episode false alarm 作为核心指标。

---

## 34. 风险四：Recovery 过度触发

应对：

- risk hysteresis；
- persistent threshold；
- cooldown；
- unnecessary replan penalty；
- `continue` winner 样本必须保留。

---

## 35. 风险五：Risk 与 Policy 联合训练不稳定

应对：

- 先冻结 probe；
- 再 LoRA；
- Stage 2 第一版冻结 risk；
- 交替收集与训练，不做完全同步在线更新；
- 保留每一轮模型和数据版本。

---

# Part XII：推荐里程碑

## 36. Milestone 1：可复现基础系统

交付：

- baseline VLN；
- checkpoint/restore；
- fixed-context prompt；
- read-only plan；
- rollout logger。

通过条件：baseline 指标可重复。

## 37. Milestone 2：Stage 1 数据与标签

交付：

- on-policy dataset；
- oracle labeler；
- 标签可视化；
- 人工抽查报告。

通过条件：标签能反映错误演化且无严重泄漏。

## 38. Milestone 3：Stage 1 Risk Monitor

交付：

- `<MONITOR>` + MLP；
- frozen probe；
- LoRA 版本；
- Stage 1 主实验和消融。

通过条件：达到 Stage 1 验收门槛。

## 39. Milestone 4：Recovery Warm-up

交付：

- tool/XML schema；
- plan update validator；
- recovery SFT model。

通过条件：格式、action 和 plan update 合法率达标。

## 40. Milestone 5：CFRP Branch Engine

交付：

- learned-risk critical state selector；
- branch checkpoint/restore；
- continue/replan runner；
- branch scorer；
- counterfactual group dataset。

通过条件：两分支严格同起点且评分可复现。

## 41. Milestone 6：End-to-end EA-CFRP-VLN

交付：

- counterfactual advantage-optimized recovery policy；
- controller state machine；
- end-to-end results；
- oracle-triggered 与 model-triggered 对比；
- 失败分析。

---

# Part XIII：当前立即执行清单

## 42. 本周优先级

- [ ] 选定并复现一个 continuous VLN baseline；
- [ ] 实现固定窗口 visual/action history；
- [ ] 将旧版 persistent plan 改为 Stage 1 read-only 输入；
- [ ] 定义 Stage 1 XML：只含 progress/subgoal/action；
- [ ] 实现 on-policy rollout logger；
- [ ] 实现 expert-route distance、goal progress、loop/stuck 统计；
- [ ] 生成第一版 soft risk 标签；
- [ ] 人工可视化 20 条成功和 20 条失败 risk curve；
- [ ] 加入 `<MONITOR>` token 和两层 MLP；
- [ ] 完成 frozen risk probe。

## 43. Stage 1 完成后再做

- [ ] Recovery SFT 格式数据；
- [ ] `continue/replan` tool；
- [ ] plan update validator；
- [ ] learned-risk critical state selector；
- [ ] counterfactual branch engine；
- [ ] counterfactual group dataset；
- [ ] counterfactual group-relative advantage optimization；
- [ ] DPO baseline；
- [ ] end-to-end inference state machine。

---

## 44. 最终执行原则

```text
先把“发现问题”做成一个独立、可监督、可校准的模型能力。
不要让 Stage 1 的 risk 改动作。
不要在 Stage 1 学 tool。
不要让 oracle 信息进入推理输入。
不要把偏离 expert path 直接等同于必然失败。
Stage 2 只在 Stage 1 通过验收后启动。
反事实 forced prompt 只生成候选，不进入最终 policy condition。
最终必须分别报告 detection、oracle-recovery 和 end-to-end 三组结果。
```
