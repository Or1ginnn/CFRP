# EA-CFRP-VLN 中文研究计划

## 1. 方法名称

**EA-CFRP-VLN**
**Error-Aware Counterfactual Recovery Planning for Continuous Vision-and-Language Navigation**

中文名称：

> 面向连续视觉语言导航的自主错误感知与反事实恢复规划

一句话概括：

> EA-CFRP-VLN 将连续 VLN 中的恢复问题拆分为两个阶段：第一阶段使用当前导航策略产生的 on-policy 轨迹和训练环境中的 oracle 标签，以监督学习训练一个共享模型内部表征的风险预测头，使 agent 自主发现当前执行过程是否正在失效；第二阶段在模型自己预测的高风险状态上，对 `continue` 与 `replan` 展开短程反事实分支，并依据未来环境结果训练恢复决策与恢复计划。

整体框架：

```text
Instruction + Read-only Plan + Visual/Action History
                         │
                         ▼
                 Shared VLN Backbone
                  /               \
       Normal Token Head          Risk MLP Head
          action/subgoal              q_t∈[0,1]
                  \               /
                   └──── Stage 1 ─┘
                         │
                learned high-risk state
                         │
                         ▼
          continue branch vs replan branch
                         │
                 environment comparison
                         │
                         ▼
          tool / plan-update / recovery learning
                       Stage 2
```

---

## 2. 研究问题

连续 VLN 中，模型常见的失败不是完全不理解指令，而是：

```text
早期动作或定位发生偏差
        ↓
当前观测逐渐不再支持原计划
        ↓
模型没有意识到计划已经失效
        ↓
继续执行旧 subgoal
        ↓
错误不断累积
        ↓
loop / stuck / wrong stop / navigation failure
```

现有 recovery 方案往往依赖外部模块或 prompt 明确告知：

```text
“You are off route. Please recover.”
```

这主要训练了“被告知错误后如何恢复”，却没有解决：

> 在没有外部提示的情况下，模型如何仅根据指令、计划、视觉历史和动作历史，自主意识到当前执行过程已经出现问题？

因此，本项目将问题拆分为两个相互衔接但可以独立评估的研究问题：

1. **错误感知（Error Awareness）**：模型能否预测当前执行过程的失效风险？
2. **恢复决策（Recovery Planning）**：模型发现风险后，能否判断应继续还是重规划，并执行有效恢复？

---

## 3. 相比旧版 CFRP-VLN 的核心修改

旧版方案的训练链条是：

```text
expert trajectory 在线检测偏航
        ↓
直接触发 critical state
        ↓
continue / replan 反事实分支
        ↓
训练 recovery preference
```

新版方案改为：

```text
Stage 1：expert/oracle 只用于离线或在线生成监督标签
        ↓
模型学习内部 risk score
        ↓
推理时由模型自己的 risk 发现问题
        ↓
Stage 2：risk-guided critical state
        ↓
continue / replan 反事实分支
        ↓
使用组内相对优势训练 recovery decision 与 plan update
```

关键变化：

- expert trajectory 不再是测试时或最终训练时的外部触发器；
- 增加独立的风险预测头，显式训练“发现问题”的能力；
- 第一阶段不学习 tool，不允许更新 plan；
- 第二阶段才引入 `continue/replan`、recovery subgoal 和 plan update；
- 第一阶段和第二阶段分别评估，避免将“没发现错误”和“不会恢复”混为一谈。

---

## 4. 方法定位

EA-CFRP-VLN 面向连续 VLN，例如：

- R2R-CE；
- RxR-CE；
- VLN-CE；
- Habitat-based continuous navigation。

该方法关注：

> unseen continuous environments 中 agent 的执行风险感知、计划可靠性判断和恢复行为。

EA-CFRP-VLN 不是：

- 单纯 prompt engineering；
- 长篇 Chain-of-Thought 生成；
- 静态单帧 off-route 分类；
- 每一步重新生成完整全局路线；
- 仅依赖 expert path 的测试时检测器；
- 普通的同 prompt 多答案采样。

EA-CFRP-VLN 是：

> 一个具有共享内部表征的双头 VLN agent：正常 token head 负责导航输出，risk head 负责持续评估执行过程；当内部风险升高时，再通过 embodied counterfactual rollouts 学习恢复决策。

---

# Part I：共享基础结构

## 5. Persistent Executable Plan

模型维护结构化的持久化计划：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="current">follow the hallway</p>
    <p id="p3" status="todo">reach the stairs</p>
  </local>
</plan>
```

Plan 的定义是：

> persistent executable control state

而不是：

> 自由形式推理过程或 Chain-of-Thought

### 5.1 Plan 字段

- `global`：紧凑的路线骨架，不替代完整 instruction；
- `local`：当前滚动执行窗口；
- `done`：已经完成且不可修改；
- `current`：当前执行的计划点；
- `todo`：后续计划点；
- `abandoned`：第二阶段中被判定失效并放弃的计划点。

### 5.2 两阶段中的角色

第一阶段：

```text
Plan = read-only process reference
```

模型可以读取并理解 plan。这里的 read-only 指模型不能重写计划内容，但 controller
仍可执行正常的计划游标推进：

```text
current -> done
next todo -> current
```

模型不能：

- 修改 plan；
- 将 current 标为 abandoned；
- 创建 recovery point；
- 输出 `replan`。

第二阶段：

```text
Plan = editable recovery control state
```

当模型选择 `replan` 时，可以：

- 保留全部 `done` points；
- 放弃失效的 `current` point；
- 插入 recovery point；
- 更新 current/future plan。

### 5.3 更新约束

```text
1. done points are immutable.
2. replan may modify only current and future points.
3. a valid plan contains exactly one current point.
4. global is a route skeleton, not a replacement for the instruction.
5. local is a compact execution window, not full trajectory history.
6. Stage 1 may advance the plan cursor, but may not rewrite plan content.
```

---

## 6. 每步模型输入

每一步只使用固定预算上下文，不无限累积完整对话历史。Phase 0 先采用
JanusVLN-style expert imitation：每个 primitive expert decision 是一个独立样本；
从 episode 起点到当前时刻最多均匀采样 9 帧，并保证 current observation 在最后。
该最小动作基线通过后，Stage 1/2 再加入 compact plan、动作历史和 bounded loop。

```text
Full instruction
At most nine uniformly sampled observations from the episode prefix
Current high-resolution observation in the last image position
Allowed primitive actions
```

形式化表示：

\[
x_t=(I,O_{s_1:s_9=t},\mathcal A_t)
\]

其中：

- \(I\)：完整导航指令；
- \(O_{s_1:s_9=t}\)：从 episode prefix 均匀采样且以 current frame 结尾的视觉历史；
- \(\mathcal A_t\)：当前环境允许的 primitive actions。

Phase 0 固定配置：

```text
habitat_rgb = 640x480, hfov = 79
forward_step_size = 0.25m
turn_angle = 15 degrees
max_episode_steps = 500
success_distance = 3.0m
expert_waypoint_radius = 1.8m intermediate / 0.25m final
stored_model_frame = 384x288 JPEG
K_visual_total <= 9
current_frame_last = true
target_actions = 1
```

Phase 0 的训练与评测必须使用同一采样规则，并且每次只执行一个预测动作后重新
观察。Plan、progress、subgoal 和 tool 不属于该动作基线的 SFT target。

---

## 7. 共享 Backbone 与双头输出

共享 VLN/VLA backbone：

\[
H_t=f_\theta(x_t)
\]

之后分成两个输出分支。

### 7.1 Normal Token Head

保持基础模型原有的 token/action 生成能力：

\[
p_\theta(y_t\mid x_t)=\operatorname{Softmax}(W_{LM}H_t)
\]

第一阶段输出：

```xml
<progress>hold</progress>
<subgoal>continue along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

其中 `<progress>` 只能为 `hold` 或 `advance`。它只通知 controller 是否推进正常
计划游标，不是 recovery tool，也不能修改计划点文本。

第二阶段输出：

```xml
<tool>continue</tool>
<subgoal>continue along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

或：

```xml
<tool>replan</tool>
<plan_update>
  <abandon>p2</abandon>
  <current>leave the wrong room and return to the hallway</current>
  <future>continue toward the stairs</future>
</plan_update>
<subgoal>turn around and return through the doorway</subgoal>
<action>TURN_LEFT</action>
```

### 7.2 Risk MLP Head

在输入中加入专门的：

```text
<MONITOR>
```

取其最后一层 hidden state：

\[
h_t^{risk}=H_t[\texttt{<MONITOR>}]
\]

接两层 MLP：

\[
q_t=\sigma\left(W_2\operatorname{GELU}(W_1h_t^{risk})\right)
\]

其中：

\[
q_t\in[0,1]
\]

表示：

> 在当前 instruction、plan、视觉历史和动作历史条件下，如果不进行恢复、继续由当前策略执行，episode 最终失败的风险。

即：

\[
q_t=P(\text{episode fails under continue}\mid I,P_t,O_{\le t},A_{<t})
\]

偏离 expert route、goal progress、loop 和 stuck 只是训练期构造该目标的 oracle
proxy signals，不等同于 risk 本身。

Risk 通过 MLP 输出，不要求语言模型生成类似 `<risk>0.83</risk>` 的数值 token。

### 7.3 三状态解释

根据验证集阈值将连续 risk 映射为：

\[
z_t=
\begin{cases}
\text{ON\_TRACK}, & q_t<\tau_1\\
\text{AT\_RISK}, & \tau_1\le q_t<\tau_2\\
\text{OFF\_TRACK}, & q_t\ge\tau_2
\end{cases}
\]

三状态是对 risk 的解释层，而不是必须额外增加的语言输出。

---

# Part II：Stage 1——自主错误感知

## 8. 第一阶段目标

第一阶段只解决：

> 模型能否在不访问 expert trajectory、真实 pose、goal distance 或 oracle 标签的推理阶段，仅根据 instruction、read-only plan、视觉历史和动作历史，准确预测当前导航过程的失效风险？

第一阶段明确不解决：

- 是否选择 `continue/replan`；
- 如何生成 recovery subgoal；
- 如何更新 plan；
- 如何回到正确路线。

Risk head 在第一阶段是被动监控器：

```text
risk prediction does not affect environment action
```

即使 \(q_t=0.95\)，第一阶段评估时仍由原导航策略继续执行，以便独立评估检测能力。

---

## 9. 为什么选择 On-policy 数据

训练数据由当前导航策略 \(\pi_k\) 自己产生：

\[
\tau\sim\pi_k
\]

原因是模型需要识别的是自己的真实错误分布，而不是仅识别人工预设错误。

模型真实错误可能包括：

- 错过关键转向；
- 误认 landmark；
- 相似房间混淆；
- 错误认为某个 plan point 已完成；
- 连续小偏差累积；
- loop；
- stuck；
- 提前 STOP；
- 对错误计划保持高置信度。

采用迭代式数据聚合：

```text
π_k on-policy rollout
        ↓
收集当前模型的成功与失败过程
        ↓
oracle 生成风险标签
        ↓
监督训练 risk head
        ↓
π_{k+1} 重新 rollout
```

这是一种面向过程状态的 on-policy supervised aggregation，而不是纯 RL。

---

## 10. 第一阶段数据采集

每条 episode 记录：

```text
episode_id
instruction
initial_plan
step_id
current_observation
recent_visual_history
recent_actions
current_plan
policy_action
agent_pose (label-only)
expert_path (label-only)
goal_distance (label-only)
collision / loop / stop flags
final_success
```

重要约束：

```text
agent_pose、expert_path、goal_distance 只用于标签生成，绝不能进入模型输入。
```

数据应覆盖：

- expert/teacher 成功轨迹；
- 当前策略成功轨迹；
- 当前策略绕路成功轨迹；
- 当前策略自然失败轨迹；
- 可选的关键路口扰动轨迹。

主训练数据以 on-policy rollout 为主，人工扰动只用于补充稀有错误。

---

## 11. Risk 标签定义

### 11.1 第一版：可实现的 oracle proxy 标签

训练环境中计算：

- 当前 pose 到 expert route 的最短距离 \(D_t^{route}\)；
- 到目标的 geodesic distance 趋势 \(D_t^{goal}\)；
- 最近窗口的 progress gap \(G_t\)；
- loop/stagnation 指标；
- plan point 是否长期无进展。

构造标准化 proxy score：

\[
s_t=
 w_r\bar D_t^{route}
+w_g\bar D_t^{goal}
+w_p\bar G_t
+w_lL_t
+w_sS_t
\]

便宜的软风险目标：

\[
q_{t,proxy}^*=\operatorname{clip}(s_t,0,1)
\]

硬状态可由两个阈值得到：

```text
ON_TRACK  : q*_{t,proxy} < δ1
AT_RISK   : δ1 ≤ q*_{t,proxy} < δ2
OFF_TRACK : q*_{t,proxy} ≥ δ2 and sustained for M steps
```

### 11.2 标签时序设计

为了学习错误演化过程，标签不应只在最终偏离点瞬间跳变。

推荐：

```text
偏离前稳定窗口          → ON_TRACK
偏离刚出现、证据较弱      → AT_RISK
偏离持续且进度恶化        → OFF_TRACK
```

例如：

```text
t0 ON_TRACK
t1 ON_TRACK
t2 AT_RISK   # 错过转向
t3 AT_RISK   # 仍可能纠正
t4 OFF_TRACK # 进入错误房间并持续深入
```

### 11.3 第二版：Future-value refinement

几何偏离不一定等价于失败。对于边界或困难样本，可从同一 simulator state 使用当前策略采样 \(M\) 条 suffix rollouts：

\[
V_t^\pi=\frac{1}{M}\sum_{m=1}^{M}\mathbf 1[\tau_t^{(m)}\text{ succeeds}]
\]

无恢复条件下的 future-failure target：

\[
q_{t,value}^*=1-V_t^\pi
\]

对于执行 suffix rollout 的困难样本，以 \(q_{t,value}^*\) 作为更高质量监督；
其余大规模状态使用 \(q_{t,proxy}^*\)。如果实验中使用插值目标，应称为
`soft risk target`，而不是严格校准的失败概率：

\[
q_t^*=\alpha q_{t,proxy}^*+(1-\alpha)q_{t,value}^*
\]

Future-value refinement 仅用于：

- 偏离 expert 但走替代可行路线；
- 离 expert 很近但当前 plan 已明显失效；
- 几何信号与最终结果冲突；
- 验证集中的边界状态。

第一版不需要对所有 step 做昂贵 suffix rollout。

---

## 12. 第一阶段训练目标

### 12.1 Token loss

保持正常导航与输出格式：

\[
\mathcal L_{token}=-\log p_\theta(y_t^*\mid x_t)
\]

### 12.2 Risk regression/classification loss

对于软标签使用 BCE：

\[
\mathcal L_{risk}=\operatorname{BCE}(q_t,q_t^*)
\]

也可以比较 MSE、Huber 和 focal BCE。

### 12.3 Trajectory ranking loss

同一轨迹中，如果后续时刻真实风险更高：

\[
q_j^*>q_i^*
\]

要求：

\[
q_j>q_i
\]

排序损失：

\[
\mathcal L_{rank}=-\log\sigma(q_j-q_i)
\]

### 12.4 Temporal regularization

抑制无证据的单步剧烈振荡，同时允许真正偏航时快速上升：

\[
\mathcal L_{temp}=w_t|q_t-q_{t-1}|
\]

其中 \(w_t\) 在 oracle 状态转移点附近降低，避免过度平滑真实突变。

### 12.5 总损失

\[
\mathcal L_{Stage1}=
\mathcal L_{token}
+\lambda_r\mathcal L_{risk}
+\lambda_{rank}\mathcal L_{rank}
+\lambda_t\mathcal L_{temp}
\]

---

## 13. 第一阶段训练顺序

### 13.1 Probe：冻结 Backbone

只训练：

- `<MONITOR>` embedding；
- risk projection；
- risk MLP。

目的：

> 验证基础 VLN 表征中是否已经包含可用于风险判断的信息。

### 13.2 Lightweight adaptation

若冻结结果不足，使用 LoRA 或只解冻顶部若干层，联合训练 token loss 与 risk loss。

目的：

> 让模型形成更适合“指令—计划—历史一致性判断”的内部表征，同时尽量不破坏导航能力。

### 13.3 On-policy iteration

更新后的模型重新 rollout，收集新的错误类型，再训练下一轮 risk model。

建议先做 2–3 轮，而不是无限在线更新。

---

## 14. 第一阶段输出与推理

第一阶段模型每一步产生：

1. 正常 progress/subgoal/action token；
2. 独立 risk score \(q_t\)。

示例：

```text
Token head:
<progress>hold</progress>
<subgoal>continue through the hallway</subgoal>
<action>MOVE_FORWARD</action>

Risk head:
q_t = 0.78
state = AT_RISK / OFF_TRACK（由阈值解释）
```

第一阶段不产生：

```text
<tool>continue</tool>
<tool>replan</tool>
<plan_update>...</plan_update>
```

---

## 15. 第一阶段评估

### 15.1 Step-level 检测

- AUROC；
- AUPRC；
- OFF_TRACK precision/recall/F1；
- macro-F1；
- Brier Score；
- Expected Calibration Error。

### 15.2 Episode-level 检测

- 成功轨迹误报率；
- 失败轨迹检出率；
- 每 episode 报警次数；
- 风险状态切换次数。

### 15.3 检测提前量

设 oracle deviation onset 为 \(t^*\)，模型首次超过报警阈值为 \(\hat t\)：

\[
\Delta t=t^*-\hat t
\]

- \(\Delta t>0\)：提前发现；
- \(\Delta t=0\)：同步发现；
- \(\Delta t<0\)：发现太晚。

同时报告从真实偏航到报警的 delay，避免只追求过早报警。

### 15.4 核心消融

- 当前单帧；
- 当前帧 + instruction；
- 视觉历史 + instruction；
- 视觉历史 + action history；
- 加 read-only plan；
- 加 `<MONITOR>` token；
- 冻结 backbone vs LoRA；
- binary label vs soft risk；
- 只用 expert/success 数据 vs 加 on-policy failure；
- 仅 deviation 标签 vs future-value refinement。

---

# Part III：Stage 2——反事实恢复规划

## 16. 第二阶段目标

第二阶段只在第一阶段达到可接受检测性能后启动。

目标：

> 当模型内部 risk 表明当前执行可能失效时，学习判断继续当前计划还是执行恢复性重规划，并生成有效的 recovery plan update、subgoal 和动作。

第二阶段不再让 expert deviation 作为最终触发器；核心 candidate states 来自模型自己的 risk。

---

## 17. Tool Space

第二阶段引入：

```text
tool ∈ {continue, replan}
```

### 17.1 continue

```xml
<tool>continue</tool>
<subgoal>continue along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

含义：当前 plan 仍值得继续。

### 17.2 replan

```xml
<tool>replan</tool>
<plan_update>
  <abandon>p2</abandon>
  <current>leave the side room and return to the hallway</current>
  <future>continue toward the stairs</future>
</plan_update>
<subgoal>turn around and leave through the doorway</subgoal>
<action>TURN_LEFT</action>
```

含义：当前 plan 已失效，需要先恢复到可靠执行状态，再继续原 instruction。

### 17.3 STOP

`STOP` 始终是 primitive action，不是 tool。

---

## 18. Risk-guided Critical State

正常 rollout 中，controller 读取模型 risk：

\[
q_t=P(\text{execution failure risk}\mid x_t)
\]

候选 critical state 条件：

```text
q_t > τ_candidate
```

推荐使用持续性与迟滞机制：

```text
high risk for M consecutive steps
or
q_t > τ_emergency
```

训练时为避免 risk model 漏检，可混合少量 oracle hard states：

```text
critical states = learned-risk states + oracle false-negative states
```

但最终推理只使用 learned risk。

---

## 19. Counterfactual Branching

在 critical state 保存分层 checkpoint：

```text
EnvironmentCheckpoint:
  simulator state, agent state, episode id, elapsed steps, metrics

PolicyCheckpoint:
  current plan and cursor
  recent visual/action buffers
  controller mode/counters and risk history
  Python/NumPy/Torch/CUDA/decoder RNG states

Shared group context:
  instruction, episode metadata, reference trajectory
  one immutable trajectory prefix before the checkpoint
```

`continue/replan` 两个 branch 共享 group context，只分别保存 checkpoint 之后的
suffix，避免复制完整历史轨迹。

从同一 checkpoint 展开：

```text
             critical state s_t
              /             \
       force continue      force replan
              \             /
              short embodied rollouts
                      ↓
              compare future outcomes
```

第一版每个 normal rollout 最多触发一次 counterfactual group，每组包含一个
`continue` 分支和一个 `replan` 分支，branch 内禁止递归分支。需要降低 rollout
方差时，可在每个 tool 下采样多个 suffix，但不作为第一版硬性要求。

---

## 20. Continue Branch

第一步强制：

```xml
<tool>continue</tool>
```

规则：

- 不更新 plan；
- 根据当前 plan 生成 subgoal/action；
- 第一动作之后恢复正常 policy rollout。

---

## 21. Replan Branch

第一步强制：

```xml
<tool>replan</tool>
```

模型需要：

1. 保留全部 `done` points；
2. 放弃不可靠 current point；
3. 创建 recovery current point；
4. 保留与原 instruction 一致的 future points；
5. 输出一个合法 primitive action。

第一动作之后恢复正常 policy rollout。

---

## 22. Forced Prompt 与训练条件

Forced branch prompt 只用于生成候选分支：

```text
branch generation:
normal_prompt + forced_tool_instruction
```

Preference/policy loss 必须在正常 prompt 下计算：

```text
policy learning:
normal_prompt only
```

否则模型学到的是服从 forced instruction，而不是自主选择 tool。

---

## 23. Branch Horizon 与预算

建议第一版：

```text
normal rollouts per instruction = 2
counterfactual groups per normal rollout ≤ 1
branches per group = 2
branch horizon H = 30–50 steps
recursive branching = disabled
```

提前停止：

- success；
- wrong STOP；
- loop；
- stuck；
- no progress；
- invalid XML/action；
- excessive collisions；
- horizon reached。

---

## 24. Branch Evaluation

分支完整轨迹只用于评价，不直接作为完整 SFT target。

分支得分可写为：

\[
G(\tau)=
 w_sR_{success}
+w_gR_{goal-progress}
+w_rR_{route-recovery}
+w_pR_{plan-consistency}
-w_lC_{loop}
-w_cC_{collision}
-w_eC_{inefficiency}
-w_iC_{invalid}
\]

正向信号：

- success；
- goal progress；
- recovery progress；
- 距离可靠路线减少；
- 当前 plan 与实际观测重新一致；
- 路径效率。

负向信号：

- loop；
- stuck；
- wrong STOP；
- collision；
- 路径过长；
- 无效 action/XML；
- 无必要频繁 replan。

同一 checkpoint 的反事实组定义为：

\[
\mathcal G_t=\{\tau_t^{continue},\tau_t^{replan}\}
\]

首先得到两条分支的环境 return：

\[
R_c=G(\tau_t^{continue}),\qquad R_r=G(\tau_t^{replan})
\]

组内均值与相对优势为：

\[
\bar R_t=\frac{R_c+R_r}{2},\qquad
A_c=R_c-\bar R_t,\quad A_r=R_r-\bar R_t
\]

等价的恢复差值为：

\[
A_t^{rec}=G(\tau_t^{replan})-G(\tau_t^{continue})
\]

- \(A_t^{rec}>m\)：replan winner；
- \(A_t^{rec}<-m\)：continue winner；
- \(|A_t^{rec}|\le m\)：样本模糊，丢弃或降权。

---

## 25. 第二阶段训练目标

真正参与 counterfactual group-relative advantage optimization 更新的是 critical
state 下的第一步正常格式输出。

Replan 输出示例：

```xml
<tool>replan</tool>
<plan_update>
  <abandon>p2</abandon>
  <current>return to hallway entrance</current>
  <future>continue toward the stairs</future>
</plan_update>
<subgoal>leave the side room and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

Continue 输出示例：

```xml
<tool>continue</tool>
<subgoal>continue the current route</subgoal>
<action>MOVE_FORWARD</action>
```

主优化目标是在同一 critical-state group 内直接使用环境 return 的相对优势：

\[
\mathcal L_{CFRP}
=-
\sum_{y\in\{y_c,y_r\}}
A_y\log\pi_\theta(y\mid x_t)
\]

其中两份 log-prob 都必须在原始 normal prompt 下计算。Forced tool instruction
只用于生成候选分支，不进入 policy condition。

DPO / pairwise preference training 仅作为 baseline，用于验证组内相对优势是否
优于把反事实结果压缩成 chosen/rejected 标签。

第二阶段损失：

\[
\mathcal L_{Stage2}=
\mathcal L_{CFRP}
+\lambda_{format}\mathcal L_{format}
+\lambda_{plan}\mathcal L_{plan-update}
+\lambda_{action}\mathcal L_{action}
\]

Risk head 可保持冻结，或仅用高置信反事实结果做小学习率校准；不应在第二阶段随意改变其“偏航风险”语义。

---

## 26. Recovery Format Warm-up

在反事实训练前，使用少量监督样本让模型掌握：

- `continue/replan` XML 格式；
- plan update 规则；
- done points 不可修改；
- recovery current point 的生成；
- future plan 与原 instruction 保持一致；
- primitive action 合法性。

该阶段只教会“怎么表达和执行 recovery”，不用于决定“何时 recovery”。

---

## 27. 第二阶段推理

推理时不访问：

- expert trajectory；
- oracle deviation；
- ground-truth pose；
- counterfactual simulator branches。

每步执行：

```text
1. Shared backbone 生成正常 token 与 risk q_t
2. Controller 对 q_t 做平滑、迟滞和 cooldown 判断
3. 低风险：执行 continue 格式输出
4. 中风险：继续观察，累计证据
5. 高风险：允许模型选择 continue 或 replan
6. replan 后应用 plan_update 并进入 cooldown
```

建议门控：

```text
q_t < τ1                 → continue only
τ1 ≤ q_t < τ2            → observe / require persistence
q_t ≥ τ2                 → enable continue vs replan decision
q_t ≥ τ_emergency        → immediately enable continue/replan decision,
                           bypassing persistence
```

Risk 来自模型内部，因此不是外部 oracle trigger。

---

## 28. Controller 与 VLM 分工

### VLM 负责

- 正常 progress/subgoal/action；
- risk hidden representation；
- 第二阶段 tool 选择；
- recovery subgoal；
- compact plan update；
- primitive action。

### Controller 负责

- 构造固定预算 prompt；
- 维护完整 persistent plan；
- 校验 `progress` 并推进正常 plan cursor；
- 维护 recent visual/action history；
- 解析并校验 XML；
- 应用 plan update；
- 读取 risk score；
- 执行迟滞、cooldown 和 branch mode；
- 保存/恢复 simulator checkpoint；
- 训练期生成 oracle labels 和 counterfactual samples。

原则：

```text
Controller maintains deterministic state.
VLM makes semantic navigation and recovery decisions.
```

---

## 29. 完整训练流程

### Phase 0：Action-only Expert Imitation Bootstrap

训练或加载可用 VLN baseline，确保：

- 使用 R2R 真正 primitive expert actions；
- 每次只输出并执行一个合法 `<action>`；
- 训练/评测均使用最多 9 帧的均匀 episode-prefix 视觉输入；
- 在 val-unseen 上得到可重复且非退化的 SR/SPL baseline。

Phase 0 不训练 plan/progress/subgoal/tool。进入 Stage 1/2 前只增加少量格式预热，
使模型认识 compact plan 和 CFRP XML 接口；语义性的 plan/tool/recovery 决策仍由
后续监督与反事实强化学习获得。

### Phase 1：On-policy Risk Data Collection

```text
当前策略自由 rollout
        ↓
保存每个 step 的模型输入和 oracle-only metadata
        ↓
生成 soft risk / state labels
```

### Phase 2：Stage 1 Supervised Risk Training

```text
freeze backbone probe
        ↓
LoRA / top-layer adaptation
        ↓
2–3 rounds on-policy aggregation
```

### Phase 3：Stage 1 Evaluation Gate

只有当 detection 达到预设标准，才进入第二阶段。

### Phase 4：Recovery Format Warm-up

用少量 supervised recovery examples 学习 tool、plan update 和 XML 规则。

### Phase 5：Risk-guided Counterfactual Collection

```text
normal rollout
        ↓
learned risk selects candidate state
        ↓
        environment + policy checkpoint
        ↓
continue / replan short branches
        ↓
score and filter
```

### Phase 6：Counterfactual Group-relative Advantage Optimization

在正常 prompt 下回算 log-prob，使用同一 checkpoint 内 continue/replan 的组内
相对优势优化两份 first-step output。由于候选输出由 forced-tool prompt 生成，
该目标不宣称为经典 on-policy GRPO。DPO 只作为 baseline。

### Phase 7：End-to-end Iteration

```text
updated policy rollout
→ updated risk distribution
→ new critical states
→ new counterfactual pairs
→ retrain
```

---

## 30. 实验设计

### 30.1 Stage 1 baselines

- 单帧视觉分类器；
- instruction + current frame；
- history encoder；
- progress monitor；
- external oracle deviation detector；
- frozen backbone linear probe；
- 完整 Risk MLP。

### 30.2 Stage 2 baselines

- no recovery；
- oracle-triggered recovery；
- prompt-triggered recovery；
- learned-risk + heuristic replan；
- learned-risk + SFT recovery；
- learned-risk + DPO recovery baseline；
- learned-risk + counterfactual group-relative advantage optimization；
- full EA-CFRP-VLN。

### 30.3 解耦评估

分别报告：

1. **Detection quality**：是否发现了问题；
2. **Oracle-triggered recovery quality**：已知走错后是否能恢复；
3. **End-to-end quality**：自主发现与恢复结合后的效果。

这样可以定位失败来自：

- 没发现；
- 发现太晚；
- 误报；
- recovery decision 错误；
- recovery execution 失败。

---

## 31. 指标

### 常规 VLN

- SR；
- SPL；
- NE；
- nDTW / sDTW；
- path length；
- collision rate。

### Error Awareness

- AUROC；
- AUPRC；
- OFF_TRACK precision/recall/F1；
- episode false-positive rate；
- detection delay / lead time；
- ECE；
- Brier Score；
- risk-state transition count。

### Recovery

- recovery success rate；
- deviation recovery rate；
- unnecessary replan rate；
- recovery path efficiency；
- recovery latency；
- replan frequency；
- plan update validity；
- distance-to-route reduction。

---

## 32. 关键消融

### Stage 1

```text
w/o visual history
w/o action history
w/o read-only plan
w/o <MONITOR> token
binary risk vs soft risk
frozen backbone vs LoRA
w/o on-policy failures
w/o ranking loss
w/o temporal loss
w/o future-value refinement
```

### Stage 2

```text
oracle trigger vs learned risk trigger
w/o counterfactual branching
w/o plan state
w/o plan update
w/o recovery format warm-up
full plan output vs compact plan_update
w/o branch margin filtering
w/o cooldown / hysteresis
```

---

## 33. 核心创新点

### 创新点一：Endogenous Navigation Risk Head

在共享 VLN backbone 上增加 `<MONITOR>` 风险头，通过 on-policy 失败过程和 oracle 监督训练，使模型在推理时无需 expert path 即可评估自身执行风险。

### 创新点二：Plan-conditioned Process Monitoring

错误检测不是单帧异常识别，而是判断：

> 当前视觉与动作历史是否仍然支持 instruction 和 read-only plan 所定义的执行过程。

### 创新点三：On-policy Supervised Error Awareness

使用当前策略自己的 rollout 产生训练分布，并利用训练环境中已知的偏航信息做密集监督，避免仅靠稀疏终局奖励从零学习错误意识。

### 创新点四：Risk-guided Counterfactual Recovery

由 learned risk 选择 embodied critical states，在同一物理状态上比较 `continue` 和 `replan` 的未来结果。

### 创新点五：Detection–Recovery Decoupling

将“能否发现问题”和“能否解决问题”分开训练、分开评估，再组合成完整闭环，提高方法可解释性与实验可信度。

---

## 34. 核心 Claim

英文：

> EA-CFRP-VLN decouples recovery in continuous VLN into endogenous error awareness and counterfactual recovery planning. A shared navigation backbone is augmented with a risk head trained by oracle-supervised on-policy trajectories to estimate the future failure risk of continuing without recovery. High-risk states identified by the model are then expanded into same-state continue-versus-replan rollouts, whose group-relative environment advantages train when and how to recover without access to expert trajectories at inference time.

中文：

> EA-CFRP-VLN 将连续 VLN 的恢复问题拆分为自主错误感知与反事实恢复规划。模型首先基于当前策略产生的 on-policy 轨迹和训练环境中的 oracle 标签，以监督学习训练共享 backbone 上的风险预测头，估计不进行恢复而继续当前执行时的未来失败风险；随后在模型自己识别的高风险状态上展开同起点的 `continue` 与 `replan` 反事实分支，以组内环境相对优势学习何时恢复以及如何更新计划。推理阶段不使用 expert trajectory、外部错误提示或反事实 simulator。

---

## 35. 最终方法短版

```text
EA-CFRP-VLN maintains a persistent executable plan and augments a VLN backbone with two outputs: a normal token/action head and a scalar risk head. In Stage 1, the current policy generates on-policy trajectories without recovery intervention. Oracle information available only in training provides scalable proxy labels, while selected difficult states are refined with suffix-rollout estimates of future failure under continue. The risk head learns this continue-failure risk while the controller may advance the normal plan cursor, but no recovery tool or structural plan update is learned. In Stage 2, model-predicted high-risk states are checkpointed and expanded into same-state continue and replan branches. Group-relative environment advantages directly train tool choice, recovery subgoals, plan updates, and primitive actions under the normal prompt. At inference time, the agent detects risk and performs recovery using only its instruction, plan, visual history, and action history.
```
