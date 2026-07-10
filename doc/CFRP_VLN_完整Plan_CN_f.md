# CFRP-VLN 完整中文版 Plan.md

## 1. 方法名称

**CFRP-VLN**  
**Counterfactual Recovery Planning for Continuous Vision-and-Language Navigation**

中文名称：

> 面向连续 VLN 的反事实恢复规划

一句话概括：

> CFRP-VLN 将连续 VLN 中的走偏恢复问题建模为反事实 plan-tool 决策问题。训练时利用 expert trajectory 在线检测 agent 当前轨迹是否偏离，并在同一偏离状态下比较 `continue` 与 `replan` 两种决策的长期结果，使 VLM 学会何时坚持当前计划、何时主动恢复重规划。

---

## 2. 研究问题

连续 VLN 中，agent 常见的问题不是完全不会执行 instruction，而是：

```text
早期一步走错
    ↓
当前 observation 已经不支持原 plan
    ↓
模型仍然继续执行旧 plan
    ↓
错误累积
    ↓
loop / stuck / wrong route / failure
```

例如 instruction 是：

```text
Exit the bedroom, turn left into the hallway, continue until the stairs,
turn right, enter the kitchen, and stop beside the sink.
```

Expert route 是：

```text
bedroom -> hallway -> stairs -> kitchen
```

Agent 实际走成：

```text
bedroom -> side room
```

此时旧 plan 仍然认为：

```text
current: walk through hallway
```

如果 agent 继续执行旧 plan，就会在错误状态中越走越远。

CFRP-VLN 解决的问题是：

> 当前 plan 已经不可靠时，agent 如何识别并主动生成恢复性的 replan，而不是盲目 continue。

---

## 3. 核心思想

CFRP-VLN 的核心不是让模型输出更长的推理，也不是让模型显式写 CoT，而是引入一个轻量的 plan-tool 决策接口：

```text
tool ∈ {continue, replan}
```

其中：

- `continue` 表示当前 plan 仍然可靠，继续执行当前 subgoal；
- `replan` 表示当前 plan 已经不可靠，需要更新 current/future plan，并生成 recovery subgoal。

训练时，CFRP-VLN 在 expert-guided deviation state 上构造反事实分支：

```text
同一个 critical state
        /          \
 continue          replan
```

然后通过短 rollout 评估两个分支的长期结果，并用结果构造 preference/RL 信号。

---

## 4. 方法定位

CFRP-VLN 面向连续 VLN，例如：

- VLN-CE
- R2R-CE
- RxR-CE
- Habitat-based continuous navigation

本文方法关注的是：

> continuous unseen environment 中 agent 的 plan reliability 与 recovery behavior。

CFRP-VLN 不是：

- prompt engineering；
- Chain-of-Thought 生成；
- 单纯 imitation learning；
- 普通多采样 group rollout；
- 每一步全局重新规划。

CFRP-VLN 是：

> 在连续 VLN 的偏离状态上，对 `continue` 与 `replan` 进行反事实干预和偏好优化的 agentic RL 方法。

---

## 5. Tool Space

最终 tool space 只保留两个工具：

```text
continue
replan
```

### 5.1 continue

含义：

> 当前 plan 仍然可信，继续执行当前 plan/subgoal。

典型输出：

```xml
<tool>continue</tool>
<subgoal>continue along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

### 5.2 replan

含义：

> 当前 plan 不再可信，需要更新 current/future plan，并生成 recovery-oriented subgoal。

典型输出：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="abandoned">walk through hallway</p>
    <p id="r1" status="current">return to hallway entrance</p>
    <p id="p3" status="todo">continue toward stairs</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

### 5.3 STOP 不是 tool

`STOP` 不属于 tool space。

`STOP` 是环境 primitive action，只能出现在 `<action>` 中。

正确：

```xml
<tool>continue</tool>
<subgoal>stop beside the sink</subgoal>
<action>STOP</action>
```

错误：

```xml
<tool>stop</tool>
```

原因：

> 本方法的核心问题是 plan 是否仍可靠，即 continue vs replan，而不是是否终止任务。

---

## 6. Action Space

Action space 由具体 benchmark/environment 决定，不在 system prompt 中硬编码。

例如 Habitat VLN-CE 中常见 primitive actions 是：

```text
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

也可能包含：

```text
LOOK_UP
LOOK_DOWN
```

最终规则：

```text
每一步 prompt 必须显式给出 Allowed actions。
模型的 <action> 必须严格来自当前 Allowed actions。
```

---

## 7. Persistent Plan State

CFRP-VLN 使用一个持久化的 plan state。

Plan 是：

> persistent executable control state

不是：

> Chain-of-Thought reasoning trace

也就是说，plan 是外部 controller 和 VLM 共享的结构化控制记忆，用于记录当前任务执行到哪里、当前 subgoal 是什么、哪些目标已经完成、哪些未来目标仍需执行。

### 7.1 Plan 结构

典型 plan：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="current">walk through hallway</p>
    <p id="p3" status="todo">reach stairs</p>
  </local>
</plan>
```

### 7.2 global

`<global>` 是 compact route skeleton。

例如：

```text
bedroom -> hallway -> stairs -> kitchen
```

它用于提供路线结构假设，但不替代完整 instruction。

完整 instruction 每一步仍然输入给 VLM。

### 7.3 local

`<local>` 是 rolling execution window。

每个 plan point 有状态：

```text
done
current
todo
abandoned
```

含义：

- `done`：已经完成，不可修改；
- `current`：当前正在执行；
- `todo`：未来计划点；
- `abandoned`：当前计划点已失效或不可靠，被放弃。

### 7.4 Plan 更新规则

核心规则：

```text
1. done points are immutable.
2. replan may only modify current and future plan points.
3. each valid plan has exactly one current point.
4. global is route skeleton, not instruction replacement.
5. local is a compact executable window, not full trajectory history.
```

当 agent 偏离原路线时，例如进入 side room：

原 plan：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="current">walk through hallway</p>
    <p id="p3" status="todo">reach stairs</p>
  </local>
</plan>
```

replan 后：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="abandoned">walk through hallway</p>
    <p id="r1" status="current">return to hallway entrance</p>
    <p id="p3" status="todo">continue through hallway toward stairs</p>
  </local>
</plan>
```

---

## 8. Controller 与 VLM 的分工

### 8.1 VLM 负责

VLM 负责输出：

```text
tool
subgoal
action
optional plan update when replan
```

即：

- 判断当前 plan 是否可靠；
- 选择 `continue` 或 `replan`；
- 生成短的 executable local subgoal；
- 选择一个 primitive action；
- 在 `replan` 时给出更新后的 plan 或 compact plan update。

### 8.2 Controller 负责

Controller 负责：

- 构造每一步 prompt；
- 维护 persistent plan state；
- 解析 XML；
- 校验 tool/action 是否合法；
- 应用 plan update；
- 执行动作到 Habitat；
- 维护 recent history/recent actions；
- 维护 recovery cooldown；
- 训练时触发 counterfactual branch。

### 8.3 为什么这样分工

不要强迫 4B VLM 每一步都完整管理复杂状态。

更合理的设计是：

```text
Controller maintains full structured state.
VLM makes semantic plan-tool decisions.
```

因此，论文中可以展示完整 plan state；实现时可以允许 VLM 在 replan 时输出 compact plan update，由 controller 映射到完整 plan state。

---

## 9. VLM 输出格式

### 9.1 Normal continue 输出

当当前 plan 仍可靠时：

```xml
<tool>continue</tool>
<subgoal>move forward along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

### 9.2 Replan 输出

当当前 plan 不可靠时：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="abandoned">walk through hallway</p>
    <p id="r1" status="current">return to hallway entrance</p>
    <p id="p3" status="todo">continue toward stairs</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

### 9.3 轻量 replan 输出

实现中也可以使用更轻的 patch 格式：

```xml
<tool>replan</tool>
<plan_update>
  <abandon>p2</abandon>
  <current>return to hallway entrance</current>
  <future>continue through hallway toward stairs</future>
</plan_update>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

Controller 负责将其应用到完整 plan state。

### 9.4 输出约束

```text
1. 只输出 XML。
2. 不输出 free-form explanation。
3. 不输出 CoT。
4. <action> 必须是一个 allowed action。
5. 不输出多个 action。
6. STOP 只能作为 action，不能作为 tool。
7. continue 时不重复完整 plan。
8. replan 时才输出 plan 或 plan_update。
```

---

## 10. 每一步输入 Prompt

每一步 VLM call 都使用固定预算上下文，不累积完整 400-turn 历史。

### 10.1 每步输入字段

```text
System prompt
Full instruction
Current observation
Recent visual history
Recent actions
Current compact plan
Allowed actions
Optional active instruction excerpt
```

其中：

- full instruction：每一步都输入；
- current observation：当前高分辨率 RGB；
- recent visual history：最近 3-5 帧低分辨率图像或 caption；
- recent actions：最近 4-8 个动作；
- current compact plan：当前 plan state；
- allowed actions：当前环境允许的 action；
- active instruction excerpt：可选，用于强调当前 instruction span，但不能替代完整 instruction。

### 10.2 不使用完整对话历史

错误做法：

```text
把过去 400 步所有 prompt/output 都放进上下文
```

正确做法：

```text
Environment interaction 可以持续 400+ steps，
但每次 VLM call 只接收固定大小的 step-level context。
```

---

## 11. System Prompt

System prompt 定义模型角色、工具、输出格式、plan 规则和 action 规则。

```text
You are a continuous Vision-and-Language Navigation planner.

Your task is to navigate in a continuous environment according to the given natural language instruction.

You maintain a persistent structured plan state.

The plan is not a reasoning trace.
The plan is an executable control state maintained by an external controller.

Output only XML.
Do not explain reasoning.
Do not output free-form analysis.

Available tools:
- continue: follow the current reliable plan without changing it.
- replan: update the current and future plan when the current plan is no longer reliable.

Stop is not a tool.
STOP is an environment action and may only appear in <action>.

Plan rules:
1. A plan contains <global> and <local>.
2. <global> is a compact route skeleton.
3. <global> does not replace the full instruction.
4. <local> is a rolling execution window.
5. Local plan points may have status:
   - done
   - current
   - todo
   - abandoned
6. Done points are immutable.
7. Replan may only modify current and future plan points.
8. A valid plan must contain exactly one current point.

Action rules:
1. <action> must contain exactly one executable action.
2. <action> must be selected from Allowed actions in the current step prompt.
3. Do not invent new actions.
4. Do not output multiple actions.
5. STOP, if used, is an action, not a tool.

Recovery policy:
If the current observation and recent history no longer support the current plan,
do not blindly continue.
Use <tool>replan</tool> to create a recovery subgoal.
A recovery subgoal may return to a reliable route location, such as leaving a wrong room and returning to the hallway, then resume the original instruction.

Output schema during normal execution:
<tool>continue</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>

Output schema during replanning:
<plan>
  <global>...</global>
  <local>
    <p id="..." status="done/current/todo/abandoned">...</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>short executable recovery or local instruction</subgoal>
<action>one allowed action</action>
```

---

## 12. Normal Step Prompt

Normal step prompt 每一步由 controller 重建。

模板：

```text
Full instruction:
{instruction}

Current observation:
{current_high_res_rgb}

Recent visual history:
{recent_low_res_frames_or_captions}

Recent actions:
{last_4_to_8_actions}

Current compact plan:
{current_plan_or_none}

Allowed actions:
{allowed_actions_from_environment}

Reminder:
Output XML only.
Choose either <tool>continue</tool> or <tool>replan</tool>.
Use <tool>replan</tool> only when the current plan is no longer reliable.
```

示例：

```text
Full instruction:
Exit the bedroom, turn left into the hallway, continue until the stairs,
turn right, enter the kitchen, and stop beside the sink.

Current observation:
[HIGH-RES RGB]
You are inside a small side room. A doorway back to the hallway is behind-left.

Recent visual history:
[LOW-RES t-3] hallway with painting
[LOW-RES t-2] side doorway on the right
[LOW-RES t-1] entering side room

Recent actions:
MOVE_FORWARD, TURN_RIGHT, MOVE_FORWARD, TURN_LEFT

Current compact plan:
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="current">walk through hallway</p>
    <p id="p3" status="todo">reach stairs</p>
  </local>
</plan>

Allowed actions:
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP
```

对应 replan 输出：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="done">exit bedroom</p>
    <p id="p2" status="abandoned">walk through hallway</p>
    <p id="r1" status="current">return to hallway entrance</p>
    <p id="p3" status="todo">continue toward stairs</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

---

## 13. 推理阶段 Inference

Inference 阶段没有 expert trajectory。

流程：

```text
Full instruction + observation + recent history + recent actions + plan
        ↓
VLM Planner
        ↓
XML: tool + subgoal + action
        ↓
Controller parses XML
        ↓
Execute action in Habitat
        ↓
Update observation/history/plan
        ↓
Next step
```

Inference 时允许多次 replan：

```text
continue
continue
replan
continue
continue
replan
continue
```

但是必须设置 replan cooldown，避免频繁重规划：

```text
replan
走一步
replan
走一步
replan
```

建议：

```text
after replan, disable or penalize another replan for N steps
```

其中 N 可以取 5-20，取决于任务长度和环境复杂度。

---

## 14. 训练阶段总览

训练阶段使用 expert trajectory，但 expert trajectory 不暴露给测试时模型。

Expert trajectory 用于：

```text
1. deviation detection
2. critical state trigger
3. branch evaluation / reward shaping
```

不是直接作为模型输入。

训练核心：

```text
normal rollout
    ↓
online deviation detection against expert trajectory
    ↓
critical state
    ↓
counterfactual continue branch vs replan branch
    ↓
branch evaluation
    ↓
preference RL
```

---

## 15. Normal Rollout 策略

每个 instruction 采样两个 normal rollouts：

```text
instruction
    ├── normal rollout 1
    └── normal rollout 2
```

每条 rollout 中 agent 正常执行当前 policy：

```text
s0 -> s1 -> s2 -> ... -> st
```

每一步都在线比较：

```text
agent trajectory vs expert trajectory
```

如果 agent 没有明显偏离 expert route，则不触发 counterfactual。

如果 agent 成功完成任务且没有触发偏离，则该 episode 只贡献普通 RL/success 信号，不贡献 recovery preference pair。

---

## 16. Deviation Trigger

触发 counterfactual 的核心依据是 expert-guided deviation。

触发条件：

```text
deviation(agent_t, expert_path) > threshold
```

可以使用以下指标：

### 16.1 Distance to expert path

计算 agent 当前 pose 到 expert trajectory 最近点的距离。

如果距离超过阈值，说明 agent 已明显离开 expert route。

### 16.2 Progress gap

比较 agent 当前进度与 expert trajectory 上对应进度。

如果 agent 长时间没有沿 expert route 前进，或 goal progress 明显落后，则可能触发。

### 16.3 Combined trigger

可以组合：

```text
D_t = distance_to_expert_path(agent_t)
G_t = progress_gap(agent_t, expert_path)

trigger if:
D_t > tau_d
or
G_t > tau_g
or
D_t increasing for K steps
```

核心原则：

```text
触发是 online 的。
不是 episode 结束后的 hindsight 分析。
```

---

## 17. Counterfactual Branching

当 normal rollout 在 t 时刻触发 critical state：

```text
s_t = critical state
```

立即 checkpoint 当前状态：

```text
Habitat agent state
current compact plan
recent visual history
recent actions
controller memory
```

不需要保存整条 400-step trajectory。

不需要保存每一步 Habitat state。

只保存当前 critical state checkpoint。

然后从同一个 checkpoint 展开两个分支：

```text
             critical state s_t
              /             \
       continue branch      replan branch
```

---

## 18. Continue Branch

Continue branch 第一部强制：

```xml
<tool>continue</tool>
```

含义：

```text
继续相信当前 plan，不更新 plan。
```

第一步之后，后续 rollout 使用 normal policy。

即：

```text
force first tool = continue
then rollout normally
```

---

## 19. Replan Branch

Replan branch 第一部强制：

```xml
<tool>replan</tool>
```

要求模型：

```text
1. 保留 done plan points
2. 放弃不可靠 current point
3. 创建 recovery subgoal
4. 更新 current/future plan
5. 输出一个 allowed action
```

第一步之后，后续 rollout 使用 normal policy。

即：

```text
force first tool = replan
then rollout normally
```

---

## 20. Forced Branch Prompt

Forced branch prompt 只在 RL 采样分支时使用。

### 20.1 Continue forced prompt

```text
For this counterfactual branch, force the first tool decision to be:
<tool>continue</tool>

Do not update the plan in this first step.
Follow the current plan and output one valid action.
```

### 20.2 Replan forced prompt

```text
For this counterfactual branch, force the first tool decision to be:
<tool>replan</tool>

Update the current and future plan.
Preserve all done plan points.
Create a recovery subgoal first, then resume the original instruction.
Output one valid action.
```

### 20.3 重要训练规则

Forced branch prompt 只用于生成分支候选。

Policy learning / preference loss 必须在 normal prompt 下计算。

即：

```text
branch generation:
normal_prompt + forced_branch_prompt

policy learning:
normal_prompt only
```

否则模型学到的是服从 forced instruction，而不是在正常状态下自主判断 `continue` 或 `replan`。

---

## 21. Branch Rollout Mode

系统有两种 rollout mode：

### 21.1 NORMAL mode

用于正常 policy rollout。

特点：

```text
monitor expert deviation
may trigger one counterfactual branch
```

### 21.2 BRANCH_ROLLOUT mode

用于 continue/replan 分支评估。

特点：

```text
counterfactual trigger disabled
no recursive branching
```

这样避免：

```text
critical state
  -> replan branch
      -> another critical state
          -> branch again
```

---

## 22. Branch 截断策略

Branch rollout 不跑完整 episode，而是短 horizon。

建议：

```text
H = 30-50 steps
```

提前停止条件：

```text
success
wrong stop
loop
stuck
no progress
too many collisions
invalid XML/action
horizon reached
```

目标不是必须在 branch horizon 内完成整个 instruction，而是判断：

```text
该分支是否呈现 recovery trend
```

例如 agent 进入 side room 后，replan branch 如果能在 30 步内：

```text
leave side room
return hallway
reduce distance to expert path
increase goal progress
```

就可以视为有效 recovery signal。

---

## 23. 采样预算

最终采样预算：

```text
per instruction:
    2 normal rollouts

per normal rollout:
    at most 1 counterfactual trigger

per trigger:
    2 branches:
        continue
        replan
```

因此每个 instruction 最多：

```text
2 counterfactual groups
4 branch trajectories
```

加上 2 条 normal rollout，整体成本可控。

明确不采用：

```text
4-way branch
recursive branch
branch every deviation
full episode branch rollout
```

---

## 24. Recovery Mode 与 Cooldown

如果 normal rollout 中模型自己输出了：

```xml
<tool>replan</tool>
```

说明模型正在尝试 recovery。

此时不要立刻触发外部 counterfactual。

进入 recovery mode：

```text
if last_tool == replan:
    enter recovery cooldown
    disable external trigger temporarily
```

cooldown 结束后，如果 agent 仍然严重偏离 expert trajectory，可以再次考虑触发。

但训练设定中每条 normal rollout 最多触发一次 counterfactual，因此即使 cooldown 后仍然偏离，也不在同一 rollout 中再次 branch。

Inference 时也使用 cooldown，避免 replan 过于频繁。

---

## 25. Branch Evaluation

两个 branch 的完整 trajectory 只用于评分，不直接作为 SFT target。

评分信号包括：

### 25.1 正向指标

```text
success
goal progress
expert route alignment
recovery progress
path efficiency
```

其中最关键的是：

```text
expert route alignment
```

也就是 branch 是否让 agent 回到 expert route 附近。

### 25.2 负向指标

```text
collision
loop
stuck
wrong stop
path inefficiency
invalid XML
invalid action
too frequent replan
```

### 25.3 示例

Continue branch：

```text
agent remains in side room
distance to expert path increases
no goal progress
score = -0.3
```

Replan branch：

```text
agent leaves side room
returns near hallway
distance to expert path decreases
goal progress improves
score = 0.6
```

得到 preference：

```text
replan > continue
```

---

## 26. Preference RL 信号

训练时不模仿完整 branch trajectory。

真正参与 preference/RL update 的是：

```text
critical state 下的第一步 XML output
```

例如 winner：

```xml
<tool>replan</tool>
<subgoal>return to hallway entrance</subgoal>
<action>TURN_LEFT</action>
```

loser：

```xml
<tool>continue</tool>
<subgoal>continue current hallway route</subgoal>
<action>MOVE_FORWARD</action>
```

训练目标：

```text
log π(winner_xml | normal_prompt(s_t))
>
log π(loser_xml | normal_prompt(s_t))
```

注意：

```text
forced branch prompt is not used in the training condition.
```

否则模型只会学会在被强制时输出对应 tool。

---

## 27. 样本过滤

不是所有 counterfactual group 都用于训练。

保留条件：

```text
winner score sufficiently high
score margin > threshold
winner XML valid
winner action valid
branch rollout long enough to evaluate
```

丢弃条件：

```text
both branches similarly good
both branches similarly bad
score margin too small
winner XML invalid
winner action invalid
branch outcome ambiguous
```

这样保证训练信号清晰。

---

## 28. Training Mask

训练时输入 tokens 不参与 loss。

```text
System prompt                  mask = 0
Full instruction               mask = 0
Current observation            mask = 0
Recent visual history          mask = 0
Recent actions                 mask = 0
Current compact plan           mask = 0
Allowed actions                mask = 0
Forced branch prompt           mask = 0

Assistant XML output           mask = 1
```

Preference/RL 中计算：

```text
log π(XML_output | normal_prompt)
```

不要让模型学习复述 prompt。

---

## 29. 完整训练算法伪代码

```text
for each instruction in training set:

    expert_path = load_expert_trajectory(instruction)

    for rollout_id in {1, 2}:

        mode = NORMAL
        branch_used = False
        reset environment
        initialize compact plan
        initialize recent history/actions

        for t in range(max_steps):

            prompt = build_normal_step_prompt(
                full_instruction,
                current_observation,
                recent_history,
                recent_actions,
                current_plan,
                allowed_actions
            )

            xml = policy.generate(prompt)
            tool, subgoal, action, optional_plan = parse(xml)

            if tool == replan:
                apply_plan_update(optional_plan)
                enter_recovery_cooldown()

            execute(action)
            update_recent_history_actions_plan()

            if success or failure:
                break

            if mode == NORMAL and not branch_used:
                if not in_recovery_cooldown():
                    deviation = compute_deviation(agent_state, expert_path)

                    if deviation > threshold:
                        checkpoint = save_current_state_and_memory()

                        continue_output, continue_traj = run_branch(
                            checkpoint,
                            forced_tool = continue,
                            horizon = H,
                            disable_recursive_branch = True
                        )

                        replan_output, replan_traj = run_branch(
                            checkpoint,
                            forced_tool = replan,
                            horizon = H,
                            disable_recursive_branch = True
                        )

                        score_continue = evaluate_branch(continue_traj, expert_path)
                        score_replan = evaluate_branch(replan_traj, expert_path)

                        if valid_preference(score_continue, score_replan):
                            add_preference_pair(
                                normal_prompt = prompt,
                                winner_output,
                                loser_output
                            )

                        branch_used = True
```

---

## 30. 与普通 group rollout / GRPO 的区别

CFRP-VLN 不声称发明 group rollout。

普通 group sampling 往往是：

```text
same prompt -> multiple sampled answers
```

CFRP-VLN 的核心是：

```text
same embodied navigation state
    -> force continue
    -> force replan
    -> compare long-term environment outcomes
```

也就是说，我们比较的不是普通文本答案，而是：

> 同一个物理偏离状态下，不同 plan-tool decision 对未来导航结果的影响。

因此创新点不是 group rollout 本身，而是：

```text
expert-guided deviation state
+
counterfactual plan-tool intervention
+
recovery-oriented preference RL
```

---

## 31. 实验设置建议

### 31.1 Benchmark

优先考虑：

```text
R2R-CE
RxR-CE
VLN-CE
Habitat
```

### 31.2 Base Model

可以使用：

```text
Qwen3-VL-4B
```

但建议先做：

```text
1. XML/action format warmup
2. navigation imitation / SFT warmup
3. simple recovery examples
4. counterfactual RL
```

不要期望模型从零开始直接学会复杂 recovery。

### 31.3 Metrics

常规 VLN 指标：

```text
SR
SPL
NE
nDTW / sDTW
path length
collision rate
```

Recovery-specific 指标：

```text
deviation recovery rate
distance-to-expert reduction
successful replan rate
unnecessary replan rate
replan frequency
cooldown violation rate
```

### 31.4 Ablation

建议 ablation：

```text
w/o plan state
w/o replan tool
w/o expert-guided trigger
w/o counterfactual preference
w/o recovery cooldown
continue/replan without branch evaluation
full plan output vs compact plan update
```

---

## 32. 最终创新点

### 创新点 1：Persistent Executable Plan State

提出结构化、持久化的 plan state，用于连续 VLN 的执行控制。

它不是 CoT，而是 control memory。

包含：

```text
global route skeleton
local execution window
done/current/todo/abandoned
```

作用：

> 让 agent 显式维护当前任务进度和 plan reliability，从而支持恢复性重规划。

---

### 创新点 2：Expert-guided Deviation Trigger

利用 benchmark expert trajectory 自动发现当前 plan 失效状态。

不是人工 failure 标注。

不是随机扰动。

不是 episode-end hindsight。

而是：

```text
online compare agent trajectory with expert trajectory
trigger when deviation exceeds threshold
```

作用：

> 在真正需要恢复的位置触发反事实比较。

---

### 创新点 3：Counterfactual Recovery RL

在同一个 deviation state 下比较：

```text
continue vs replan
```

通过短 horizon branch rollout 评估两种 tool decision 的长期导航结果。

作用：

> 让 VLM 学会何时坚持当前 plan，何时主动 recovery replan。

---

### 创新点 4：轻量化分支与截断策略

CFRP-VLN 不做大规模 tree search。

采用：

```text
2 normal rollouts per instruction
1 branch per rollout
2 branches per group
30-50 step horizon
no recursive branching
```

作用：

> 在保证训练信号有效的同时控制 Habitat + VLM rollout 成本。

---

### 创新点 5：Recovery-oriented Replanning

`replan` 不是重新生成整条路线，而是：

```text
preserve done points
abandon unreliable current point
insert recovery subgoal
resume original instruction
```

作用：

> 强调从错误状态恢复到可靠路线，而不是无约束重新规划。

---

## 33. 最终 Claim

英文：

> CFRP-VLN formulates recovery in continuous VLN as a counterfactual plan-tool decision problem. During RL training, expert trajectories identify deviation states where the current plan becomes unreliable. From each deviation state, the agent compares continuing the current plan against recovery-oriented replanning through short counterfactual rollouts, and optimizes the VLM planner to choose the better decision under the normal prompt.

中文：

> CFRP-VLN 将连续 VLN 中的恢复问题建模为反事实 plan-tool 决策问题。训练时利用 expert trajectory 检测当前 plan 失效的偏离状态，并从同一状态展开 `continue` 与 recovery-oriented `replan` 两个短反事实分支，通过环境结果构造偏好信号，使 VLM 学会何时坚持当前计划、何时主动恢复重规划。

---

## 34. 最终关键约束

```text
Tool:
continue / replan

Stop:
action，不是 tool

Plan:
persistent executable control state，不是 CoT

Prompt:
每步输入 full instruction + current observation + recent history + recent actions + current compact plan + allowed actions

Trigger:
agent-expert deviation > threshold

Branch:
continue vs replan

Budget:
2 normal rollouts per instruction
1 branch per rollout
2 branches per counterfactual group

Branch horizon:
30-50 steps

Branch mode:
no recursive branching

Inference:
no expert trajectory
multiple replans allowed
cooldown enabled

Training:
expert only for deviation detection and branch evaluation
branch trajectory only for scoring
first-step XML output used for preference RL
forced branch prompt only for branch generation
loss computed under normal prompt
```

---

## 35. 最终方法短版

```text
CFRP-VLN maintains a persistent executable plan state and lets a VLM choose between continue and replan at each step. During training, expert trajectories are used to detect deviation states online. At each detected critical state, the current environment and memory are checkpointed, and two short counterfactual branches are rolled out: one forcing continue and one forcing replan. The branch outcomes are scored by success, goal progress, expert-route alignment, and recovery progress. The resulting preference pair trains the VLM to choose the better first-step XML output under the normal prompt. At inference time, no expert trajectory is used; the agent relies on the learned continue/replan policy and a recovery cooldown mechanism.
```
