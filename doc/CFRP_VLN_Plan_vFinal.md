# CFRP-VLN Plan.md

## 1. 方法名称

**CFRP-VLN**  
**Counterfactual Recovery Planning for Continuous Vision-and-Language Navigation**

中文名称：

> 面向连续 VLN 的反事实恢复规划

一句话概括：

> CFRP-VLN 将连续 VLN 中的可恢复导航建模为稀疏持久计划状态重规划问题。VLM 每轮接收完整原始 instruction、当前高分辨率观测、低分辨率历史关键帧、最近动作和 compact plan，并通过反事实 rollout 学习何时继续、直接重规划、恢复式重规划或停止。

---

## 2. 方法定位

CFRP-VLN 面向 **continuous VLN**，例如 VLN-CE / R2R-CE / RxR-CE 一类连续环境。

核心问题：

```text
agent 在连续环境中执行 instruction 时容易走偏；
走偏后如果继续执行旧 plan，会出现 loop、stuck、wrong stop 或长绕路；
我们希望 agent 能学会发现当前 plan 不可靠，并通过 replan 恢复到正轨。
```

本方法不是 Chain-of-Thought，也不是单纯 prompt engineering。

核心区别：

```text
<plan> 是持久控制状态，不是每步 reasoning trace；
<subgoal> 是局部可执行目标，不是解释；
<action> 是环境可执行 primitive action；
训练信号来自环境 rollout outcome，而不是语言文本好坏。
```

关键表述：

> The plan is a persistent control state, not a per-step reasoning trace.

---

## 3. 总体交互方式

CFRP-VLN 是一个 **multi-turn VLM agent loop**。

每一轮：

```text
current observation
-> build VLM prompt
-> VLM outputs XML
-> controller parses XML
-> execute action in environment
-> update plan/history
-> next turn
```

每轮都重新输入完整 instruction。  
不要只在最开始 system prompt 放一次 instruction。  
不要压缩 instruction。

原因：

```text
VLN instruction 中很多细节会影响最终导航：
- walk past the painting
- turn right after the stairs
- stop beside the sink near the window

如果压缩成 "go to the kitchen"，模型后期容易漂移。
```

因此每轮输入采用：

```text
Full instruction:
  每轮完整输入，不压缩。

Current observation:
  高分辨率当前图像。

Recent visual history:
  低分辨率历史关键帧，限长。

Recent actions:
  最近 k 步动作。

Current compact plan:
  持久 plan 的滚动窗口。

Active instruction excerpt:
  可选，来自原文的当前相关片段，不是 summary。

Allowed actions:
  当前 benchmark config 中允许动作。
```

---

## 4. Initial System Prompt

episode 开始前给 VLM 一个固定的 system prompt。  
它定义角色、格式、工具、plan 规则和 recovery policy。  
它不包含具体当前 observation，也不替代每轮 full instruction。

训练时 system prompt 是 input context，mask=0。

推荐 system prompt：

```text
You are a continuous Vision-and-Language Navigation planner.

Your task is to navigate in a continuous environment according to the given natural language instruction.

You must maintain a persistent structured plan state.

The plan is not a reasoning trace.
The plan is an executable control state maintained by an external controller.

Output rules:
1. Output only XML.
2. Do not explain your reasoning.
3. Do not output free-form analysis.
4. Output a new <plan> only when initializing or replanning.
5. During normal execution, do not repeat the full <plan>.

Available tools:
- continue: follow the current plan.
- replan: update the current and future plan when the current plan is no longer reliable.
- stop: stop only when the instruction goal is reached.

Plan rules:
1. A plan contains <global> and <local>.
2. <global> is a compact route skeleton inside the plan.
3. <global> does not replace the full instruction.
4. <local> is a rolling execution window.
5. Each local plan point has a status:
   - done
   - current
   - todo
   - abandoned
6. Done points are immutable.
7. Replan may only modify the current and future plan.
8. A new plan must contain exactly one current point.

Recovery policy:
If the current observation and recent history no longer support the current plan,
do not blindly continue the old plan.
Use <tool>replan</tool> to create a recovery subgoal.
A recovery subgoal may first return to a previously observed reliable route place,
such as leaving a wrong room and returning to the hallway,
then resume the original instruction.

Action rules:
1. The <action> field must contain exactly one executable action.
2. The action must be selected from the Allowed actions listed in the current step prompt.
3. Do not output actions that are not listed.
4. Do not output multiple actions.
5. When using <tool>stop</tool>, the action must be STOP.

Output schema:

When initializing or replanning:
<plan>
  <global>...</global>
  <local>
    <p id="..." status="done/current/todo/abandoned">...</p>
  </local>
</plan>
<tool>continue / replan / stop</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>

During normal execution:
<tool>continue</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>

When stopping:
<tool>stop</tool>
<subgoal>stop at the target location</subgoal>
<action>STOP</action>
```

注意：

```text
具体 allowed actions 不建议硬编码在 system prompt 中。
system prompt 只定义 action 规则。
每一轮 step prompt 明确列出当前环境允许的具体动作。
```

---

## 5. 每轮 Step Prompt

每轮 step prompt 都包含完整 instruction。

模板：

```text
Full instruction:
...

Allowed actions:
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP

Current observation:
[HIGH-RES current image]
...

Recent visual history:
[LOW-RES keyframe t-k] ...
...

Recent actions:
...

Current compact plan:
<plan>...</plan>
or
None. Please initialize the plan.

Active instruction excerpt:
"..."
```

其中：

```text
Full instruction:
  原始完整 instruction，每轮重复输入，不压缩。

Allowed actions:
  从 Habitat / benchmark config 读取，不能硬编码在模型中。

Current observation:
  当前高分辨率图像或 panorama。

Recent visual history:
  最近 k 个低分辨率关键帧或 caption，控制视觉 token 和 KV cache 成本。

Recent actions:
  最近动作序列，用于帮助模型理解自己如何到达当前状态。

Current compact plan:
  持久 plan 的滚动窗口，不是完整历史。

Active instruction excerpt:
  可选，必须是原文片段或 pointer，不是模型自由改写的 summary。
```

---

## 6. 视觉上下文预算

采用：

```text
current observation: high resolution
recent history frames: low resolution
older history: dropped or converted to captions
```

原因：

```text
当前观测决定下一步动作，需要高分辨率；
历史帧主要提供路径上下文，不需要高分辨率；
高分辨率历史帧会快速消耗视觉 token 和 KV cache。
```

建议实现：

```text
current frame: original / high-res
recent history: low-res keyframes, k=3~5 for first version
older history: captions or discarded
```

---

## 7. Plan State 结构

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> dining area -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="current">continue along the hallway toward the stairs</p>
    <p id="p3" status="todo">turn right after the stairs toward the dining area</p>
  </local>
</plan>
```

含义：

```text
<global>:
  plan 内部的长期路线骨架。
  它不替代 full instruction。

<local>:
  当前滚动执行窗口。
  不需要保存完整路线。

<p>:
  局部 plan point。
```

状态：

| status | 含义 |
|---|---|
| done | 已完成，不可修改 |
| current | 当前正在执行 |
| todo | 未来计划点 |
| abandoned | 当前计划失败或不可靠，已放弃 |

关键规则：

```text
done points are immutable.
replan only modifies current and future plan.
```

---

## 8. Global 与 Full Instruction 的关系

`<global>` 保留，但不能替代完整 instruction。

分工：

```text
Full instruction:
  ground-truth task source，每轮完整输入。

<global>:
  plan 内部的路线骨架，帮助 replan 时防止目标漂移。

<local>:
  当前 rolling execution window。

<subgoal>:
  当前短时间窗口的可执行目标。
```

---

## 9. 输出接口

最终测试时模型输出：

```xml
<tool>continue / replan / stop</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>
```

只有初始化或 replan 时额外输出：

```xml
<plan>...</plan>
```

完整接口：

```xml
<plan>...</plan>   <!-- only when initializing or replanning -->
<tool>continue / replan / stop</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>
```

不使用：

```xml
<op>
<target>
<now>
<backtrack>
<replan_type>
```

---

## 10. Continue

普通执行时，不输出 `<plan>`。

```xml
<tool>continue</tool>
<subgoal>continue straight down the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

controller 维持当前 plan 不变。

---

## 11. Replan

`replan` 是一次性计划更新，不是每一步都重新规划。

状态机：

```text
replan once -> update persistent plan -> continue new plan
```

replan 时必须输出新 `<plan>`：

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> dining area -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="abandoned">continue along the hallway toward the stairs</p>
    <p id="r1" status="current">turn around, leave the side room, and return to the hallway</p>
    <p id="p3" status="todo">continue along the hallway toward the stairs</p>
    <p id="p4" status="todo">turn right after the stairs toward the dining area</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room through the doorway, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

---

## 12. Direct Replan 与 Recovery Replan

测试 XML 中只有：

```xml
<tool>replan</tool>
```

训练内部区分：

```text
direct_replan
recovery_replan
```

direct replan：从当前位置直接找新路线。  
recovery replan：先生成恢复子目标，返回历史中可靠路线位置，再继续原始 instruction。

二者不通过 `<replan_type>` 显式输出。  
区别由 `<plan>` 和 `<subgoal>` 自然体现。

训练内部变量：

```python
branch_type = {
  "continue",
  "direct_replan",
  "recovery_replan",
  "stop"
}
```

`branch_type` 只用于训练分支构造和统计，不进入测试 prompt，不进入最终 XML。

---

## 13. 训练 Mask

每轮训练样本中：

```text
System prompt                 mask = 0
Full instruction              mask = 0
Current observation           mask = 0
Recent visual history         mask = 0
Recent actions                mask = 0
Current compact plan          mask = 0
Active instruction excerpt    mask = 0
Allowed actions               mask = 0
Forced branch prompt          mask = 0

Assistant XML output          mask = 1
```

RL 时计算：

```text
log π(output_xml | full_instruction, obs, history, actions, plan)
```

不让模型学习复述输入 context。

---

## 14. Counterfactual Plan-Tool RL

这是 CFRP-VLN 的主创新。

在 critical state 上，从同一个环境状态展开：

```text
continue
direct_replan
recovery_replan
stop
```

每个分支分别 rollout。

例子：

```text
continue        -> loop
direct_replan   -> long detour
recovery_replan -> return to hallway and succeed
stop            -> wrong stop
```

排序：

```text
recovery_replan > direct_replan > continue > stop
```

模型学习：

```text
什么时候 continue 会失败；
什么时候 direct replan 足够；
什么时候必须 recovery replan；
什么时候不能 stop。
```

---

## 15. Forced Branch Prompt 与 Prompt Leakage

训练时可以用 forced branch prompt 生成候选：

```text
For this rollout branch, use <tool>replan</tool>.
Create a recovery subgoal first, then resume the original instruction.
```

但它只用于候选生成。  
最终 policy update 必须在 normal prompt 下计算。

```python
x = normal_prompt(state)

y_continue = generate(x + forced_continue_prompt)
y_direct   = generate(x + forced_direct_replan_prompt)
y_recovery = generate(x + forced_recovery_replan_prompt)
y_stop     = generate(x + forced_stop_prompt)

ranked_outputs = rank_by_environment_outcome(
    y_continue, y_direct, y_recovery, y_stop
)

loss = preference_loss(
    prompt=x,
    ranked_outputs=ranked_outputs
)
```

核心原则：

```text
forced prompt 用于 counterfactual exploration；
normal prompt 用于 policy learning。
```

---

## 16. Reward

训练 reward：

```text
R =
  + Success
  + SPL / path efficiency
  + GoalProgress
  - WrongStop
  - Collision / Stuck
  - Loop
  - PathLength
  - InvalidFormatOrAction
  - TooFrequentReplan
```

工具约束：

```text
continue:
  不应输出 <plan>

replan:
  必须输出 <plan>
  必须保留 done points
  新 plan 必须有且只有一个 current point

stop:
  action 必须是 STOP
```

---

## 17. 训练建议

Qwen3-VL-4B 可以尝试，但建议分阶段：

```text
Stage 1: XML format / action validity warmup
Stage 2: normal VLN action tuning
Stage 3: short recovery scenarios
Stage 4: counterfactual plan-tool RL
```

第一版建议限制：

```text
history frames: 3~5
local plan window: 3~5 points
action: single primitive action
tools: continue / replan / stop
replan frequency penalty: enabled
```

---

## 18. 贡献点

1. **Sparse Persistent Plan State**  
   维护稀疏持久化 plan，只在初始化和 replan 时更新，不是每步 CoT。

2. **Recovery-oriented Replanning**  
   模型基于完整 instruction、当前观测、低分辨率历史、最近动作和 compact plan 生成具体恢复子目标。

3. **Counterfactual Plan-Tool RL**  
   在同一个 critical state 下比较 continue、direct replan、recovery replan、stop 的长期结果，学习何时应该恢复。

---

## 19. 最终 Claim

英文：

> CFRP-VLN is a counterfactual recovery planning framework for continuous VLN. It repeats the full instruction at every interaction step, maintains a sparse persistent executable plan, and learns via counterfactual plan-tool rollout when to continue, directly replan, perform recovery-oriented replanning, or stop.

中文：

> CFRP-VLN 是一个面向连续 VLN 的反事实恢复规划框架。它在每轮交互中保留完整原始指令，维护稀疏持久化可执行计划，并通过反事实 plan-tool rollout 学习何时继续、直接重规划、恢复式重规划或停止。
