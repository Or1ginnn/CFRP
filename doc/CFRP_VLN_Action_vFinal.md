# CFRP-VLN Action.md

## 1. 方法名称

**CFRP-VLN**  
**Counterfactual Recovery Planning for Continuous Vision-and-Language Navigation**

该文档定义 CFRP-VLN 的 VLM 交互、tool space、action interface、controller 逻辑、反事实分支和训练约束。

---

## 2. 每轮 VLM 交互

CFRP-VLN 是 multi-turn VLM agent loop。

每一轮：

```text
build prompt
-> VLM outputs XML
-> parse XML
-> validate tool/action/plan
-> execute action
-> update plan/history
```

每轮 prompt 都包含完整 instruction：

```text
Full instruction:
  每轮完整输入，不压缩。

Current observation:
  当前高分辨率图像。

Recent visual history:
  最近 k 个低分辨率历史关键帧。

Recent actions:
  最近动作历史。

Current compact plan:
  持久 plan 的 rolling window。

Allowed actions:
  当前环境允许动作。
```

训练时，以上全部是 input context，mask=0。  
只有 VLM 的 XML output 算 loss / logprob。

---

## 3. System Prompt 与 Step Prompt

### 3.1 System Prompt

system prompt 负责定义规则：

```text
role
output schema
tool definitions
plan rules
recovery policy
action rules
```

system prompt 不负责保存完整 instruction。  
完整 instruction 每轮出现在 step prompt 中。

system prompt 中 action 规则：

```text
Action rules:
1. The <action> field must contain exactly one executable action.
2. The action must be selected from the Allowed actions listed in the current step prompt.
3. Do not output actions that are not listed.
4. Do not output multiple actions.
5. When using <tool>stop</tool>, the action must be STOP.
```

### 3.2 Step Prompt

每一轮 step prompt 包含具体状态：

```text
Full instruction:
...

Allowed actions:
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP

Current observation:
[HIGH-RES current image]
...

Recent visual history:
[LOW-RES keyframes]

Recent actions:
...

Current compact plan:
<plan>...</plan>

Active instruction excerpt:
"..."
```

注意：

```text
具体 allowed actions 放在每轮 step prompt 中；
system prompt 只写 action 必须来自 allowed actions。
```

---

## 4. 输出接口

模型每轮输出：

```xml
<tool>continue / replan / stop</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>
```

初始化或 replan 时，额外输出：

```xml
<plan>...</plan>
```

完整格式：

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

## 5. Tool Space

CFRP-VLN 的 tool space：

```text
continue
replan
stop
```

### continue

继续执行当前 plan point。

```xml
<tool>continue</tool>
<subgoal>continue straight down the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

约束：

```text
continue 不输出 <plan>；
默认执行 current plan point；
action 必须合法。
```

### replan

一次性更新当前和未来 plan。

```xml
<plan>...</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

约束：

```text
replan 必须输出 <plan>；
done points 必须保留；
只能修改 current/future；
新 plan 必须有且只有一个 current point；
旧 current 可标记 abandoned；
replan 后应 continue 新 plan。
```

### stop

停止在目标位置。

```xml
<tool>stop</tool>
<subgoal>stop beside the sink near the window</subgoal>
<action>STOP</action>
```

约束：

```text
stop 时 action 必须是 STOP；
wrong stop 强惩罚。
```

---

## 6. Subgoal

`<subgoal>` 是当前短时间窗口的局部可执行目标。  
它不是 CoT，也不是解释。

好的 subgoal：

```xml
<subgoal>turn around, leave the side room through the doorway, and return to the hallway</subgoal>
```

不好的 subgoal：

```text
I think I went wrong because the hallway should be somewhere else.
```

原因：这是解释，不是可执行局部目标。

---

## 7. Action Space

`<action>` 必须来自每轮 step prompt 中的 `Allowed actions`。

常见 Habitat/VLN-CE primitive actions：

```text
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

某些 config 可能包含：

```text
LOOK_UP
LOOK_DOWN
```

因此：

```text
不要在 system prompt 中固定具体动作集合；
每轮从 benchmark config 注入 Allowed actions。
```

非法 action：

```text
GO_LEFT
WALK
MOVE_FORWARD, TURN_LEFT
```

都应判 invalid。

---

## 8. 每轮输入输出例子

### Input

```text
Full instruction:
Exit the bedroom, turn left into the hallway, walk past the painting,
continue until the stairs, turn right, pass the dining area,
enter the kitchen, go to the counter near the window, and stop beside the sink.

Allowed actions:
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP

Current observation:
[HIGH-RES current image]
You are inside a small side room. A desk is in front of you.
The doorway back to the hallway is behind-left.

Recent visual history:
[LOW-RES t0] inside bedroom facing the open doorway
[LOW-RES t1] entered the hallway
[LOW-RES t2] hallway with painting and a side doorway on the right

Recent actions:
t0: MOVE_FORWARD
t1: MOVE_FORWARD
t2: TURN_RIGHT
t3: MOVE_FORWARD

Current compact plan:
<plan>
  <global>bedroom -> hallway -> stairs -> dining area -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="current">turn left into the hallway and walk past the painting</p>
    <p id="p3" status="todo">continue toward the stairs</p>
    <p id="p4" status="todo">turn right after the stairs toward the dining area</p>
  </local>
</plan>

Active instruction excerpt:
"walk past the painting, continue until the stairs, turn right, pass the dining area"
```

### Output

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> dining area -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="abandoned">turn left into the hallway and walk past the painting</p>
    <p id="r1" status="current">turn around, leave the side room, and return to the hallway</p>
    <p id="p3" status="todo">continue along the hallway past the painting toward the stairs</p>
    <p id="p4" status="todo">turn right after the stairs toward the dining area</p>
    <p id="p5" status="todo">enter the kitchen and stop beside the sink</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room through the doorway, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

---

## 9. Controller 状态机

controller 很薄，只负责状态管理。

```python
def step(output, current_plan, allowed_actions):
    parsed = parse_xml(output)

    if parsed.tool == "replan":
        assert parsed.plan is not None
        validate_plan(parsed.plan)
        current_plan = parsed.plan

    elif parsed.tool == "continue":
        assert parsed.plan is None

    elif parsed.tool == "stop":
        assert parsed.action == "STOP"

    validate_action(parsed.action, allowed_actions)
    obs_next, reward, done, info = env.step(parsed.action)

    update_history(obs_next, parsed.action, parsed.tool, parsed.subgoal)
    return current_plan, obs_next, reward, done, info
```

controller 不负责：

```text
semantic anchor manager
backtrack controller
traditional planner
free-form reasoning
```

---

## 10. Plan Progress Update

controller 可以根据环境进展或简单规则更新 plan point 状态：

```text
如果 current plan point 完成：
  current -> done
  next todo -> current
```

---

## 11. Replan Frequency

为了避免模型每一步 replan，加入频率约束：

```text
- small penalty per replan
- stronger penalty if replan occurs again within last k steps
```

目的：

```text
必要时 replan；
replan 后坚持执行新 plan；
再次失败时才重新 replan。
```

---

## 12. Training-only Branch Type

测试时 XML 没有 `<replan_type>`。

训练内部使用：

```python
branch_type = {
  "continue",
  "direct_replan",
  "recovery_replan",
  "stop"
}
```

用途：

```text
构造反事实分支；
记录 rollout metadata；
计算 branch ranking；
做 ablation 和统计。
```

不用于：

```text
测试 prompt；
最终 XML；
normal policy input。
```

---

## 13. Counterfactual Group Rollout

在 critical state `s_t`：

```text
G(s_t) = {
  continue,
  direct_replan,
  recovery_replan,
  stop
}
```

每个分支从同一个环境状态 rollout。

分支结果例子：

```text
continue        -> loop
direct_replan   -> long detour
recovery_replan -> success
stop            -> wrong stop
```

排序：

```text
recovery_replan > direct_replan > continue > stop
```

---

## 14. Forced Branch Generation

候选生成时可以临时添加 forced branch prompt：

```text
For this rollout branch, use <tool>continue</tool>.
Follow the current plan without changing the plan.
```

或：

```text
For this rollout branch, use <tool>replan</tool>.
Create a recovery subgoal first, then resume the original instruction.
```

这些 forced branch prompts：

```text
只用于生成候选；
mask=0；
不进入最终 policy update 的 normal prompt。
```

---

## 15. No Prompt Leakage

最终优化时使用 normal prompt：

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

核心：

```text
forced branch prompt 用于 counterfactual exploration；
normal prompt 用于 policy learning。
```

---

## 16. Training Mask

每轮：

```text
system prompt                 mask = 0
full instruction              mask = 0
current observation           mask = 0
recent visual history         mask = 0
recent actions                mask = 0
current compact plan          mask = 0
active instruction excerpt    mask = 0
allowed actions               mask = 0
forced branch prompt          mask = 0

assistant XML output          mask = 1
```

RL 时：

```text
log π(output_xml | full_instruction, obs, history, actions, plan)
```

---

## 17. Branch Scoring

每个分支 rollout 后打分：

```text
Score(branch) =
  + success
  + SPL / path efficiency
  + goal distance progress
  - wrong stop
  - collision / stuck
  - loop
  - path length
  - invalid format/action
  - too frequent replan
```

goal distance progress 只在训练时使用 simulator 信息，测试时不使用。

---

## 18. Reward

总 reward：

```text
R =
  w_succ * Success
+ w_spl  * SPL
+ w_prog * GoalProgress
- w_stop * WrongStop
- w_col  * CollisionOrStuck
- w_loop * Loop
- w_len  * PathLength
- w_fmt  * InvalidFormatOrAction
- w_rep  * TooFrequentReplan
```

---

## 19. Critical State Mining

无需人工偏航标签。  
从 rollout 自动挖 critical states：

```text
goal progress 变慢或变负
loop
repeated states
collision/stuck
wrong stop tendency
current observation 与 plan mismatch
tool ambiguity
```

这些状态用于构造 counterfactual branches。

---

## 20. Testing Pipeline

测试时不做 group rollout，也没有 forced branch prompt。

流程：

```text
normal step prompt
-> VLM output XML
-> controller parse/validate
-> env.step(action)
-> update plan/history
```

测试时模型自己选择：

```text
continue
replan
stop
```

direct/recovery 的区别由 `<plan>` 和 `<subgoal>` 自然体现。

---

## 21. Evaluation

标准 VLN 指标：

```text
SR
SPL
nDTW
SDTW
Path Length
```

恢复相关指标：

```text
Wrong Stop Rate
Loop Rate
Collision/Stuck Rate
Replan Frequency
Recovery Replan Success Rate
Post-Replan Success Rate
Average Steps After Replan
```

核心诊断：

```text
critical states 上 recovery_replan 是否显著优于 continue。
```

---

## 22. Ablation

推荐 ablation：

```text
No plan
Plan every step
No replan
No recovery replan
No recent history
No recent actions
No full instruction each step
Compressed instruction instead of full instruction
No counterfactual group rollout
Standard RL without counterfactual branches
No hard format/action constraints
No replan frequency penalty
```

最关键 ablation：

```text
No counterfactual group rollout
No full instruction each step
```

---

## 23. 最终方法表述

英文：

> CFRP-VLN learns executable plan-tool actions by comparing counterfactual branches from the same navigation state. At each interaction step, the VLM receives the full instruction, high-resolution current observation, low-resolution history keyframes, recent actions, and a compact persistent plan, then outputs XML containing a tool, subgoal, action, and optionally an updated plan.

中文：

> CFRP-VLN 在每轮交互中输入完整 instruction、当前高分辨率观测、低分辨率历史关键帧、最近动作和 compact persistent plan，并输出 tool、subgoal、action 以及可选更新 plan。训练时通过同一导航状态下的反事实分支比较学习可恢复导航策略。
