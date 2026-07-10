# CFRP-VLN 完整中文版 Action Plan.md

## 1. 文件目的

本文档是 CFRP-VLN 的 **Action Plan / 实施计划**。

它对应 `Plan.md` 中的方法设计，重点回答：

> 这个 idea 具体怎么实现、怎么训练、怎么跑实验、每个模块需要做什么、输入输出是什么、有哪些工程约束。

本文档不重新论证方法动机，而是作为后续 coding、实验推进、论文 method 落地的执行清单。

---

## 2. 最终方法固定版本

最终版本保持：

```text
tool space = { continue, replan }
```

不引入第三个 tool。

### continue

含义：

```text
当前 plan 仍然可靠。
继续执行当前 plan / subgoal。
不更新 plan。
```

### replan

含义：

```text
当前 plan 不再可靠。
保留 done points。
放弃不可靠 current point。
更新 current / future plan。
生成 recovery-oriented subgoal。
```

### STOP

`STOP` 不是 tool。

`STOP` 是环境 action，只能出现在：

```xml
<action>STOP</action>
```

---

## 3. 总体实现路线

推荐分四个阶段实现：

```text
Stage 0: 环境与数据准备
Stage 1: XML planner + controller 基础闭环
Stage 2: persistent plan state + continue/replan 推理闭环
Stage 3: expert-guided counterfactual branching
Stage 4: preference RL / GRPO-style optimization
Stage 5: 实验、ablation、论文结果整理
```

每个阶段都应该有可验证输出，避免直接上复杂 RL。

---

# Stage 0: 环境与数据准备

## 4. 环境选择

优先使用 Habitat-based continuous VLN。

推荐 benchmark：

```text
R2R-CE
RxR-CE
VLN-CE
```

最小可行实验可以先选：

```text
R2R-CE
```

原因：

```text
路线相对短
instruction 相对清晰
expert trajectory 可用
工程复杂度低于 RxR-CE
```

---

## 5. 需要的数据

每个 episode 至少需要：

```text
instruction
scene id
start pose
goal / target information
expert trajectory
allowed actions
success condition
```

其中 expert trajectory 是训练时关键资源，用于：

```text
1. deviation detection
2. recovery branch evaluation
3. expert-route alignment reward
```

测试时不能把 expert trajectory 输入模型。

---

## 6. 环境封装接口

需要实现一个统一环境 wrapper。

建议接口：

```python
class VLNEnvWrapper:
    def reset(self, episode_id):
        ...

    def step(self, action):
        ...

    def get_observation(self):
        ...

    def get_agent_state(self):
        ...

    def set_agent_state(self, state):
        ...

    def get_allowed_actions(self):
        ...

    def is_success(self):
        ...

    def is_failure(self):
        ...

    def get_metrics(self):
        ...
```

重点是必须支持：

```text
save current state
restore current state
```

因为 counterfactual branching 需要从同一个 critical state 展开 continue 和 replan 两个 branch。

---

## 7. Checkpoint 内容

触发 critical state 时保存：

```text
Habitat agent state
current compact plan
recent visual history
recent actions
controller internal memory
cooldown status
step index
```

不保存：

```text
完整 400-step trajectory
每一步高分辨率图像
所有历史 prompt/output
```

最小 checkpoint：

```python
checkpoint = {
    "agent_state": agent_state,
    "plan_state": plan_state,
    "recent_history": recent_history,
    "recent_actions": recent_actions,
    "cooldown": cooldown_state,
    "step_id": t,
}
```

---

# Stage 1: XML Planner + Controller 基础闭环

## 8. XML 输出格式

VLM 每一步输出 XML。

### continue 输出

```xml
<tool>continue</tool>
<subgoal>move forward along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
```

### replan 输出

完整 plan 版本：

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

轻量 patch 版本：

```xml
<tool>replan</tool>
<plan_update>
  <abandon>p2</abandon>
  <current>return to hallway entrance</current>
  <future>continue toward stairs</future>
</plan_update>
<subgoal>turn around, leave the side room, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
```

论文层面可以展示完整 plan；工程实现可优先用 patch。

---

## 9. XML Parser

需要实现稳健 parser。

输入：

```text
assistant raw output
```

输出：

```python
ParsedOutput(
    tool="continue" or "replan",
    subgoal=str,
    action=str,
    plan=None or PlanState,
    plan_update=None or PlanPatch,
    valid=True/False,
    error_type=None
)
```

Parser 要检查：

```text
1. 是否包含 <tool>
2. tool 是否属于 {continue, replan}
3. 是否包含 <action>
4. action 是否在 allowed actions 中
5. 是否包含 <subgoal>
6. replan 时是否包含 plan 或 plan_update
7. continue 时是否错误输出了完整 plan
8. XML 是否可解析
9. 是否输出多个 action
```

无效输出处理：

```text
invalid_xml
invalid_tool
invalid_action
missing_action
missing_subgoal
invalid_plan_update
```

---

## 10. Action Validation

Controller 执行前必须验证 action。

规则：

```python
if parsed.action not in allowed_actions:
    mark_invalid_action()
    action = fallback_action
```

fallback 策略可以先简单设置：

```text
TURN_LEFT
```

或者：

```text
previous valid action
```

但训练打分时必须惩罚 invalid action。

建议：

```text
invalid action penalty = strong negative
```

---

## 11. Prompt Builder

实现 `build_normal_step_prompt()`。

输入：

```python
instruction
current_observation
recent_visual_history
recent_actions
current_plan
allowed_actions
```

输出：

```text
normal step prompt
```

模板：

```text
Full instruction:
{instruction}

Current observation:
{current_high_res_rgb_or_caption}

Recent visual history:
{recent_low_res_frames_or_captions}

Recent actions:
{last_4_to_8_actions}

Current compact plan:
{current_plan_or_none}

Allowed actions:
{allowed_actions}

Reminder:
Output XML only.
Choose either <tool>continue</tool> or <tool>replan</tool>.
Use <tool>replan</tool> only when the current plan is no longer reliable.
```

---

## 12. System Prompt

System prompt 必须固定。

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
```

---

# Stage 2: Persistent Plan State

## 13. PlanState 数据结构

建议实现：

```python
@dataclass
class PlanPoint:
    id: str
    status: Literal["done", "current", "todo", "abandoned"]
    text: str

@dataclass
class PlanState:
    global_route: str
    local_points: list[PlanPoint]
```

约束：

```python
def validate_plan(plan):
    assert count_status(plan, "current") == 1
    assert all(point.status in {"done", "current", "todo", "abandoned"}
               for point in plan.local_points)
```

---

## 14. 初始化 Plan

episode 开始时可以有两种方式：

### 14.1 VLM 初始化

给 VLM instruction，让它生成初始 compact plan。

```xml
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen</global>
  <local>
    <p id="p1" status="current">exit bedroom</p>
    <p id="p2" status="todo">turn left into hallway</p>
    <p id="p3" status="todo">reach stairs</p>
  </local>
</plan>
```

### 14.2 规则初始化

不显式生成 global，只用：

```text
current: start following the instruction
todo: continue the remaining instruction
```

最小实现建议先用规则初始化，后续再引入 VLM plan initialization。

---

## 15. Plan Update

当 tool 是 `continue`：

```text
不更新 plan
只更新 recent history / recent actions
```

当 tool 是 `replan`：

```text
1. 保留 done points
2. 将旧 current 标为 abandoned
3. 插入新的 current recovery subgoal
4. 保留或更新 future todo points
5. 校验 exactly one current
```

伪代码：

```python
def apply_replan(plan_state, plan_update):
    for p in plan_state.local_points:
        if p.status == "done":
            continue
        if p.status == "current":
            p.status = "abandoned"

    recovery_point = PlanPoint(
        id=new_recovery_id(),
        status="current",
        text=plan_update.current
    )

    new_todos = build_future_points(plan_update.future)
    return PlanState(
        global_route=plan_state.global_route,
        local_points=done_points + abandoned_points + [recovery_point] + new_todos
    )
```

---

## 16. Plan Progress Update

需要一个机制把 current 变成 done。

简单版本：

```text
VLM 显式 replan 或 controller 根据动作/观察判断 progress。
```

推荐先用轻量 heuristic：

```text
如果连续 K 步 subgoal 变化，或者 VLM 输出新 subgoal 接近 todo point，则更新 current。
```

更稳妥的最小实现：

```text
只在 replan 时改变 plan structure。
done/current/todo 作为模型输入记忆，不做过强自动推进。
```

论文中强调 persistent plan state；工程上先做保守更新。

---

# Stage 3: Inference Loop

## 17. Inference 主循环

```python
env.reset(episode_id)
plan_state = initialize_plan(instruction)
recent_history = []
recent_actions = []
cooldown = 0

for t in range(max_steps):
    obs = env.get_observation()
    allowed_actions = env.get_allowed_actions()

    prompt = build_normal_step_prompt(
        instruction=instruction,
        current_observation=obs,
        recent_visual_history=recent_history,
        recent_actions=recent_actions,
        current_plan=plan_state,
        allowed_actions=allowed_actions,
    )

    raw = vlm.generate(system_prompt, prompt)
    parsed = parse_xml(raw, allowed_actions)

    if not parsed.valid:
        action = fallback_action
        log_invalid_output(parsed)
    else:
        if parsed.tool == "replan" and cooldown == 0:
            plan_state = apply_replan(plan_state, parsed.plan_or_update)
            cooldown = REPLAN_COOLDOWN
        elif parsed.tool == "replan" and cooldown > 0:
            # inference 时可以选择强制转 continue 或惩罚
            handle_replan_during_cooldown()
        action = parsed.action

    obs_next, reward, done, info = env.step(action)

    update_recent_history(recent_history, obs)
    update_recent_actions(recent_actions, action)

    cooldown = max(0, cooldown - 1)

    if done:
        break
```

---

## 18. Replan Cooldown

推荐参数：

```text
REPLAN_COOLDOWN = 5 到 20 steps
```

最初可以设置：

```text
REPLAN_COOLDOWN = 10
```

策略：

```text
if tool == replan and cooldown > 0:
    either reject replan
    or allow action but do not update plan
    or add penalty during training
```

训练时建议：

```text
too frequent replan penalty
```

推理时建议：

```text
cooldown 内禁止连续 plan update
```

---

# Stage 4: Expert-guided Deviation Detection

## 19. Expert Path 表示

Expert trajectory 表示为 pose 序列：

```python
expert_path = [
    pose_0,
    pose_1,
    pose_2,
    ...
]
```

每个 pose 包含：

```text
position
rotation / heading
optional timestamp
optional navmesh node
```

---

## 20. Distance to Expert Path

计算 agent 当前 pose 到 expert path 最近点距离：

```python
def distance_to_expert_path(agent_pose, expert_path):
    return min(distance(agent_pose.position, p.position) for p in expert_path)
```

触发：

```python
if distance_to_expert_path > tau_d:
    trigger = True
```

初始建议：

```text
tau_d = 2.0m 到 3.0m
```

具体需要按 Habitat 场景调参。

---

## 21. Progress Gap

定义 expert progress：

```text
agent 当前 pose 在 expert_path 上的最近点 index
```

如果最近点 index 长时间不增长，说明 stuck 或偏离。

```python
nearest_idx = argmin_distance(agent_pose, expert_path)
progress = nearest_idx / len(expert_path)
```

可以维护：

```text
best_progress_so_far
```

触发条件：

```python
if progress < best_progress_so_far - margin:
    trigger = True

if progress does not improve for K steps:
    trigger = True
```

---

## 22. Combined Trigger

建议最终 trigger：

```python
D_t = distance_to_expert_path(agent_pose, expert_path)
P_t = expert_progress(agent_pose, expert_path)

trigger = (
    D_t > tau_d
    or progress_stagnant_for_K_steps
    or goal_distance_increasing_for_K_steps
)
```

但第一版实现可以先用：

```text
distance_to_expert_path > tau_d
```

后续再加 progress gap。

---

## 23. Trigger 限制

训练时：

```text
只在 NORMAL mode 触发
branch rollout 不触发
每个 normal rollout 最多触发一次
cooldown 中不触发
如果模型刚刚自己 replan，不触发
```

伪代码：

```python
if mode == "NORMAL":
    if not branch_used:
        if cooldown == 0:
            if last_tool != "replan":
                if deviation > threshold:
                    trigger_counterfactual()
```

---

# Stage 5: Counterfactual Branching

## 24. Branch 入口

当触发 critical state：

```python
checkpoint = save_checkpoint(
    env_state,
    plan_state,
    recent_history,
    recent_actions,
    cooldown_state
)
```

然后执行：

```python
continue_output, continue_traj = run_branch(
    checkpoint=checkpoint,
    forced_tool="continue",
    horizon=H
)

replan_output, replan_traj = run_branch(
    checkpoint=checkpoint,
    forced_tool="replan",
    horizon=H
)
```

---

## 25. Forced Continue Prompt

在 normal prompt 后追加：

```text
For this counterfactual branch, force the first tool decision to be:
<tool>continue</tool>

Do not update the plan in this first step.
Follow the current plan and output one valid action.
```

第一步必须是：

```xml
<tool>continue</tool>
```

如果模型没有输出 continue，强制判 invalid 或重新采样。

---

## 26. Forced Replan Prompt

在 normal prompt 后追加：

```text
For this counterfactual branch, force the first tool decision to be:
<tool>replan</tool>

Update the current and future plan.
Preserve all done plan points.
Create a recovery subgoal first, then resume the original instruction.
Output one valid action.
```

第一步必须是：

```xml
<tool>replan</tool>
```

如果模型没有输出 replan，强制判 invalid 或重新采样。

---

## 27. Branch Rollout 函数

```python
def run_branch(checkpoint, forced_tool, horizon):
    restore_checkpoint(checkpoint)
    mode = "BRANCH_ROLLOUT"
    trajectory = []

    for k in range(horizon):
        prompt = build_normal_step_prompt(...)

        if k == 0:
            prompt = prompt + build_forced_tool_prompt(forced_tool)

        raw = vlm.generate(system_prompt, prompt)
        parsed = parse_xml(raw, allowed_actions)

        if k == 0 and parsed.tool != forced_tool:
            mark_invalid_branch()
            break

        if parsed.tool == "replan":
            apply_replan_if_valid(parsed)

        action = parsed.action if parsed.valid else fallback_action
        obs, reward, done, info = env.step(action)

        trajectory.append({
            "obs": obs,
            "action": action,
            "tool": parsed.tool,
            "subgoal": parsed.subgoal,
            "valid": parsed.valid,
            "agent_state": env.get_agent_state(),
            "info": info,
        })

        if early_stop_condition(trajectory, info):
            break

    return first_output, trajectory
```

重要：

```text
BRANCH_ROLLOUT mode 下禁止再次 trigger counterfactual。
```

---

## 28. Branch Horizon

推荐：

```text
H = 30 到 50 steps
```

初始实验：

```text
H = 30
```

如果 recovery 经常需要更长距离，再调到：

```text
H = 50
```

不要一开始就跑完整 episode。

---

## 29. Early Stop Conditions

Branch 提前停止：

```text
success
wrong stop
loop
stuck
no progress
too many collisions
invalid output repeatedly
horizon reached
```

### loop

可以检查最近位置重复：

```python
if agent revisits same region many times:
    stop_branch("loop")
```

### stuck

```python
if movement distance over last K steps < epsilon:
    stop_branch("stuck")
```

### no progress

```python
if distance_to_expert_path not decreasing and goal progress not improving for K steps:
    stop_branch("no_progress")
```

### collision

```python
if collision_count > threshold:
    stop_branch("too_many_collisions")
```

---

# Stage 6: Branch Evaluation

## 30. Branch Score

每个 branch 得到一个分数：

```python
score = (
    w_success * success
    + w_goal * goal_progress
    + w_align * expert_route_alignment
    + w_recovery * recovery_progress
    - w_collision * collisions
    - w_loop * loop
    - w_stuck * stuck
    - w_path * path_length_penalty
    - w_invalid * invalid_output_penalty
)
```

---

## 31. 推荐评分项

### success

```text
成功到达目标并 STOP。
```

### goal progress

```text
branch 结束时比开始时更接近目标。
```

### expert route alignment

```text
branch 结束时距离 expert path 更近。
```

### recovery progress

```text
是否从错误状态回到可靠路线附近。
```

例如：

```python
recovery_progress = D_start - D_end
```

其中：

```text
D_start = critical state 到 expert path 距离
D_end = branch end state 到 expert path 距离
```

### collision penalty

```text
collision 越多，分数越低。
```

### loop / stuck penalty

```text
loop 或 stuck 直接强负分。
```

---

## 32. Score Margin

只有当分数差异明显时才产生 preference pair。

```python
if abs(score_replan - score_continue) > margin:
    keep_pair = True
else:
    discard_pair = True
```

推荐初始：

```text
margin = 0.2
```

后续调参。

---

## 33. Preference Label

如果：

```text
score_replan > score_continue + margin
```

则：

```text
winner = replan first output
loser = continue first output
```

如果：

```text
score_continue > score_replan + margin
```

则：

```text
winner = continue first output
loser = replan first output
```

注意：

虽然我们希望 replan 能恢复，但不是所有偏离状态都一定应该 replan。

有些轻微偏离下 continue 可能更好。

这点能让模型学到更真实的 decision boundary。

---

# Stage 7: Preference Data Construction

## 34. 数据格式

每个 preference sample：

```python
sample = {
    "normal_prompt": normal_prompt_at_critical_state,
    "winner_xml": winner_first_output,
    "loser_xml": loser_first_output,
    "score_winner": score_winner,
    "score_loser": score_loser,
    "metadata": {
        "episode_id": episode_id,
        "step_id": t,
        "trigger_reason": trigger_reason,
        "distance_to_expert": D_t,
        "progress": P_t,
    }
}
```

---

## 35. 必须使用 normal prompt 训练

分支生成时：

```text
normal_prompt + forced_continue_prompt
normal_prompt + forced_replan_prompt
```

训练时：

```text
normal_prompt only
```

训练目标：

```text
log π(winner_xml | normal_prompt)
>
log π(loser_xml | normal_prompt)
```

不能使用 forced prompt 作为训练输入。

---

## 36. 样本过滤

保留：

```text
winner XML valid
winner action valid
score margin sufficient
branch outcome clear
first output contains required fields
```

丢弃：

```text
invalid XML
invalid action
both branches too short
both branches ambiguous
score margin too small
same action and same tool with no meaningful difference
```

---

## 37. 数据统计

训练前统计：

```text
number of critical states
continue wins
replan wins
average score margin
average branch length
invalid XML rate
invalid action rate
average distance-to-expert at trigger
average recovery progress
```

这些统计可以直接写进论文实验分析。

---

# Stage 8: RL / Preference Optimization

## 38. 可选训练方法

可以使用：

```text
DPO-style preference loss
GRPO-style group relative optimization
PPO-style RL
```

最小可行建议：

```text
先做 DPO-style preference tuning
再考虑 GRPO
```

原因：

```text
DPO 实现简单
不需要在线更新太复杂
更适合先验证 idea
```

---

## 39. DPO-style Loss

给定：

```text
prompt x
winner y_w
loser y_l
```

优化：

```text
log π(y_w | x) > log π(y_l | x)
```

强调：

```text
x = normal_prompt
不是 forced_branch_prompt
```

---

## 40. GRPO-style Loss

如果使用 GRPO，可以将同一 critical state 的两个 branch 看成 group：

```text
group = {continue_output, replan_output}
```

相对优势：

```text
A_i = score_i - mean(score_group)
```

优化：

```text
increase probability of positive-advantage XML
decrease probability of negative-advantage XML
```

但两个样本的 group size 较小，实际可以扩展为：

```text
continue sampled K times
replan sampled K times
```

不过第一版不建议增加采样成本。

---

## 41. Training Mask

训练时只对 assistant XML 输出计算 loss。

```text
System prompt                  mask = 0
Full instruction               mask = 0
Current observation            mask = 0
Recent history                 mask = 0
Recent actions                 mask = 0
Current compact plan           mask = 0
Allowed actions                mask = 0
Forced branch prompt           mask = 0

Assistant XML output           mask = 1
```

---

## 42. Warmup 建议

不要直接从 base VLM 上做 complex RL。

推荐顺序：

```text
1. XML format SFT
2. action validity SFT
3. basic VLN imitation / navigation SFT
4. synthetic recovery examples SFT
5. counterfactual preference RL
```

最小 warmup 数据：

```text
continue examples
replan examples
STOP-as-action examples
invalid action correction examples
plan update examples
```

---

# Stage 9: Evaluation

## 43. 标准 VLN 指标

需要报告：

```text
SR
SPL
NE
nDTW
sDTW
path length
collision rate
```

---

## 44. Recovery-specific 指标

建议新增：

```text
Deviation Recovery Rate
Replan Success Rate
Unnecessary Replan Rate
Distance-to-Expert Reduction
Recovery Progress
Replan Frequency
Cooldown Violation Rate
Invalid XML Rate
Invalid Action Rate
```

### Deviation Recovery Rate

在发生 deviation 后，agent 是否能回到 expert path 附近或继续向目标推进。

### Replan Success Rate

模型输出 replan 后，接下来 H 步是否带来 positive recovery progress。

### Unnecessary Replan Rate

在没有明显偏离时输出 replan 的比例。

### Distance-to-Expert Reduction

```text
D_before_replan - D_after_replan
```

### Replan Frequency

平均每个 episode replan 次数。

太低说明模型不会恢复；太高说明模型过度重规划。

---

## 45. Ablation

推荐 ablation：

```text
1. w/o plan state
2. w/o replan tool
3. w/o expert-guided trigger
4. w/o counterfactual branch
5. w/o recovery cooldown
6. continue/replan without branch evaluation
7. full plan output vs compact plan update
8. branch horizon 30 vs 50
9. one rollout vs two rollouts per instruction
10. distance-only trigger vs distance+progress trigger
```

---

## 46. Baselines

可以比较：

```text
Base VLM planner
VLM + CoT prompt
VLM + persistent plan only
VLM + replan prompt only
SFT recovery model
standard RL without counterfactual branching
```

重点证明：

```text
不是仅仅因为多了 plan
不是仅仅因为多了 replan 词
而是 counterfactual recovery RL 带来了 improvement
```

---

# Stage 10: 实验日志与 Debug

## 47. 每步日志

每个 step 记录：

```python
{
    "episode_id": episode_id,
    "step": t,
    "instruction": instruction,
    "tool": tool,
    "subgoal": subgoal,
    "action": action,
    "valid_xml": valid_xml,
    "valid_action": valid_action,
    "plan_state": plan_state,
    "cooldown": cooldown,
    "agent_pose": agent_pose,
    "distance_to_expert": D_t,
    "progress": P_t,
}
```

---

## 48. Branch 日志

每个 branch 记录：

```python
{
    "episode_id": episode_id,
    "critical_step": t,
    "forced_tool": "continue" or "replan",
    "first_output": xml,
    "branch_length": length,
    "score": score,
    "success": success,
    "recovery_progress": recovery_progress,
    "end_distance_to_expert": D_end,
    "invalid_rate": invalid_rate,
}
```

---

## 49. 可视化 Debug

建议保存：

```text
trajectory top-down map
expert path
agent normal rollout path
continue branch path
replan branch path
critical state marker
```

这对论文 figure 也有用。

一张典型图：

```text
expert path: green
normal rollout before trigger: blue
continue branch: red
replan branch: orange
critical state: star
```

---

## 50. 常见失败模式

### 50.1 过度 replan

表现：

```text
模型频繁输出 replan
但没有真正改善路径
```

解决：

```text
增加 unnecessary replan penalty
增加 cooldown
过滤低质量 replan winner
```

### 50.2 不愿 replan

表现：

```text
明显偏离仍输出 continue
```

解决：

```text
增加 replan winner 样本
提高 deviation trigger 质量
加 synthetic recovery SFT
```

### 50.3 replan 生成不可执行 subgoal

表现：

```text
subgoal 太抽象
例如 "recover the route"
```

解决：

```text
prompt 要求 short executable local instruction
训练数据中加入具体 recovery subgoal
```

### 50.4 XML 格式不稳定

表现：

```text
缺标签
多 action
free-form explanation
```

解决：

```text
XML format SFT
parser retry
strong invalid format penalty
```

### 50.5 Branch 成本太高

表现：

```text
训练太慢
Habitat rollout 成本过大
```

解决：

```text
H 从 30 开始
每个 rollout 只 branch 一次
只对明显 deviation 触发
先离线收集 preference dataset
```

---

# Stage 11: 最小可行版本 MVP

## 51. MVP 目标

先证明：

```text
expert-guided deviation state 上，
replan branch 比 continue branch 更容易恢复。
```

不必一开始完整训练大模型。

---

## 52. MVP 配置

```text
benchmark: R2R-CE small split
model: Qwen3-VL-4B or similar VLM
action space: MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP
normal rollouts per instruction: 1 or 2
branch horizon: 30
trigger: distance-to-expert > tau_d
branch: continue vs replan
training: collect preference pairs first
```

---

## 53. MVP 验证指标

优先验证：

```text
1. 触发点是否合理
2. replan branch 是否经常优于 continue branch
3. preference pair 是否清晰
4. replan 输出是否可解析
5. recovery progress 是否为正
```

如果这些成立，再进入 RL tuning。

---

# Stage 12: 论文实现描述

## 54. Method 中要写清楚

Method 部分建议结构：

```text
1. Persistent Plan State
2. Plan-tool Interface
3. Expert-guided Deviation Detection
4. Counterfactual Branching
5. Branch Evaluation
6. Preference Optimization
7. Inference with Recovery Cooldown
```

---

## 55. 核心 Claim

英文：

```text
CFRP-VLN formulates recovery in continuous VLN as a counterfactual plan-tool decision problem. At training time, expert trajectories identify deviation states where the current plan becomes unreliable. From each deviation state, we compare continuing the current plan against recovery-oriented replanning through short counterfactual rollouts, and optimize the VLM planner to choose the better decision under the normal prompt.
```

中文：

```text
CFRP-VLN 将连续 VLN 中的恢复问题建模为反事实 plan-tool 决策问题。训练时利用 expert trajectory 检测当前 plan 失效的偏离状态，并从同一状态展开 continue 与 recovery-oriented replan 两个短反事实分支，通过环境结果构造偏好信号，使 VLM 学会何时坚持当前计划、何时主动恢复重规划。
```

---

# Stage 13: 执行优先级

## 56. 第一优先级

```text
1. Habitat/R2R-CE 环境 wrapper
2. expert trajectory 加载
3. XML parser
4. prompt builder
5. basic VLM step loop
6. action validation
```

## 57. 第二优先级

```text
1. PlanState 数据结构
2. continue/replan 输出接口
3. replan cooldown
4. plan update application
5. logging
```

## 58. 第三优先级

```text
1. deviation trigger
2. checkpoint/restore
3. continue branch
4. replan branch
5. branch truncation
6. branch evaluation
```

## 59. 第四优先级

```text
1. preference dataset construction
2. DPO-style tuning
3. GRPO-style tuning
4. ablation
5. paper figures
```

---

## 60. 最终 Checklist

实现前检查：

```text
[ ] 环境可以 reset/step
[ ] 可以读取 expert trajectory
[ ] 可以获取 agent pose
[ ] 可以计算 distance to expert path
[ ] 可以保存/恢复 critical state
[ ] VLM 能输出 XML
[ ] Parser 能解析 continue/replan
[ ] Action 一定来自 allowed actions
[ ] Controller 能维护 plan state
[ ] Replan 能更新 current/future
[ ] STOP 被当作 action 而不是 tool
[ ] Normal prompt 完整实现
[ ] Forced branch prompt 完整实现
[ ] Forced prompt 不参与最终 loss
[ ] Branch rollout 禁止递归 branching
[ ] Branch horizon 截断实现
[ ] Branch score 实现
[ ] Preference pair 构造实现
[ ] Cooldown 实现
[ ] 日志和可视化实现
```

---

## 61. 最终一句话执行原则

> 先让系统能稳定执行 `continue/replan` XML 闭环，再让 expert trajectory 触发反事实分支，最后用 branch 结果训练模型学会在 normal prompt 下自主选择 `continue` 或 `replan`。
