# PI0 推理调试指南

在 `robotic_ultrasound` 工作流上调试 PI0 **推理**（LoRA 微调入口已封装，尚未完整验证）。

完整 Cursor 对话备份见：`backup/cursor-conversation-pi0-deploy-debug-2026-06-14.md`

---

## 0. 前置条件（GPU 服务器 `106.75.237.179`）

```bash
cd /data/i4h-workflows

# 一次性：从镜像同步 third_party（openpi / IsaacLab / lerobot 等）
bash deploy/debug/pi0-debug.sh sync-deps

# 模型权重：优先从旧容器复制，否则镜像下载
bash deploy/debug/pi0-debug.sh sync-hf-cache \
  || bash deploy/debug/pi0-debug.sh download-model   # hf-mirror.com，可断点续传

# 冒烟验证（有缓存时离线加载）
bash deploy/debug/pi0-debug.sh verify

# 调试前务必停掉 full_pipeline / 旧调试容器
bash deploy/debug/pi0-debug.sh stop
```

**GUI 模式（Isaac Sim / VNC）额外要求：**

```bash
export DISPLAY=:1          # TigerVNC 已启动
xhost +local:docker        # 允许容器使用 X11
```

权重默认缓存在 `/data/cache/huggingface`；有缓存时脚本设 `HF_HUB_OFFLINE=1`。

---

## 1. Cursor / VS Code 连接

1. **Remote-SSH** → `ubuntu@106.75.237.179`
2. 打开文件夹 **`/data/i4h-workflows`**（宿主机路径，非容器内）
3. 安装 Python 扩展（debugpy）
4. 复制 launch 配置（`.vscode/` 被 gitignore，用示例文件）：

   ```bash
   mkdir -p .vscode
   cp workflows/robotic_ultrasound/scripts/debug/launch.json.example .vscode/launch.json
   ```

5. SSH 端口转发（本机终端）：

   ```bash
   ssh -L 5678:localhost:5678 -L 5679:localhost:5679 -L 5680:localhost:5680 ubuntu@106.75.237.179
   ```

6. 路径映射：宿主机 `/data/i4h-workflows` ↔ 容器 `/workspace/i4h-workflows`

---

## 2. 六种调试模式对照表

| Cursor Attach 配置 | `pi0-debug.sh` 命令 | 端口 | 进程模型 | 适用场景 |
|-------------------|---------------------|------|----------|----------|
| **PI0: Attach Policy Only (fast)** | `infer-policy` | 5678 | 单进程，无 Isaac Sim | **最快**：~30s 到 `infer()` 断点，学 openpi 推理 |
| **PI0: Attach Inference (eval.py)** | `infer` | 5678 | 单进程 `eval.py` | 完整仿真推理，headless，启动慢（5–15 min） |
| **PI0: Attach Frame GUI (unified freeze)** | `infer-frame-gui` | 5678 | 单进程 + Isaac 视口 | **推荐**：逐帧调试，断点冻结仿真+推理 |
| **PI0: Attach Full Frame (DDS all windows)** | `infer-full-frame` | 5678 | 多进程 DDS + SIGSTOP | 三画面 + 超声，接近生产 pipeline |
| **PI0: Attach Policy DDS** | `sim` + `policy` | 5680 | 双终端分进程 | 只调试 policy DDS，需先起 sim |
| **PI0: Attach LoRA Train** | `train` | 5679 | 单进程训练 | 微调入口（**未完整验证**） |

### 不要用 `full_pipeline` 做断点调试

`full_pipeline` 适合 VNC 演示；调试请用上面六种模式之一。

---

## 3. 各模式详细步骤

### 3.1 Policy Only（最快学推理）

**目标**：不进 Isaac Sim，直接断在 `PI0PolicyRunner.infer()` 和 `openpi/policies/policy.py`。

```bash
bash deploy/debug/pi0-debug.sh infer-policy
```

1. 终端出现 `Listening on 0.0.0.0:5678`
2. Cursor → **PI0: Attach Policy Only (fast)** → F5
3. 建议断点：
   - `policy/pi0/runners.py` → `infer()`
   - `third_party/openpi/src/openpi/policies/policy.py` → `infer()`
4. F5 Continue → 打印 `actions shape` 后退出

**首次 infer**：JAX 编译约 20s（正常）；后续更快。

---

### 3.2 Inference / eval.py（headless 完整仿真）

**目标**：与生产 `eval.py` 相同逻辑，无 GUI。

```bash
# 可选：无 DISPLAY 时默认 headless
bash deploy/debug/pi0-debug.sh infer
```

1. Attach **PI0: Attach Inference (eval.py)**
2. 等待 Isaac Sim 初始化（5–15 分钟）
3. 断点同上；循环中每 5 步 `replan_steps` 调用一次 `infer()`

环境变量：

```bash
HEADLESS=0 bash deploy/debug/pi0-debug.sh infer   # 需要 DISPLAY + VNC
```

---

### 3.3 Frame GUI（单进程逐帧，推荐）

**目标**：一个断点冻结 **Isaac 视口 + 推理 +（可选）子进程**；适合「推一帧看一帧」。

```bash
export DISPLAY=:1 && xhost +local:docker
bash deploy/debug/pi0-debug.sh infer-frame-gui
```

1. TigerVNC 连 `106.75.237.179:5901`，应看到 **Isaac Sim 窗口**
2. Attach **PI0: Attach Frame GUI (unified freeze)**
3. 终端日志示例：

   ```
   [frame-debug] frame=5 stage=infer timestep=5 before infer()
   [frame-debug] infer took 108.8 ms
   [frame-debug] frame=5 stage=step timestep=5 before env.step()
   ```

4. **每停一次 = 等你 F5 Continue**，不是卡死

#### 暂停频率（`I4H_DEBUG_STOP`）

| 值 | 行为 |
|----|------|
| `both`（默认） | 每个 `infer` 和每个 `env.step` 都停 |
| `infer` | 仅 `infer` 前停（约每 5 步 1 次） |
| `step` | 仅 `env.step` 前停 |
| `off` | 不停，只靠手动断点 |

```bash
I4H_DEBUG_STOP=infer bash deploy/debug/pi0-debug.sh infer-frame-gui
```

#### 其他环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `I4H_GUI` | `1` | `1` = 有 Isaac 视口 |
| `I4H_INLINE_PANEL` | `1` | 同进程 DearPyGui 小面板 |
| `I4H_DEBUG_RESET` | `0` | `0` = 跳过 reset 阶段 40 步断点 |
| `I4H_USE_DEBUGPY` | `1` | `0` = 不用 `debugpy.breakpoint()` |
| `JAX_LOG_LEVEL` | `WARNING` | 抑制 JAX 编译 DEBUG 刷屏 |

#### 已知注意点

- **openpi 必须在 AppLauncher 之前 import**（否则 pydantic 与 Isaac Sim 冲突）
- 第一次 `infer` ~20s（JAX 编译），之后 ~100ms
- 内联 DearPyGui 可能被 Isaac 窗口挡住；完整三画面用 `infer-full-frame`

---

### 3.4 Full Frame（DDS 三画面 + 超声）

**目标**：与 `full_pipeline` 相同的 **4 进程**（sim + vis + ultrasound + policy），policy 进程可断点；子进程用 **SIGSTOP** 与 policy 同步冻结。

```bash
export DISPLAY=:1 && xhost +local:docker
bash deploy/debug/pi0-debug.sh infer-full-frame
```

1. 脚本先启动 `utils.visualization`、`ultrasound_raytracing`、`sim_with_dds`
2. 等待 **~120s**（`SIM_WAIT_SEC`）让 Isaac Sim 起来
3. VNC 应看到：**Isaac Sim** + **Robotic Ultrasound Visualization**（Room / Wrist / B-mode）
4. 终端 `Listening on 0.0.0.0:5678` 后 → Attach **PI0: Attach Full Frame (DDS all windows)**
5. 收到 DDS 三路数据后 → `_pause("infer")` → `policy.infer()` → `_pause("step")` → 发布控制

```bash
I4H_DEBUG_STOP=infer bash deploy/debug/pi0-debug.sh infer-full-frame
SIM_WAIT_SEC=180 bash deploy/debug/pi0-debug.sh infer-full-frame   # sim 慢时加长
```

**曾修复的 bug**：`PolicyPublisher` 缺少 `__init__` 导致 attach 后立即 `TypeError`（已对齐 `run_policy.py`）。

---

### 3.5 Policy DDS（分进程，生产架构）

**目标**：终端 1 跑仿真，终端 2 单独调试 policy DDS。

```bash
# 终端 1（GPU 服务器）
bash deploy/debug/pi0-debug.sh sim

# 终端 2（sim 就绪后）
bash deploy/debug/pi0-debug.sh policy
```

Attach **PI0: Attach Policy DDS**（端口 **5680**）。

---

### 3.6 LoRA Train（未完整验证）

```bash
bash deploy/debug/pi0-debug.sh train
```

Attach **PI0: Attach LoRA Train**（端口 **5679**）。

建议断点：`training/pi_zero/train.py`、`third_party/openpi/src/openpi/train.py`

> 训练需大量显存；调试前务必 `pi0-debug.sh stop` 释放 GPU。

---

## 4. 停止调试

```bash
bash deploy/debug/pi0-debug.sh stop
```

Cursor 里 **Shift+F5** 只断开 debugpy，不杀容器；卡住时用 `stop` 或 `docker rm -f i4h-pi0-debug`。

---

## 5. 推荐断点

| 阶段 | 文件 | 函数 |
|------|------|------|
| i4h 推理封装 | `policy/pi0/runners.py` | `infer()` |
| OpenPI 推理 | `third_party/openpi/src/openpi/policies/policy.py` | `infer()` |
| PI0 网络 | `third_party/openpi/src/openpi/models/pi0.py` | forward |
| eval 主循环 | `simulation/imitation_learning/pi0_policy/eval.py` | main 循环 |
| 帧调试入口 | `debug/debug_infer_frame_gui.py` | `_debug_pause()` |
| DDS 帧调试 | `debug/debug_policy_dds_frame.py` | `_pause()` |
| 训练入口 | `training/pi_zero/train.py` | `main` |

---

## 6. 推理调用链（简图）

```text
eval.py / debug_infer_frame_gui.py / run_policy.py
    → PI0PolicyRunner.infer()          [policy/pi0/runners.py]
        → Policy.infer()               [openpi/policies/policy.py]
            → PI0 模型 forward         [openpi/models/pi0.py]
```

`replan_steps=5`：每 5 个仿真步调用一次 `infer()`，一次产出 50 步 action chunk。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 推 Continue 后「停住」 | `_debug_pause` 在等下一次 F5 | 正常；看 Call Stack 是否在 `_debug_pause` |
| 第一次 infer ~23s | JAX XLA 首次编译 | 正常，仅一次 |
| `ModuleNotFoundError: isaaclab` | 未 `sync-deps` | `bash deploy/debug/pi0-debug.sh sync-deps` |
| 卡在 `Starting simulation` | 无 `DISPLAY` | `export DISPLAY=:1 && xhost +local:docker` |
| infer-full-frame policy 崩溃 | `PolicyPublisher` 参数 | 已修复，拉最新代码 |
| HuggingFace 很慢 | 直连 huggingface.co | `sync-hf-cache` 或 `download-model`（镜像） |

---

## 8. 源码位置

| 层级 | 路径 |
|------|------|
| i4h 封装 | `workflows/robotic_ultrasound/scripts/policy/pi0/` |
| 训练入口 | `workflows/robotic_ultrasound/scripts/training/pi_zero/` |
| OpenPI 核心 | `third_party/openpi/src/openpi/`（`sync-deps` 后可见） |
| 调试脚本 | `workflows/robotic_ultrasound/scripts/debug/` |
| 调试入口 | `deploy/debug/pi0-debug.sh` |
