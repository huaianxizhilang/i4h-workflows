# i4h-workflows 分层部署指南

在空白 Ubuntu GPU 服务器上，以 Docker 方式部署并验证 [`robotic_ultrasound`](../workflows/robotic_ultrasound/README.md) 工作流。

本目录提供**可复用的分层脚本**（L0–L6）和**两种部署模式**：

| 模式 | 脚本 | 适用场景 |
|------|------|----------|
| **Mode A** | `modes/deploy-on-gpu-server.sh` | 你已 SSH 登录 GPU 服务器，在服务器上 clone 并部署 |
| **Mode B** | `modes/deploy-to-remote-gpu.sh` | 你在跳板机/本机，通过 root 密码或 SSH 密钥，远程部署到新购 GPU 服务器 |

---

## 目录结构

```text
deploy/
├── README.md                          # 本文档
├── config/
│   ├── defaults.env                   # 默认配置（可提交 git）
│   └── local.env.example              # 本地覆盖模板（复制为 local.env）
├── scripts/
│   └── common.sh                      # 公共函数
├── layers/                            # 可复用分层（L0–L6）
│   ├── L0-preflight.sh                # 硬件/OS 预检
│   ├── L1-system.sh                   # 系统包 + 缓存目录
│   ├── L2-nvidia-driver.sh            # NVIDIA 驱动
│   ├── L3-docker.sh                   # Docker + NVIDIA Container Toolkit
│   ├── L4-gui.sh                      # X11 / VNC GUI
│   ├── L5-i4h-project.sh              # 克隆仓库 + RTI 许可
│   └── L6-robotic-ultrasound.sh       # 构建镜像
├── modes/
│   ├── deploy-on-gpu-server.sh        # Mode A
│   └── deploy-to-remote-gpu.sh        # Mode B
├── verify/
│   └── verify-all.sh                  # 部署后验证
└── run/
    └── run-robotic-ultrasound.sh      # 启动 full_pipeline
```

### 分层复用说明

| 层级 | 作用 | 复用于 |
|------|------|--------|
| L0 | GPU/内存/磁盘预检 | 所有 i4h GPU 工作流 |
| L1 | apt 基础包、缓存目录、DDS 防火墙 | 所有 i4h 项目 |
| L2 | NVIDIA 驱动 | 所有 GPU 项目 |
| L3 | Docker + GPU 透传 | 所有 i4h Docker 工作流 |
| L4 | 远程 GUI | 需要 Isaac Sim 窗口的项目 |
| L5 | git clone、`./i4h`、RTI 许可 | 本仓库所有子项目 |
| L6 | 构建指定 workflow 镜像 | 按 `I4H_WORKFLOW` 切换 |

切换其他工作流（如 `robotic_surgery`）只需修改 `config/local.env`：

```bash
I4H_WORKFLOW=robotic_surgery
```

然后重新运行 L6。

---

## 硬件与系统要求

部署 `robotic_ultrasound` 前请确认：

| 项目 | 要求 |
|------|------|
| GPU | Compute Capability ≥ 8.6，**必须有 RT Core**（RTX 3090/4090/A6000 等） |
| 不可用 GPU | A100、H100（无 RT Core，超声射线追踪无法工作） |
| 显存 | ≥ 24 GB（微调建议 ≥ 48 GB） |
| 系统内存 | ≥ 64 GB 推荐 |
| 磁盘 | ≥ 100 GB 可用 |
| 系统 | Ubuntu 22.04 / 24.04 x86_64 |
| 驱动 | ≥ 535（推荐 ≥ 555） |

验证 GPU：

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

---

## Mode A：SSH 到 GPU 服务器本地部署

**流程**：你的电脑 → SSH → GPU 服务器 → clone 仓库 → 在服务器上执行部署。

### 步骤 1：购买并登录 GPU 服务器

选择带 **RTX 4090** 等型号的按量实例，系统选 **Ubuntu 22.04/24.04**。

```bash
# 本地终端（需要 GUI 时加 -X）
ssh -X root@<GPU_SERVER_IP>
# 或
ssh root@<GPU_SERVER_IP>
```

### 步骤 2：克隆仓库

```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
```

### 步骤 3：（可选）自定义配置

```bash
cp deploy/config/local.env.example deploy/config/local.env
vim deploy/config/local.env
```

常用配置：

```bash
# GUI 模式：auto（自动）| x11-ssh | vnc | headless
I4H_GUI_MODE=vnc

# 跳过已完成的镜像构建
I4H_SKIP_BUILD=0
```

### 步骤 4：一键部署

```bash
chmod +x deploy/layers/*.sh deploy/modes/*.sh deploy/verify/*.sh deploy/run/*.sh deploy/scripts/common.sh
bash deploy/modes/deploy-on-gpu-server.sh
```

脚本按顺序执行 L0 → L6。若 L2 安装了新驱动，会提示 **重启**，重启后：

```bash
cd ~/i4h-workflows   # 或你的 clone 路径
bash deploy/modes/deploy-on-gpu-server.sh --from L3
```

### 步骤 5：验证

```bash
bash deploy/verify/verify-all.sh
```

### 步骤 6：运行工作流

```bash
source ~/.i4h-deploy.env
bash deploy/run/run-robotic-ultrasound.sh
```

等价于：

```bash
cd ~/i4h-workflows
xhost +local:docker
./i4h run robotic_ultrasound full_pipeline --as-root --no-docker-build
```

**首次运行**：镜像构建 30–60 分钟，Isaac 初始化 10–15 分钟属正常。

### Mode A 流程图

```text
[你的电脑]
    │ ssh (-X 可选)
    ▼
[GPU 服务器]
    │ git clone i4h-workflows
    │ bash deploy/modes/deploy-on-gpu-server.sh
    │   ├─ L0 预检
    │   ├─ L1 系统包
    │   ├─ L2 驱动 (可能需 reboot)
    │   ├─ L3 Docker+GPU
    │   ├─ L4 GUI
    │   ├─ L5 项目+RTI
    │   └─ L6 构建镜像
    │ bash deploy/verify/verify-all.sh
    │ bash deploy/run/run-robotic-ultrasound.sh
    ▼
[Isaac Sim + Pi0 + 超声可视化]
```

---

## Mode B：从跳板机远程部署到新购 GPU 服务器

**流程**：你的电脑 → SSH → **跳板机**（非 GPU 服务器）→ 通过 root 账户 → 部署到 **新购 GPU 按量服务器**。

适用于：GPU 服务器刚创建、你手头只有 root 密码、希望从稳定环境一键推送部署。

### 步骤 1：登录跳板机

```bash
ssh user@<JUMP_HOST>
```

跳板机需要：`git`（可选）、`ssh`、`rsync`。若使用密码登录 GPU，还需 `sshpass`：

```bash
sudo apt install -y git rsync openssh-client sshpass
```

### 步骤 2：获取部署脚本

**方式 A** — 在跳板机 clone 仓库：

```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
```

**方式 B** — 你已在本地有仓库，scp 上去：

```bash
# 在本地执行
scp -r deploy user@<JUMP_HOST>:~/i4h-deploy/
ssh user@<JUMP_HOST>
cd ~/i4h-deploy/..
```

### 步骤 3：配置 GPU 服务器连接信息

```bash
cd i4h-workflows
cp deploy/config/local.env.example deploy/config/local.env
vim deploy/config/local.env
```

填写：

```bash
I4H_REMOTE_HOST=<新购GPU服务器IP>
I4H_REMOTE_USER=root
I4H_REMOTE_PORT=22
I4H_REMOTE_PASSWORD=<root密码>

# 远程无桌面时用 VNC
I4H_GUI_MODE=vnc
```

> **安全建议**：优先配置 SSH 密钥，避免在配置文件中写密码：
>
> ```bash
> ssh-keygen -t ed25519 -N "" -f ~/.ssh/i4h_gpu
> ssh-copy-id -i ~/.ssh/i4h_gpu.pub root@<GPU_SERVER_IP>
> # 然后省略 I4H_REMOTE_PASSWORD
> ```

### 步骤 4：远程一键部署

```bash
chmod +x deploy/layers/*.sh deploy/modes/*.sh deploy/verify/*.sh deploy/run/*.sh deploy/scripts/common.sh
bash deploy/modes/deploy-to-remote-gpu.sh
```

或命令行传参（不落盘密码）：

```bash
I4H_REMOTE_HOST=203.0.113.10 \
I4H_REMOTE_PASSWORD='your-password' \
bash deploy/modes/deploy-to-remote-gpu.sh
```

脚本会：

1. SSH 测试连接 GPU 服务器
2. `rsync` 上传 `deploy/` 到 GPU 服务器 `/tmp/i4h-deploy/`
3. 在 GPU 服务器上远程执行 `deploy-on-gpu-server.sh`（L0–L6）
4. L5 会在 GPU 服务器上 `git clone` i4h-workflows（除非使用 `--use-jump-repo`）

若驱动安装需重启：

```bash
# GPU 服务器重启后，在跳板机重新执行
bash deploy/modes/deploy-to-remote-gpu.sh --from L3
```

### 步骤 5：远程验证

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --verify-only
```

或 SSH 到 GPU 服务器：

```bash
ssh root@<GPU_SERVER_IP>
bash /tmp/i4h-deploy/verify/verify-all.sh
```

### 步骤 6：SSH 到 GPU 服务器运行工作流

```bash
ssh -X root@<GPU_SERVER_IP>   # GUI 用 -X
source ~/.i4h-deploy.env
bash /tmp/i4h-deploy/run/run-robotic-ultrasound.sh
```

**VNC 方式**（推荐无 `-X` 时）：

```bash
ssh root@<GPU_SERVER_IP>
export I4H_GUI_MODE=vnc
bash /tmp/i4h-deploy/layers/L4-gui.sh
# 本地 VNC 客户端连接 <GPU_IP>:5901
bash /tmp/i4h-deploy/run/run-robotic-ultrasound.sh
```

### Mode B 可选：同步整个仓库（免远程 git clone）

若跳板机网络比 GPU 服务器好，可把整个仓库 rsync 过去：

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --use-jump-repo
```

### Mode B 流程图

```text
[你的电脑]
    │ ssh
    ▼
[跳板机] ──ssh/rsync──► [新购 GPU 服务器 root@IP]
    │                      │
    │ deploy-to-remote-gpu │ deploy-on-gpu-server (L0-L6)
    │                      │ git clone (L5)
    │                      │ docker build (L6)
    ▼                      ▼
  查看日志              verify + run workflow
```

---

## 分层脚本单独执行

任意层可独立运行，便于调试或复用到其他项目：

```bash
cd i4h-workflows
source deploy/config/defaults.env
[[ -f deploy/config/local.env ]] && source deploy/config/local.env

bash deploy/layers/L3-docker.sh
bash deploy/layers/L5-i4h-project.sh
```

从指定层继续：

```bash
bash deploy/modes/deploy-on-gpu-server.sh --from L4 --to L6
```

---

## GUI 模式选择

| `I4H_GUI_MODE` | 说明 | 适用 |
|----------------|------|------|
| `auto` | 有 `DISPLAY` 用 X11，否则回退 VNC | 默认 |
| `x11-ssh` | 要求 `ssh -X`，失败则报错 | Mode A 本地 X11 |
| `vnc` | 启动 TigerVNC + XFCE | 远程 GPU 无桌面 |
| `headless` | Xvfb 虚拟显示 | 仅测试、无窗口 |

Docker 访问 X11：

```bash
xhost +local:docker
```

---

## 部署后常用命令

```bash
# 加载环境变量
source ~/.i4h-deploy.env

# 仅验证
bash deploy/verify/verify-all.sh

# 完整流水线（仿真 + Pi0 + 超声 + 可视化）
bash deploy/run/run-robotic-ultrasound.sh

# 分进程调试
cd ~/i4h-workflows
./i4h run robotic_ultrasound sim_env --as-root --no-docker-build
./i4h run robotic_ultrasound pi0_policy --as-root --no-docker-build
./i4h run robotic_ultrasound visualization --as-root --no-docker-build

# 强制重建镜像
I4H_BUILD_NO_CACHE=1 bash deploy/layers/L6-robotic-ultrasound.sh

# 进入容器
./i4h run-container robotic_ultrasound --as-root
```

---

## 故障排查

| 现象 | 处理 |
|------|------|
| L2 后退出码 2 | 重启 GPU 服务器， `--from L3` 继续 |
| `No NVIDIA GPU detected` | 检查 L3：`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` |
| 黑屏 / 无 GUI | 设置 `I4H_GUI_MODE=vnc` 或 `ssh -X` + `xhost +local:docker` |
| RTI license 错误 | 确认 `RTI_LICENSE_FILE` 存在；或设置 `RTI_LICENSE_URL` 自动下载 |
| 无超声图像 | GPU 无 RT Core（A100/H100）；换 RTX 4090 等 |
| 首次极慢 | 正常；资产缓存于 `~/docker/isaac-sim/` 和 `~/.cache/` |
| Mode B SSH 失败 | 检查 IP/密码；安装 `sshpass`；或配置 SSH 密钥 |
| docker 权限 denied | `sudo usermod -aG docker $USER` 后重新登录 |

清理残留进程：

```bash
bash workflows/robotic_ultrasound/reset.sh
```

---

## 与其他 i4h 工作流复用

1. 在 GPU 服务器上完成 **L0–L5**（一次即可）
2. 修改 `deploy/config/local.env`：

   ```bash
   I4H_WORKFLOW=robotic_surgery   # 或 so_arm_starter, telesurgery 等
   ```

3. 仅运行：

   ```bash
   bash deploy/layers/L6-robotic-ultrasound.sh
   bash deploy/run/run-robotic-ultrasound.sh <mode>
   ```

---

## 相关文档

- [robotic_ultrasound README](../workflows/robotic_ultrasound/README.md)
- [robotic_ultrasound Docker 指南](../workflows/robotic_ultrasound/docker/README.md)
- [i4h CLI 说明](../i4h)
