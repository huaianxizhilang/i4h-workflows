# i4h-workflows 分层部署指南

在空白 Ubuntu GPU 服务器上，以 Docker 方式部署并验证 [`robotic_ultrasound`](../workflows/robotic_ultrasound/README.md) 工作流。

---

## 快速开始：选一条路

| 你的情况 | 阅读章节 |
|----------|----------|
| 已 SSH 登录 **GPU 服务器**，在服务器上 clone 并部署 | [Mode A](#mode-a在-gpu-服务器上部署) |
| 在 **跳板机**上，通过 root 密码/密钥部署到**新购 GPU 服务器** | [Mode B](#mode-b从跳板机部署到-gpu-服务器) |
| 想了解无桌面 Ubuntu 如何显示 Isaac Sim 画面 | [无桌面 Ubuntu 与远程桌面](#无桌面-ubuntu-与远程桌面) |
| 系统盘小（如 50GB）、有大容量数据盘 | [数据盘配置](#数据盘配置系统盘-50gb) |
| 部署完成后的运行、调试命令 | [部署后常用命令](#部署后常用命令) |

两种模式都执行同一套分层脚本（L0–L6），区别只在**命令在哪里执行**：

| 模式 | 执行位置 | 入口脚本 |
|------|----------|----------|
| **Mode A** | GPU 服务器 | `modes/deploy-on-gpu-server.sh` |
| **Mode B** | 跳板机（自动 SSH 到 GPU 服务器） | `modes/deploy-to-remote-gpu.sh` |

---

## 目录与分层

```text
deploy/
├── config/          defaults.env、local.env（本地配置，不提交 git）
├── layers/          L0–L6 可复用分层脚本
├── modes/           Mode A / Mode B 入口
├── scripts/         公共函数
├── verify/          部署验证
└── run/             启动工作流
```

| 层级 | 作用 |
|------|------|
| L0 | GPU / 内存 / 磁盘预检 |
| L1 | 系统包、缓存目录、DDS 防火墙 |
| L2 | NVIDIA 驱动（可能需要重启） |
| L3 | Docker + NVIDIA Container Toolkit |
| L4 | GUI（X11 / TigerVNC + XFCE） |
| L5 | 克隆仓库、RTI 许可 |
| L6 | 构建 `robotic_ultrasound` 镜像 |

切换其他工作流：在 `config/local.env` 设 `I4H_WORKFLOW=robotic_surgery`，再单独跑 L6。

---

## 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | Compute Capability ≥ 8.6，**必须有 RT Core**（RTX 3090/4090/A6000 等） |
| 不可用 | A100、H100（无 RT Core，超声射线追踪无法工作） |
| 显存 | ≥ 24 GB |
| 内存 / 磁盘 | ≥ 64 GB RAM，≥ 100 GB 可用磁盘 |
| 系统 | Ubuntu 22.04 / 24.04 x86_64 |

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

---

## 数据盘配置（系统盘 50GB）

云 GPU 常见布局：**系统盘 ~50GB**，大容量盘挂载在 **`/data`**。  
Isaac 镜像、Docker 层、模型缓存体积很大，**务必在首次部署前**配置数据盘。

### 一行配置（推荐）

在 `deploy/config/local.env` 中设置：

```bash
I4H_DATA_ROOT=/data
```

脚本会自动把以下内容放到数据盘，并在 `$HOME` 下创建符号链接（兼容 `./i4h` 默认的 `~/docker`、`~/.cache` 路径）：

| 路径 | 内容 | 典型体积 |
|------|------|----------|
| `/data/docker-engine` | Docker 镜像与层（`daemon.json` data-root） | 数十 GB |
| `/data/i4h-workflows` | git 仓库与 build 产物 | 数 GB |
| `/data/docker/isaac-sim/` | Isaac Sim 运行缓存 | 数十 GB |
| `/data/cache/huggingface` | 模型下载 | 数 GB～数十 GB |
| `/data/cache/i4h-assets` | i4h 资产 | 数 GB |

仍留在系统盘上的（体积很小）：apt 包、驱动、TigerVNC 配置、`~/.vnc`、部署脚本。

### 自定义子路径（可选）

```bash
I4H_DATA_ROOT=/data
# I4H_INSTALL_DIR=/data/i4h-workflows
# I4H_DOCKER_ROOT=/data/docker
# I4H_CACHE_ROOT=/data/cache
# I4H_DOCKER_DATA_ROOT=/data/docker-engine
```

### 注意

1. **先挂载数据盘再部署** — `I4H_DATA_ROOT` 目录必须已存在（如 `df -h /data` 可见）。
2. **在 L3 之前配置** — Docker 首次拉镜像前就指向数据盘；已有过镜像的机器需手动迁移（脚本会提示）。
3. **Mode B** — 在跳板机 `local.env` 里写好 `I4H_DATA_ROOT=/data`，会随部署传到 GPU 服务器生效。

验证：

```bash
bash deploy/verify/verify-all.sh   # 检查 data-root 与 symlink
docker info | grep "Docker Root Dir"
df -h / /data
```

---

## 无桌面 Ubuntu 与远程桌面

云 GPU 实例多为**无图形界面的 Ubuntu**。这不影响部署，L4 会自动处理显示环境。

### 原理（只需连一次 VNC）

```text
你的电脑 ──VNC──► GPU 服务器 :5901
                      │
                      ├── XFCE 桌面（TigerVNC 虚拟显示 :1）
                      └── Docker 容器（Isaac Sim、DearPyGUI）
                              └── 共用 DISPLAY=:1，窗口出现在同一 VNC 桌面
```

**不需要**为 Docker 单独再配一套 VNC 或远程桌面。

### 建议在 `local.env` 里加的 GUI 配置

在 Mode A / Mode B 的步骤里创建 `deploy/config/local.env` 时，无桌面服务器建议写入：

```bash
I4H_GUI_MODE=vnc
I4H_VNC_PASSWORD=your-secure-password
```

| 变量 | 说明 |
|------|------|
| `I4H_GUI_MODE` | `vnc` 显式启用 TigerVNC；`auto`（默认）在无 `DISPLAY` 时也会自动回退到 VNC |
| `I4H_VNC_PASSWORD` | VNC 密码；**Mode B 必填**（非交互部署）；Mode A 建议填写 |
| `I4H_VNC_DISPLAY` | 默认 `:1`，对应端口 **5901** |
| `I4H_VNC_ALLOW_REMOTE` | 默认 `1`，允许外网连接；需在云安全组放行 **TCP 5901** |
| `I4H_VNC_AUTOSTART` | 默认 `1`，L4 注册 **systemd `i4h-vnc.service`**，重启后自动监听 5901 |

### 部署完成后如何看画面

1. 本地 VNC 客户端连接 `<GPU服务器IP>:5901`
2. 云安全组放行 TCP 5901
3. 在 VNC 桌面终端或 SSH 中运行工作流（见各 Mode 最后一步）

### GUI 模式一览

| 模式 | 说明 |
|------|------|
| `auto` | 有 `DISPLAY` 用 X11；无则自动装 VNC + XFCE |
| `vnc` | 显式安装 TigerVNC + XFCE（无桌面服务器推荐写明） |
| `x11-ssh` | 需 `ssh -X`，窗口转发到本机 |
| `headless` | 仅 Xvfb，**无远程桌面**，适合不看画面的后台跑 |

> **部署步骤不在此重复**，请按下方 Mode A 或 Mode B 操作。

---

## Mode A：在 GPU 服务器上部署

```text
你的电脑 ──ssh──► GPU 服务器
                    git clone → deploy-on-gpu-server.sh (L0–L6) → 运行工作流
```

### 1. 登录并克隆

```bash
ssh root@<GPU_SERVER_IP>
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
```

### 2. 配置（可选）

```bash
cp deploy/config/local.env.example deploy/config/local.env
vim deploy/config/local.env
```

无桌面 Ubuntu 建议加上（详见 [无桌面 Ubuntu 与远程桌面](#无桌面-ubuntu-与远程桌面)）：

```bash
I4H_DATA_ROOT=/data                    # 有大容量数据盘时强烈推荐
I4H_GUI_MODE=vnc
I4H_VNC_PASSWORD=your-secure-password
```

### 3. 一键部署

```bash
chmod +x deploy/layers/*.sh deploy/modes/*.sh deploy/verify/*.sh deploy/run/*.sh deploy/scripts/common.sh
bash deploy/modes/deploy-on-gpu-server.sh
```

若 L2 安装驱动后需重启，重启后继续：

```bash
bash deploy/modes/deploy-on-gpu-server.sh --from L3
```

### 4. 验证

```bash
bash deploy/verify/verify-all.sh
```

### 5. 连接 VNC 并运行

L4 部署阶段已启动 TigerVNC。本地 VNC 连接 `<GPU_SERVER_IP>:5901`（安全组放行 TCP 5901），然后：

```bash
source ~/.i4h-deploy.env
bash deploy/run/run-robotic-ultrasound.sh
```

Isaac Sim 与超声可视化窗口会出现在 VNC 桌面上。首次运行可能需 30–60 分钟构建 + 10–15 分钟初始化。

---

## Mode B：从跳板机部署到 GPU 服务器

```text
你的电脑 ──ssh──► 跳板机 ──ssh/rsync──► 新购 GPU 服务器 (root)
```

适用于：GPU 服务器刚创建、只有 root 密码、希望从稳定环境推送部署。

### 1. 登录跳板机

```bash
ssh user@<JUMP_HOST>
sudo apt install -y git rsync openssh-client sshpass   # 密码登录 GPU 时需要 sshpass
```

### 2. 获取仓库

```bash
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
cd i4h-workflows
```

### 3. 配置 GPU 连接与 GUI

```bash
cp deploy/config/local.env.example deploy/config/local.env
vim deploy/config/local.env
```

```bash
I4H_REMOTE_HOST=<新购GPU服务器IP>
I4H_REMOTE_USER=root
I4H_REMOTE_PORT=22
# I4H_REMOTE_PASSWORD=              # 已配 SSH 密钥则不需要

I4H_DATA_ROOT=/data                 # GPU 数据盘挂载点
I4H_GUI_MODE=vnc
I4H_VNC_PASSWORD=your-secure-password
```

### 4. 远程一键部署

```bash
chmod +x deploy/layers/*.sh deploy/modes/*.sh deploy/verify/*.sh deploy/run/*.sh deploy/scripts/common.sh
bash deploy/modes/deploy-to-remote-gpu.sh
```

驱动安装需重启时，重启后执行：

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --from L3
```

可选：跳板机网络更好时，同步整个仓库免远程 git clone：

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --use-jump-repo
```

### 5. 验证

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --verify-only
```

### 6. 连接 VNC 并运行

本地 VNC 连接 `<GPU_SERVER_IP>:5901`，然后 SSH 到 GPU 服务器：

```bash
ssh root@<GPU_SERVER_IP>
source ~/.i4h-deploy.env
bash /tmp/i4h-deploy/run/run-robotic-ultrasound.sh
```

> 部署时 L4 已在 GPU 服务器上启动 VNC，**无需**再手动执行 `L4-gui.sh`。

---

## 分层脚本单独执行

```bash
cd i4h-workflows
bash deploy/modes/deploy-on-gpu-server.sh --from L4 --to L6   # 从 L4 继续
bash deploy/layers/L3-docker.sh                                # 只跑某一层
```

---

## 部署后常用命令

```bash
source ~/.i4h-deploy.env
bash deploy/verify/verify-all.sh
bash deploy/run/run-robotic-ultrasound.sh

# 分进程调试
cd ~/i4h-workflows
./i4h run robotic_ultrasound sim_env --as-root --no-docker-build
./i4h run robotic_ultrasound pi0_policy --as-root --no-docker-build
./i4h run robotic_ultrasound visualization --as-root --no-docker-build

# 进入容器
./i4h run-container robotic_ultrasound --as-root
```

---

## 故障排查

| 现象 | 处理 |
|------|------|
| L2 后需重启 | 重启后 `--from L3` 继续 |
| `No NVIDIA GPU detected` | 重跑 L3，检查 `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` |
| VNC 连不上 | 检查安全组 TCP 5901、`I4H_VNC_PASSWORD`、`I4H_VNC_ALLOW_REMOTE=1`；重启后执行 `systemctl status i4h-vnc` 或 `bash deploy/layers/L4-gui.sh` |
| 黑屏 / 容器无窗口 | 确认 VNC 已连上；`xhost +local:docker` |
| RTI license 错误 | 设置 `RTI_LICENSE_FILE` 或 `RTI_LICENSE_URL` |
| 无超声图像 | GPU 无 RT Core，换 RTX 4090 等 |
| Mode B SSH 失败 | 检查 IP/密码，或配置 SSH 密钥 |

```bash
bash workflows/robotic_ultrasound/reset.sh   # 清理残留进程
```

---

## 与其他工作流复用

完成 L0–L5 后，修改 `I4H_WORKFLOW` 并只跑 L6：

```bash
# deploy/config/local.env
I4H_WORKFLOW=robotic_surgery
```

```bash
bash deploy/layers/L6-robotic-ultrasound.sh
bash deploy/run/run-robotic-ultrasound.sh
```

---

## 相关文档

- [robotic_ultrasound README](../workflows/robotic_ultrasound/README.md)
- [robotic_ultrasound Docker 指南](../workflows/robotic_ultrasound/docker/README.md)
- [i4h CLI](../i4h)
