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

## 优云智算 CompShare 购机指南

在 [CompShare 创建 GPU 实例](https://console.compshare.cn/light-gpu/resources/create) 时建议：

| 选项 | 推荐值 |
|------|--------|
| 区域 | **上海 B 区** |
| GPU | **48GB RTX 4090** |
| 数据盘 | **300GB**（挂载到 `/data`） |
| CPU 平台 | **x86_64** |
| 防火墙 | **vlatest**（已含 TCP **5901** VNC） |
| 系统镜像 | **虚机类型 → ubuntu-nvidia** |

购机后首次部署，在 `deploy/config/local.env` 开启新机初始化：

```bash
I4H_COMPSHARE_BOOTSTRAP=1          # DNS + 数据盘 + SSH 公钥
I4H_DATA_ROOT=/data
I4H_LOGIN_USER=ubuntu
I4H_SET_LOGIN_PASSWORD=vlatest     # 部署结束时改登录密码（替代平台随机密码）
I4H_VNC_PASSWORD=vlatest
I4H_GUI_MODE=vnc
```

`I4H_COMPSHARE_BOOTSTRAP=1` 时 L0 会自动完成：

1. **DNS 加速**（[优云 UAAA 文档](https://www.compshare.cn/docs/operation/gpu/uaaa)）— 写入 netplan `100.90.90.90` / `100.90.90.100` 并 `netplan apply`
2. **数据盘** — 自动识别约 200–400GB 的块设备（常见 `/dev/vdb`），`mkfs`/`mount`/`resize2fs`、UUID 写入 `/etc/fstab`、`chown ubuntu`
3. **SSH 公钥** — 写入 `deploy/config/ssh-authorized-keys.pub` 中的密钥到 `ubuntu`（及 `root`）
4. **DNS 验证** — 解析 github.com、nvidia.com、huggingface.co 等代表域名

### SSH 登录配置提醒

| 机器 | 操作 |
|------|------|
| **Mode B 跳板机** | 部署成功后会自动写入 `~/.ssh/config`，别名形如 `i4h-gpu-20260614-106-75-237-179`，直接 `ssh i4h-gpu-...` |
| **Mac / 公司电脑** | 需自行在 `~/.ssh/config` 添加 Host（跳板机 **约 3 个月过期**，过期后更新 HostName） |
| 公钥文件 | `deploy/config/ssh-authorized-keys.pub`（可增删，勿提交私钥） |

Mode B 推荐命令：

```bash
bash deploy/modes/deploy-to-remote-gpu.sh --use-jump-repo
```

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

云 GPU 常见布局：**系统盘 ~50GB**，大容量盘挂载在 **`/data`**（优云智算 300GB 盘常见为整块 `/dev/vdb` ext4）。  
Isaac 镜像、Docker 层、模型缓存体积很大，**务必在 L3 之前**让 `/data` 可用。

设置 `I4H_COMPSHARE_BOOTSTRAP=1` 时脚本会自动挂载；已手动挂载则跳过。

### `/data` 里到底有什么？（避免「黑盒」）

部署完成后，`/data` 典型布局与体积（**首轮 L6 后合计约 150–250GB**）：

| 路径 | 内容 | 典型体积 | 说明 |
|------|------|----------|------|
| `/data/docker-engine` | Docker 镜像与层（`daemon.json` data-root） | **50–80GB** | `i4h_build:robotic_ultrasound` 镜像标签约 70GB |
| `/data/containerd` | buildkit 构建中间层（**L3 自动迁出系统盘**） | **60–150GB** | 构建时暴涨；构建完可稳定在数十 GB |
| `/data/i4h-workflows` | git 仓库、build、`.cache` | **5–15GB** | Mode B `--use-jump-repo` 从跳板机 rsync |
| `/data/docker/isaac-sim/` | Isaac Sim 运行缓存（kit/ov/pip） | **运行时增长** | 首次跑 full_pipeline 后显著增大 |
| `/data/docker/rti/` | RTI 许可证 | **<1MB** | |
| `/data/cache/huggingface` | PI0 等模型权重 | **5–20GB** | L6 后 `prefetch-pi0-model.sh` 预拉（可 `I4H_PREFETCH_PI0=0` 跳过） |
| `/data/cache/i4h-assets` | 仿真场景资产 | **5–15GB** | 首次跑工作流时下载 |

仍留在**系统盘**（体积小）：apt 包、NVIDIA 驱动、`~/.vnc`、systemd 单元。

`$HOME/docker`、`~/.cache/*` 是指向 `/data` 下目录的**符号链接**（兼容 `./i4h` 默认路径）。

### Docker 镜像能否复用 / 迁移到新机器？

可以，但有条件：

- 镜像实体在 **`/data/docker-engine`** 与 **`/data/containerd`**，不是单独一个文件。
- **同一平台、挂载原数据盘**到新实例（`/data` 完整保留）→ 设 `I4H_SKIP_BUILD=1` 可跳过 L6 重建（省 30–60 分钟）。
- 镜像可能有多个 tag 指向同一层，例如：
  - `i4h_build:robotic_ultrasound`
  - `i4h_build-robotic_ultrasound:dev-xzl`
  - 修复 warp 后 `docker commit` 会产生新 image ID，旧 tag 可能仍指向旧层 — 以 `docker images` 为准，可 `docker image prune` 清理悬空层。
- **新购空盘**无法直接「挂载旧盘」时，只能重跑 L6（有 apt/pip/conda 镜像加速后比首轮快）。

### 一行配置（推荐）

在 `deploy/config/local.env` 中设置：

```bash
I4H_DATA_ROOT=/data
```

脚本会自动把以上内容放到数据盘，并在 `$HOME` 下创建符号链接（详见上文「`/data` 里到底有什么」表格）。

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

## 各层耗时估算（优云智算 4090，首轮参考）

基于 Mode B 实测与日志，**主要耗时在 L6 Docker 构建**：

| 层级 | 典型耗时 | 主要耗时点 |
|------|----------|------------|
| **L0** | 1–2 min | CompShare bootstrap（DNS、挂盘、SSH）；GPU 预检 |
| **L1** | 2–5 min | apt 包（国内镜像）；创建 cache 目录 |
| **L2** | 5–15 min 或跳过 | 驱动已预装则跳过；否则安装 + **可能需重启** |
| **L3** | 3–8 min | Docker + NVIDIA toolkit；**containerd→/data**（避免系统盘满） |
| **L4** | 2–5 min | TigerVNC + XFCE；**systemd 开机自启** |
| **L5** | 2–10 min | rsync 仓库（`--use-jump-repo`）或 git clone；RTI 许可 |
| **L6** | **30–60 min** | 见下表 Docker 构建各 Step |
| **收尾** | 1 min | 改登录密码、`/data` 占用摘要 |
| **首次跑工作流** | 10–15 min | Isaac Sim 冷启动 + HF 模型下载（有缓存后更快） |

**L6 Dockerfile 各 Step（`I4H_BUILD_SHOW_PROGRESS=1` 每 60s 打印磁盘/buildctl 快照）：**

| Step | 内容 | 首轮耗时（国内镜像） | 备注 |
|------|------|---------------------|------|
| 1–7 | apt + 系统依赖 | ~1–3 min | `APT_MIRROR` 国内源；国外 archive 曾卡 20+ min |
| 8 | Miniconda 下载安装 | ~2–5 min | 清华 anaconda 镜像 |
| 9 | `env_setup_robot_us.sh` | **15–25 min** | IsaacSim、IsaacLab、openpi、Holoscan 等 pip/conda |
| 10 | `ml_dtypes` pip | ~1 min | |
| 11–15 | 权限、导出层、unpack | **15–25 min** | containerd 峰值；**必须在 /data** |

开启 `I4H_COMPSHARE_BOOTSTRAP=1` + 多 apt 镜像回退 + pip/conda 镜像后，L6 通常比首轮无优化快 **30–50%**。

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
| L6 构建系统盘满 | 确认 L0 bootstrap 与 L3 `containerd` 已迁到 `/data`；`df -h / /data` |
| apt 慢 / 被封 IP | 改 `I4H_APT_MIRRORS` 顺序或设 `I4H_APT_MIRROR=mirrors.aliyun.com` |
| 构建进度不透明 | L3 安装 buildkit；L6 开启 `I4H_BUILD_SHOW_PROGRESS=1` 看磁盘/buildctl du |
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
