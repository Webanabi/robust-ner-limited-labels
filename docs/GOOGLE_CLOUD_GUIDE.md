# Google Cloud Notebook 入门

当前 Person 3 优先使用自己的 RTX 3060 Laptop；本指南仅在本地显存不足或无法稳定完成八次微调时，作为学校 T4 的备用操作说明。不要因为有模板就默认开通云资源。

学校提供的 google_cloud_steps.pdf 指示：使用 CityU Gmail 登录 Google Cloud Console，进入 cs5489-gp-<gid> 项目；在 Notebooks 创建区域为 asia-east2（Hong Kong）的 runtime，模板为 template-n1-standard-4-T4；再建同区域 notebook，使用 Connect -> Connect to Runtime。先核实组号、项目权限、GPU 配额、计费规则和学校的实际界面；不要在未确认授权时创建资源。报告里的“约 16 GB 单 GPU”只是规划假设。

连接后在 Notebook 终端或 SSH 会话运行 nvidia-smi 检查 GPU。SSH 若获允许，可在 Console 的 VM/Workbench 界面取得连接方式；不要猜测实例名或公网 IP。clone 团队 GitHub 仓库，创建虚拟环境并安装 requirements-gpu.txt，先运行 python -m unittest discover -s tests -v。数据集来源和许可需先确认；原始 Reuters 文件仅放受控存储，不提交仓库。

## 实际操作清单

1. 向学校确认组号、cs5489-gp-<gid> 项目可见性、Notebook/VM 创建权限、GPU 配额和区域；核对 template-n1-standard-4-T4 是否在 asia-east2 可选。若学校已建立实例，只连接现有实例。
2. 在学校 Notebook 界面创建或连接 runtime 和 notebook。若提供的是独立 VM，先在 Console 找到 VM 名、zone 与 SSH 按钮。可直接点 SSH；若学校允许 gcloud CLI，命令结构如下，替换全部占位符后才能执行：

       gcloud compute ssh VM_NAME --zone ZONE --project PROJECT_ID

3. 登录后查看 nvidia-smi，再在 Python 中核对 torch.cuda.is_available()。若没有 GPU，先不要运行四种微调；联系课程管理员确认资源。CPU 上仍可运行公共测试与 CRF。
4. 在云端终端克隆团队仓库、建环境和测公共代码：

       git clone GITHUB_REPO_URL
       cd REPO_DIRECTORY
       python3 -m venv .venv
       source .venv/bin/activate
       python -m pip install -r requirements-gpu.txt
       python -m unittest discover -s tests -v

5. 以课程允许的方式将合法原始文件放入 data/raw，运行 python -m scripts.prepare_data。训练命令由 Person 2/3 的实现提供；通过终端会话管理工具保留日志，配置与随机 seed 写入每个运行目录。先试一个小 batch，检查 GPU 显存与耗时，再决定完整运行。
6. 每次运行将 checkpoint 写入 checkpoints/独立目录，并保存 manifest、metadata、配置和日志。把 predictions JSONL、evaluation.json/CSV 与日志下载到本地供 Person 1 汇总。若有允许的 VM SSH，可用 gcloud compute scp 或平台提供的文件下载；命令的项目、zone、远端路径必须以实际实例为准。
7. 最后在 Console 停止不再使用的 runtime/VM，检查磁盘与计费资源。需要继续使用的 checkpoint 可先转移到学校批准的受控存储。

训练时给每个运行单独目录保存配置、日志、checkpoint、预测和 metadata。用终端多路复用器或 Notebook 的持久运行方式避免断线中断；周期性保存 checkpoint。下载轻量的 JSONL 预测、指标和日志供 Person 1 汇总，权重留在受控云存储。传输可用平台下载功能或学校允许的 gcloud scp，具体命令取决于实际项目与实例。

离开前停止或删除不再使用的 runtime/VM，核对账单与配额。学校若只提供预建 runtime，直接连接，不创建新资源。Google 的 Workbench 产品名称和 Console 入口可能更新；以学校项目中实际显示的入口为准。当前官方 Workbench 介绍：https://docs.cloud.google.com/gemini-enterprise-agent-platform/notebooks/workbench/introduction 。

