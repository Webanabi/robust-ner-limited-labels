# 把项目上传到 GitHub（Windows）

本项目计划上传到 **Webanabi 账号下的 Public 公开仓库**。两份课程 PDF 含小组/学校信息，已由 .gitignore 排除；CoNLL 英文文本涉及 Reuters 使用条件。仓库只传代码、配置、说明和不含原文的进度摘要。data/raw、data/processed、checkpoints、results、.venv 也已排除。

## 第一次上传

1. 公开空仓库已建立：https://github.com/Webanabi/robust-ner-limited-labels 。本地 main 已提交项目文件并配置 origin。现在打开本项目文件夹的 PowerShell，运行下面命令检查状态并推送：

       git -c "safe.directory=C:/Users/27267/Desktop/ML/Group Project" status --short
       git -c "safe.directory=C:/Users/27267/Desktop/ML/Group Project" push -u origin main

   第一条正常情况下没有输出。这里用 -c 是因为本地 .git 目录由 Codex 沙箱建立，避免你的 Windows 账号遇到 Git 的 dubious ownership 提示；它只对当次命令生效。如果 GitHub 要求登录，按 Git Credential Manager 打开的网页登录流程完成；不要把密码或 token 写入命令或文件。第二条命令将本地提交上传到已创建的公开仓库。
3. 打开 GitHub 仓库页面，检查 README、AGENTS.md、docs/TEAM_START_HERE.md、src/、scripts/、configs/ 是否可见；确认 data/raw/train.txt、data/processed/train.jsonl、checkpoints 下的权重、results 下的逐句预测和两份课程 PDF 均不可见。
4. 公开仓库任何人都能阅读；若两位队友需要提交代码，仍要在仓库 Settings → Collaborators → Add people，按 GitHub 用户名发送邀请。GitHub 官方说明：https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository 。

## 队友如何开始

队友可直接克隆公开仓库；若要推送分支和 PR，需接受协作邀请。运行：

    git clone https://github.com/OWNER/REPO.git
    cd REPO

然后把仓库根目录 AGENTS.md 和 docs/TEAM_START_HERE.md 交给自己的 AI 助手，并说明自己是 Person 2 或 Person 3。仓库里没有语料和权重，队友还需按学校许可通过受控渠道取得 CoNLL 源文件，运行 docs/TEAM_START_HERE.md 的数据准备命令。**只下载仓库文件夹不足以直接训练。**

## 日常协作

Person 2 在 feature/crf-lr 分支做 lr_raw；Person 3 在 feature/distilbert 分支做四种微调；Person 1 管理 main 的公共接口与实验矩阵。每人先 git pull 更新，再在自己的分支修改并提交，通过 Pull Request 合并到 main。PR 里写明代码改动、测试命令、是否已有真实训练、模型/比例/seed 和结果路径。合并后所有人 git pull。

完整训练输出和含原文的错误案例不要通过公开 GitHub 分享。队友可以把 checkpoint、预测 JSONL、日志通过学校批准的私有渠道交给 Person 1；Person 1 核验后将不含原文的汇总数字写入 EXPERIMENT_STATUS.md。12 个运行及 48 项测试完成后再做综合分析。

GitHub 官方本地项目上传说明：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github 。

