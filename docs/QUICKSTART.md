# 🚀 傻瓜式使用教程（零基础 10 分钟上手）

> 目标读者：没接触过 Python / Git 也没关系，跟着一步步做就能跑起来。
> 全程**不需要任何 API Key、不需要联网调用大模型**，克隆下来即可演示全部功能。

---

## 第 0 步：检查电脑上有没有 Python 和 Git

按 `Win + R` 输入 `powershell` 回车（或开始菜单搜 PowerShell），输入：

```bash
python --version
git --version
```

- 显示 `Python 3.10.x` 或更高版本 → 跳过下面的 Python 安装；
- 提示"不是内部或外部命令" → 需要先安装。

### 安装 Python（Windows）

1. 打开 <https://www.python.org/downloads/> 点黄色按钮下载最新版；
2. 双击安装，**务必勾选最下面的 `Add python.exe to PATH`**；
3. 一路下一步，完成后**关掉重新打开** PowerShell，再输入 `python --version` 确认。

### 安装 Git（Windows）

打开 <https://git-scm.com/downloads> 下载安装，全部默认"下一步"即可。

---

## 第 1 步：把项目下载到电脑

```bash
git clone https://github.com/Artlatte/cortex-agent.git
cd cortex-agent
```

> 不想用 Git？也可以到仓库主页点绿色 `Code` 按钮 → `Download ZIP`，解压后进入
> `cortex-agent` 文件夹，在文件夹**地址栏**输入 `powershell` 回车，即可在当前目录打开命令行。

---

## 第 2 步：一键安装

在 `cortex-agent` 文件夹里执行（复制粘贴这两行，回车）：

```bash
python -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
```

看到最后一行 `Successfully installed ...` 就是成功了 ✅

> pip 下载慢/超时？加清华镜像重试：
>
> ```bash
> .\.venv\Scripts\pip.exe install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 第 3 步：跑三个演示（重点！）

安装成功后，`cortex` 命令可以直接用：

```bash
.\.venv\Scripts\cortex.exe demo agent     # 演示 1：ReAct Agent 工具调用
.\.venv\Scripts\cortex.exe demo rag       # 演示 2：RAG 知识库问答
.\.venv\Scripts\cortex.exe demo multi     # 演示 3：多 Agent 协作编排
```

三个演示分别展示：

| 命令 | 演示什么 | 成功标志 |
| --- | --- | --- |
| `demo agent` | AI 自动调用"计算器 / 时钟"工具回答问题 | 看到 `最终答案:` |
| `demo rag` | 文档入库 → 混合检索 → 带来源引用回答 | 看到 `入库完成: 5 个文件` 和检索结果 |
| `demo multi` | 规划 → 并行研究 → 核查 → 成文 | 看到 `◆ 最终答案` |

---

## 第 4 步：启动网页服务

```bash
.\.venv\Scripts\cortex.exe serve
```

看到 `Uvicorn running on http://127.0.0.1:8000` 后：

1. 浏览器打开 **http://127.0.0.1:8000/docs** —— 自动生成的接口文档，点任意接口 → `Try it out` → `Execute` 就能直接测试；
2. 或者用 PowerShell 直接调接口：

```powershell
# 问一个问题
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat -Method Post -ContentType "application/json" -Body '{"question":"请计算 2+2"}'

# 查看监控指标
Invoke-RestMethod -Uri http://127.0.0.1:8000/metrics
```

---

## 第 5 步（可选）：接入真实大模型

以 DeepSeek 为例（便宜好用，约 1 元/百万 token）：

1. 到 <https://platform.deepseek.com> 注册 → 充值 10 元 → 创建 API Key（`sk-` 开头，保存好）；
2. 把 `examples/config.example.json` 复制一份，改名为 `config.json`（放在项目根目录）；
3. PowerShell 里设置密钥并启动：

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的密钥"
$env:CORTEX_CONFIG = "config.json"
.\.venv\Scripts\cortex.exe serve
```

此时 `/v1/chat`、Agent 等接口就是真实大模型在回答了。
换 OpenAI / Claude / Gemini：编辑 `config.json`，把 `default_provider` 改成对应名字，并设置对应环境变量（`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`）即可。

---

## 常见问题（遇到先看这里）

| 现象 | 解决办法 |
| --- | --- |
| `python` 不是内部或外部命令 | 重装 Python 时勾选 `Add python.exe to PATH`，然后重启终端 |
| pip 安装超时 / 太慢 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 镜像 |
| 端口 8000 被占用 | `cortex serve --port 9000` 换端口 |
| 中文显示乱码 | 用 Windows Terminal 或 VS Code 的终端运行 |
| 想用自己的文档做知识库 | `cortex rag ingest "你的文件夹路径"` 入库，然后 `cortex rag search "你的问题"` |
| 公司网络有代理 | 设 `$env:HTTPS_PROXY = "http://代理地址:端口"` 后再 pip install |

---

更多细节见 [README](../README.md)、[架构文档](ARCHITECTURE.md) 与 [需求分析与技术方案](DESIGN.md)。
