# 一键跑完整流程（在本目录的 PowerShell 中执行）：
#   .\run_all.ps1            完整规模（12–16GB 显卡约 1.5–3 小时，大部分时间花在生成上）
#   .\run_all.ps1 -Quick     快速版（约 20–40 分钟），先确认能跑通、能看到趋势
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

if ($Quick) {
    $prep = @("--n_sft", "600", "--n_dpo", "150", "--n_eval", "100")
    $dpoK = @("--k", "4")
} else {
    $prep = @()
    $dpoK = @()
}

function Step($title, [scriptblock]$cmd) {
    Write-Host "`n========== $title ==========" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$title 失败" }
}

Step "0. 检查 GPU" { uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" }
Step "1. 准备数据" { uv run python prepare_data.py @prep }
Step "2. 评测原模型" { uv run python evaluate.py --name base }
Step "3. SFT 训练" { uv run python train_sft.py }
Step "4. 评测 SFT" { uv run python evaluate.py --name sft --adapters outputs/sft }
Step "5. 采样构造偏好对" { uv run python build_dpo_pairs.py --adapters outputs/sft @dpoK }
Step "6. DPO 训练" { uv run python train_dpo.py }
Step "7. 评测 DPO" { uv run python evaluate.py --name dpo --adapters outputs/sft outputs/dpo }
Step "8. 对比" { uv run python evaluate.py --compare base sft dpo }
