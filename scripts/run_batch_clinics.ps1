# 依序跑三個院區的前 7 天 ASR（不併行，避免 GPU 互搶）
# limit 值＝該院區 202509 前 7 個日期資料夾的檔案總數
$dir = Split-Path $PSScriptRoot -Parent   # repo 根目錄
Set-Location $dir

$jobs = @(
    @{ clinic = "A院區"; limit = 1053 },
    @{ clinic = "B院區"; limit = 778 },
    @{ clinic = "C院區"; limit = 751 }
)

foreach ($j in $jobs) {
    $c = $j.clinic
    $n = $j.limit
    Write-Output ("=" * 60)
    Write-Output ("[{0}] 開始 {1}（前 7 天，{2} 檔）" -f (Get-Date -Format "HH:mm:ss"), $c, $n)
    Write-Output ("=" * 60)
    & python run_month.py --month 202509 --clinic $c --stage asr --limit $n 2>&1 | Out-String -Stream
    Write-Output ("[{0}] {1} 結束，exit={2}" -f (Get-Date -Format "HH:mm:ss"), $c, $LASTEXITCODE)
}
Write-Output ("[{0}] 三個院區全部結束" -f (Get-Date -Format "HH:mm:ss"))
