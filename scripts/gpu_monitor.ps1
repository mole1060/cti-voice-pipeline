# 每 10 秒取樣一次 GPU 溫度／使用率／功耗，寫成 CSV。
# 長批次跑之前先開著，事後才有東西可以回頭對「當機是不是熱或功耗造成的」。
param(
    [string]$LogPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "gpu_monitor.csv")
)
$logPath = $LogPath
if (-not (Test-Path $logPath)) {
    "timestamp,name,temp_c,util_pct,power_w,power_limit_w,mem_used_mib,mem_total_mib,clock_mhz,fan_pct" | Out-File -FilePath $logPath -Encoding utf8
}
while ($true) {
    $line = nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,clocks.sm,fan.speed --format=csv,noheader,nounits
    Add-Content -Path $logPath -Value $line -Encoding utf8
    Start-Sleep -Seconds 10
}
