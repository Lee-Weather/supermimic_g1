# Monitor the running flux task: log tail + checkpoint detection.
# Usage: powershell -File monitor_task.ps1 -TaskId TASK_xxx -Iterations 60 -SleepSec 180
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 60,
  [int]$SleepSec = 180,
  [string]$OutFile = "monitor_$TaskId.log"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'
Set-Content -LiteralPath $OutFile -Value "# monitor $TaskId start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8

for ($i = 1; $i -le $Iterations; $i++) {
  Start-Sleep -Seconds $SleepSec
  $ts = Get-Date -Format 'HH:mm:ss'
  $info = & $flux task info --task-id $TaskId 2>&1 | Out-String
  $status = ([regex]::Match($info, '"taskStatus":\s*"([^"]+)"')).Groups[1].Value
  $url = ([regex]::Match($info, '"argoLogDownloadUrl":\s*"([^"]+)"')).Groups[1].Value -replace '\\u0026', '&'
  $runtime = ([regex]::Match($info, '"runtime":\s*"([^"]+)"')).Groups[1].Value

  $body = ''
  if ($url) {
    try { $body = (Invoke-WebRequest -Uri $url -TimeoutSec 25 -UseBasicParsing).Content } catch { }
  }
  $len = $body.Length

  # interesting lines only: bootstrap markers, torch/warp versions, it/s, errors, checkpoints
  $keep = ($body -split "`n") | Where-Object {
    $_ -match 'launch\]|=== [0-9]+\.|torch |warp |mjlab \+|\[FATAL\]|Traceback|Error|it/s|Iteration|checkpoint|Saved|save_interval|Model recovery|train_exit_code|status updated'
  } | Select-Object -Last 18

  Add-Content -LiteralPath $OutFile -Value "`n===== $ts iter=$i status=$status runtime=${runtime}s logLen=$len =====" -Encoding UTF8
  if ($keep) { $keep | ForEach-Object { Add-Content -LiteralPath $OutFile -Value $_.TrimEnd() -Encoding UTF8 } }
  else { Add-Content -LiteralPath $OutFile -Value '(no matching lines yet)' -Encoding UTF8 }

  $models = & $flux task model list --task-id $TaskId --page 1 --limit 5 2>&1 | Out-String
  $cnt = ([regex]::Matches($models, '"checkpoint"')).Count
  if ($cnt -gt 0) {
    Add-Content -LiteralPath $OutFile -Value "CHECKPOINTS FOUND: $cnt" -Encoding UTF8
    ($models -split "`n" | Where-Object { $_ -match 'checkpoint|fileName|modelName' } | Select-Object -First 6) |
      ForEach-Object { Add-Content -LiteralPath $OutFile -Value $_.Trim() -Encoding UTF8 }
    if ($cnt -gt 0) { break }
  }

  if ($status -eq '6') { Add-Content -LiteralPath $OutFile -Value "TASK ENDED" -Encoding UTF8; break }
}
Add-Content -LiteralPath $OutFile -Value "# monitor done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
"monitor finished"