# Monitor a flux training task: filtered log tail + real checkpoint detection.
# Usage: .\monitor_task.ps1 -TaskId TASK_xxx -Iterations 60 -SleepSec 180
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 60,
  [int]$SleepSec = 180,
  [string]$OutFile = "monitor_$TaskId.log"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'
Set-Content -LiteralPath $OutFile -Value "# monitor $TaskId start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8

$noise = 'New file detected|Upload queue|Preparing upload|uploaded successfully|Model recovery scan|Detected stable PT|\[SDK\]\[INFO\] 2026.*File scanning'

for ($i = 1; $i -le $Iterations; $i++) {
  Start-Sleep -Seconds $SleepSec
  $ts = Get-Date -Format 'HH:mm:ss'
  $info = & $flux task info --task-id $TaskId 2>&1 | Out-String
  $status = ([regex]::Match($info, '"taskStatus":\s*"([^"]+)"')).Groups[1].Value

  $raw = & $flux task logs --task-id $TaskId 2>&1 | Out-String
  $m = [regex]::Match($raw, '"data":\s*"(.*?)",\s*\n\s*"meta"', 'Singleline')
  $body = ''
  if ($m.Success) { $body = $m.Groups[1].Value -replace '\\r\\n', "`n" -replace '\\n', "`n" }

  $keep = ($body -split "`n") | Where-Object {
    $_ -notmatch $noise -and $_.Trim() -and
    ($_ -match '=== \d|\[FATAL\]|\[launch\]|torch |warp |mjlab \+|Iteration|it/s|Traceback|Error|assert|Mean reward|checkpoint|Saved')
  } | Select-Object -Last 20

  Add-Content -LiteralPath $OutFile -Value "`n===== $ts iter=$i status=$status rawLen=$($body.Length) =====" -Encoding UTF8
  if ($keep) { $keep | ForEach-Object { Add-Content -LiteralPath $OutFile -Value $_.TrimEnd() -Encoding UTF8 } }
  else { Add-Content -LiteralPath $OutFile -Value '(no matching lines yet)' -Encoding UTF8 }

  # Real checkpoint = a .pt model file (ignore venv .pth noise)
  $models = & $flux task model list --task-id $TaskId --page 1 --limit 20 2>&1 | Out-String
  if ($models -match '"fileName":\s*"[^"]*\.pt"') {
    Add-Content -LiteralPath $OutFile -Value "*** CHECKPOINT FOUND ***" -Encoding UTF8
    ($models -split "`n" | Where-Object { $_ -match '"checkpoint"|"fileName".*\.pt' } | Select-Object -First 8) |
      ForEach-Object { Add-Content -LiteralPath $OutFile -Value $_.Trim() -Encoding UTF8 }
    break
  }

  if ($status -eq '6') { Add-Content -LiteralPath $OutFile -Value "TASK ENDED (status=6)" -Encoding UTF8; break }
}
Add-Content -LiteralPath $OutFile -Value "# monitor done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
"monitor finished"