# Watch a replay task: report progress, catch tracebacks, download *_mp4.pt artifacts.
# Usage: .\monitor_replay.ps1 -TaskId TASK_xxx -Iterations 60 -SleepSec 60
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 60,
  [int]$SleepSec = 60,
  [string]$OutDir = "replay_out",
  [string]$LogFile = "monitor_replay_$TaskId.log"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Set-Content -LiteralPath $LogFile -Value "# replay monitor $TaskId start $(Get-Date -Format 'HH:mm:ss')" -Encoding UTF8

$seen = @{}
$lastStage = ''
for ($i = 1; $i -le $Iterations; $i++) {
  Start-Sleep -Seconds $SleepSec
  $ts = Get-Date -Format 'HH:mm:ss'

  $info = & $flux task info --task-id $TaskId 2>&1 | Out-String
  $status = ([regex]::Match($info, '"taskStatus":\s*"([^"]+)"')).Groups[1].Value

  $raw = & $flux task logs --task-id $TaskId --raw --no-request-log 2>&1 | Out-String
  $clean = $raw -replace '\x1b\[[0-9;]*m', '' -replace '\r', ''
  $lines = ($clean -split "`n") | Where-Object { $_ -match '\S' -and $_ -notmatch 'New file detected|Upload queue|Preparing upload|uploaded successfully|Model recovery scan|Detected stable PT|solver iterations|linesearch|warn_overflow|limit reached - please increase|Downloading|Downloaded' }
  $tail = ($lines | Select-Object -Last 6) -join ' | '

  $stage = if ($tail -match '\[replay\]') { 'replay' } elseif ($lines -match '=== [0-9]') { 'bootstrap' } else { '?' }
  $tb = ($lines | Where-Object { $_ -match 'Traceback|^\w*Error:|\[FATAL\]' } | Select-Object -Last 4) -join ' / '

  Add-Content -LiteralPath $LogFile -Value ("{0} status={1} stage={2} | {3}" -f $ts, $status, $stage, $tail) -Encoding UTF8
  if ($tb) { Add-Content -LiteralPath $LogFile -Value "    !! $tb" -Encoding UTF8 }

  # replay 段结果行
  ($lines | Where-Object { $_ -match '\[replay\] .*mean_speed' }) | ForEach-Object {
    if (-not $seen.ContainsKey($_)) { $seen[$_] = $true; Add-Content -LiteralPath $LogFile -Value "    RESULT $_" -Encoding UTF8 }
  }

  # 下载 *_mp4.pt 产物
  $ml = & $flux task model list --task-id $TaskId --page 1 --limit 200 2>&1 | Out-String
  $fns = [regex]::Matches($ml, '"fileName": "([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
  $urls = [regex]::Matches($ml, '"policUrlDown": "([^"]+)"') | ForEach-Object { $_.Groups[1].Value -replace '\\u0026', '&' }
  for ($j = 0; $j -lt $fns.Count; $j++) {
    if ($fns[$j] -notmatch '_mp4\.pt$') { continue }
    $dest = Join-Path $OutDir $fns[$j]
    if (Test-Path $dest) { continue }
    try {
      Invoke-WebRequest -Uri $urls[$j] -OutFile $dest -TimeoutSec 300 -UseBasicParsing
      $size = (Get-Item $dest).Length
      Add-Content -LiteralPath $LogFile -Value ("    GOT {0} {1:N2} MB" -f $fns[$j], ($size / 1MB)) -Encoding UTF8
    } catch { Add-Content -LiteralPath $LogFile -Value "    DLFAIL $($fns[$j])" -Encoding UTF8 }
  }

  if ($status -eq '6' -or $status -eq '5') {
    Add-Content -LiteralPath $LogFile -Value "TASK ENDED status=$status" -Encoding UTF8
    break
  }
}
Add-Content -LiteralPath $LogFile -Value "# monitor done $(Get-Date -Format 'HH:mm:ss')" -Encoding UTF8
"monitor finished"