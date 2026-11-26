# Sample training metrics into a CSV every N seconds (for later reference / plotting).
# Usage: .\sample_metrics.ps1 -TaskId TASK_xxx -Iterations 40 -SleepSec 600
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 40,
  [int]$SleepSec = 600,
  [string]$Csv = "metrics_$TaskId.csv"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'
if (-not (Test-Path $Csv)) {
  "ts,iter,iter_s,reward,err_joint,err_anchor_pos,err_body_pos,err_body_rot,term_time_out,term_motion_end,term_anchor_pos,term_anchor_ori,status,ckpt_total" |
    Set-Content -LiteralPath $Csv -Encoding UTF8
}

function Field($text, $name) {
  $m = [regex]::Match($text, [regex]::Escape($name) + ':\s*(-?[\d.]+)')
  if ($m.Success) { $m.Groups[1].Value } else { '' }
}

for ($i = 1; $i -le $Iterations; $i++) {
  Start-Sleep -Seconds $SleepSec
  $ts = Get-Date -Format 'HH:mm:ss'

  $info = & $flux task info --task-id $TaskId 2>&1 | Out-String
  $status = ([regex]::Match($info, '"taskStatus":\s*"([^"]+)"')).Groups[1].Value

  $raw = & $flux task logs --task-id $TaskId 2>&1 | Out-String
  $m = [regex]::Match($raw, '"data":\s*"(.*?)",\s*\n\s*"meta"', 'Singleline')
  $body = ''
  if ($m.Success) {
    $body = $m.Groups[1].Value -replace '\\r\\n', "`n" -replace '\\n', "`n" -replace '\\u001b\[[0-9;]*m', ''
  }
  $lines = $body -split "`n"
  $lastIdx = ($lines | Select-String -Pattern 'Learning iteration' | Select-Object -Last 1).LineNumber
  $block = if ($lastIdx) { ($lines[($lastIdx - 1)..([Math]::Min($lastIdx + 45, $lines.Count - 1))] -join "`n") } else { '' }

  $iter = ([regex]::Match($block, 'Learning iteration (\d+)/')).Groups[1].Value
  $iterS = Field $block 'Iteration time'
  $reward = Field $block 'Mean reward'
  $errJ = Field $block 'error_joint_pos'
  $errAP = Field $block 'error_anchor_pos'
  $errBP = Field $block 'error_body_pos'
  $errBR = Field $block 'error_body_rot'
  $tTO = Field $block 'Episode_Termination/time_out'
  $tME = Field $block 'Episode_Termination/motion_end'
  $tAP = Field $block 'Episode_Termination/anchor_pos'
  $tAO = Field $block 'Episode_Termination/anchor_ori'

  $ml = & $flux task model list --task-id $TaskId --page 1 --limit 50 2>&1 | Out-String
  $ck = ([regex]::Match($ml, '"total":\s*(\d+)')).Groups[1].Value

  "$ts,$iter,$iterS,$reward,$errJ,$errAP,$errBP,$errBR,$tTO,$tME,$tAP,$tAO,$status,$ck" |
    Add-Content -LiteralPath $Csv -Encoding UTF8

  if ($status -eq '6') { "END,,0,0,,,,,,,,,6,$ck" | Add-Content -LiteralPath $Csv -Encoding UTF8; break }
}
"sampler finished"