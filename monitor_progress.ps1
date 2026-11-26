# Track training iteration + auto-download new checkpoints.
# Usage: .\monitor_progress.ps1 -TaskId TASK_xxx -Iterations 48 -SleepSec 300
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 48,
  [int]$SleepSec = 300,
  [string]$OutFile = "progress_$TaskId.log",
  [string]$CkptDir = "cloud_ckpt"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
Set-Content -LiteralPath $OutFile -Value "# progress monitor $TaskId start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8

$seen = @{}
foreach ($f in Get-ChildItem $CkptDir -File -ErrorAction SilentlyContinue) { $seen[$f.Name] = $true }

for ($i = 1; $i -le $Iterations; $i++) {
  Start-Sleep -Seconds $SleepSec
  $ts = Get-Date -Format 'HH:mm:ss'

  $info = & $flux task info --task-id $TaskId 2>&1 | Out-String
  $status = ([regex]::Match($info, '"taskStatus":\s*"([^"]+)"')).Groups[1].Value

  $raw = & $flux task logs --task-id $TaskId 2>&1 | Out-String
  $m = [regex]::Match($raw, '"data":\s*"(.*?)",\s*\n\s*"meta"', 'Singleline')
  $body = ''
  if ($m.Success) {
    $body = $m.Groups[1].Value -replace '\\r\\n', "`n" -replace '\\n', "`n" -replace '\\"', '"' -replace '\\u001b\[[0-9;]*m', ''
  }
  $iter = ([regex]::Match($body, 'Learning iteration (\d+)/(\d+)'))
  $line = if ($iter.Success) { "iter=" + $iter.Groups[1].Value + "/" + $iter.Groups[2].Value } else { "iter=?" }
  $tline = ([regex]::Match($body, 'Iteration time: ([\d.]+)s')).Groups[1].Value
  $eta = ([regex]::Match($body, 'ETA: ([0-9:]+)')).Groups[1].Value
  $errj = ([regex]::Match($body, 'error_joint_pos: ([\d.]+)')).Groups[1].Value

  # checkpoint rows
  $ml = & $flux task model list --task-id $TaskId --page 1 --limit 100 2>&1 | Out-String
  $fns = [regex]::Matches($ml, '"fileName": "([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
  $tss = [regex]::Matches($ml, '"createTime": "([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
  $urls = [regex]::Matches($ml, '"policUrlDown": "([^"]+)"') | ForEach-Object { $_.Groups[1].Value -replace '\\u0026', '&' }
  $newCount = 0
  for ($j = 0; $j -lt $fns.Count; $j++) {
    $tag = $tss[$j].Replace(':', '').Replace(' ', '_').Replace('-', '')
    $name = "$tag__$($fns[$j])"
    if ($seen.ContainsKey($name)) { continue }
    $dest = Join-Path $CkptDir $name
    try {
      Invoke-WebRequest -Uri $urls[$j] -OutFile $dest -TimeoutSec 180 -UseBasicParsing
      if ((Get-Item $dest).Length -gt 1024) {
        $seen[$name] = $true
        $newCount++
        Add-Content -LiteralPath $OutFile -Value ("  + ckpt {0} {1:N2} MB" -f $name, ((Get-Item $dest).Length / 1MB)) -Encoding UTF8
      } else {
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
      }
    } catch { }
  }

  Add-Content -LiteralPath $OutFile -Value ("{0} status={1} {2} it={3}s eta={4} err_joint={5} ckpts={6} new={7}" -f `
    $ts, $status, $line, $tline, $eta, $errj, $fns.Count, $newCount) -Encoding UTF8

  if ($status -eq '6') { Add-Content -LiteralPath $OutFile -Value 'TASK ENDED' -Encoding UTF8; break }
  if ($iter.Success -and $iter.Groups[1].Value -eq $iter.Groups[2].Value) {
    Add-Content -LiteralPath $OutFile -Value 'TARGET ITERATIONS REACHED' -Encoding UTF8
  }
}
Add-Content -LiteralPath $OutFile -Value "# done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
"monitor finished"