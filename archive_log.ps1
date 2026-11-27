# Archive the platform's rolling log tail to a local file, so nothing rotates away.
# Usage: .\archive_log.ps1 -TaskId TASK_xxx -Iterations 120 -SleepSec 300
param(
  [Parameter(Mandatory = $true)][string]$TaskId,
  [int]$Iterations = 120,
  [int]$SleepSec = 300,
  [string]$OutFile = "logs_archive_$TaskId.txt"
)

$flux = 'C:\Users\cuizeyu\AppData\Roaming\npm\flux.cmd'
Set-Location 'E:\G1\G1_supermimic'

$seen = [System.Collections.Generic.HashSet[string]]::new()
if (Test-Path $OutFile) {
  foreach ($l in Get-Content $OutFile -Tail 400) { [void]$seen.Add($l) }
}
$noise = 'New file detected|Upload queue|Preparing upload|uploaded successfully|Model recovery scan|Detected stable PT|solver iterations limit|linesearch iterations limit|warn_overflow|Obtained python dependency|getting pip list'

for ($i = 1; $i -le $Iterations; $i++) {
  $raw = & $flux task logs --task-id $TaskId --raw --no-request-log 2>&1 | Out-String
  if ($raw) {
    $new = @()
    foreach ($line in ($raw -split "`n")) {
      $l = $line -replace '\\r$', '' -replace '\\u001b\[[0-9;]*m', ''
      if ([string]::IsNullOrWhiteSpace($l)) { continue }
      if ($l -match $noise) { continue }
      if ($seen.Add($l)) { $new += $l }
    }
    if ($new.Count -gt 0) {
      Add-Content -LiteralPath $OutFile -Value $new -Encoding UTF8
      Add-Content -LiteralPath $OutFile -Value ("--- appended {0} lines at {1} ---" -f $new.Count, (Get-Date -Format 'HH:mm:ss')) -Encoding UTF8
    }
  }
  if ($i -lt $Iterations) { Start-Sleep -Seconds $SleepSec }
}
"archive finished"