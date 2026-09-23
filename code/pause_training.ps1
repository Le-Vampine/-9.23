# pause_training.ps1 -- suspend the training python process (reversible, keeps progress in memory)
$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name Nt -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern uint NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern uint NtResumeProcess(IntPtr h);
'@
$root = Split-Path -Parent $PSScriptRoot

Write-Output "=== current time ==="
Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Output ""
Write-Output "=== python processes ==="
$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'"
if (-not $procs) { Write-Output "(none)" }
foreach ($p in $procs) {
    $cl = ($p.CommandLine -replace '\s+', ' ')
    if ($cl.Length -gt 200) { $cl = $cl.Substring(0, 200) }
    Write-Output ("PID {0}  started {1}" -f $p.ProcessId, $p.CreationDate)
    Write-Output ("    " + $cl)
}

Write-Output ""
Write-Output "=== training log tail ==="
$logs = Get-ChildItem -Path $PSScriptRoot -Filter "_train*.log" | Sort-Object LastWriteTime -Descending
if ($logs) {
    $lg = $logs[0]
    Write-Output ("file: " + $lg.Name + "   last write: " + $lg.LastWriteTime)
    Get-Content $lg.FullName | Select-String -Pattern "ep[0-9]+|SEED|best valid|early|TH\]|VALID|TEST|SAVE|ALL_TRAINING" | Select-Object -Last 10 | ForEach-Object { Write-Output ("    " + $_.Line) }
}

Write-Output ""
Write-Output "=== checkpoints in runs dir ==="
foreach ($d in @("q2", "q2v3", "v3check")) {
    $dir = Join-Path $root ("runs\" + $d)
    if (Test-Path $dir) {
        $fs = Get-ChildItem -Path $dir -Filter "student_*.pt" -ErrorAction SilentlyContinue
        if ($fs) { foreach ($f in $fs) { Write-Output ("  {0}\{1}  {2:N2} MB" -f $d, $f.Name, ($f.Length / 1MB)) } }
        else { Write-Output ("  {0}\  (no checkpoint yet)" -f $d) }
    }
}

Write-Output ""
Write-Output "=== SUSPEND ==="
$train = @($procs | Where-Object { $_.CommandLine -like '*train.py*' })
$mon = @($procs | Where-Object { $_.CommandLine -like '*monitor_progress.py*' })
$paused = @()
foreach ($t in $train) {
    try {
        $h = (Get-Process -Id $t.ProcessId).Handle
        $r = [Win32.Nt]::NtSuspendProcess($h)
        Write-Output ("SUSPENDED train PID {0}  (ntstatus={1})" -f $t.ProcessId, $r)
        $paused += $t.ProcessId
    } catch {
        Write-Output ("FAILED to suspend PID {0}: {1}" -f $t.ProcessId, $_.Exception.Message)
    }
}
foreach ($m in $mon) {
    Stop-Process -Id $m.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Output ("STOPPED monitor PID {0}" -f $m.ProcessId)
}
if ($paused.Count -eq 0) { Write-Output "no training process found to suspend" }
$pidsFile = Join-Path $PSScriptRoot "paused_pids.txt"
$paused | Set-Content -Path $pidsFile -Encoding ascii
Write-Output ("saved paused pids -> " + $pidsFile)

Write-Output ""
Write-Output "=== after suspend (CPU should stop growing) ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("PID {0}  CPU(s)={1:N1}  WS={2:N0} MB" -f $_.Id, $_.CPU, ($_.WorkingSet64 / 1MB)) }
