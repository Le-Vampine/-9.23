# resume_training.ps1 -- resume the suspended training python process
$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name Nt -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern uint NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern uint NtResumeProcess(IntPtr h);
'@
$pidsFile = Join-Path $PSScriptRoot "paused_pids.txt"
if (-not (Test-Path $pidsFile)) { Write-Output "paused_pids.txt not found - nothing to resume"; exit }
$pids = Get-Content $pidsFile | Where-Object { $_ -match '^\d+$' }
$ckptBefore = Get-ChildItem -Path (Join-Path $PSScriptRoot "..\runs") -Recurse -Filter "student_s43.pt" -ErrorAction SilentlyContinue
foreach ($id in $pids) {
    try {
        $h = (Get-Process -Id ([int]$id)).Handle
        $r = [Win32.Nt]::NtResumeProcess($h)
        Write-Output ("RESUMED PID {0}  (ntstatus={1})" -f $id, $r)
    } catch {
        Write-Output ("FAILED to resume PID {0}: {1}" -f $id, $_.Exception.Message)
    }
}
Write-Output ""
Write-Output "=== python processes now ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("PID {0}  CPU(s)={1:N1}  WS={2:N0} MB" -f $_.Id, $_.CPU, ($_.WorkingSet64 / 1MB)) }
Write-Output ""
Write-Output "restart the live monitor with:  code\monitor.cmd 20"
