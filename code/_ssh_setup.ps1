# Generate a GitHub-dedicated SSH key pair, write ~/.ssh/config, register with ssh-agent
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\code\_ssh_setup.ps1
# NOTE: kept ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI/GBK).

$ErrorActionPreference = 'Continue'
$sshDir = Join-Path $env:USERPROFILE '.ssh'
$newKey = Join-Path $sshDir 'id_ed25519_2'
$email  = '971849435@qq.com'

if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Force -Path $sshDir | Out-Null }

Write-Output '=== [1/5] generate new key pair ==='
if (Test-Path $newKey) {
    Write-Output "already exists, skip: $newKey"
} else {
    # invoke via cmd so that -N "" is parsed as an empty passphrase
    $cmdLine = 'ssh-keygen -t ed25519 -C "' + $email + '" -f "' + $newKey + '" -N ""'
    cmd /c $cmdLine
}

Write-Output ''
Write-Output '=== [2/5] write ~/.ssh/config ==='
$configPath = Join-Path $sshDir 'config'
$block = @(
    'Host github.com',
    '    HostName github.com',
    '    User git',
    '    IdentityFile ~/.ssh/id_ed25519_2',
    '    IdentitiesOnly yes'
)
$existing = if (Test-Path $configPath) { [string](Get-Content $configPath -Raw) } else { '' }
if ($existing -match 'id_ed25519_2') {
    Write-Output 'config already contains id_ed25519_2, skip.'
} else {
    $new = $existing.TrimEnd()
    if ($new) { $new += "`r`n" }
    $new += ($block -join "`r`n") + "`r`n"
    Set-Content -Path $configPath -Value $new -Encoding ASCII -NoNewline
    Write-Output "written: $configPath"
}
Write-Output '--- config content ---'
Get-Content $configPath | ForEach-Object { '    ' + $_ }

Write-Output ''
Write-Output '=== [3/5] start ssh-agent ==='
try {
    Set-Service ssh-agent -StartupType Automatic -ErrorAction Stop
    Start-Service ssh-agent -ErrorAction Stop
    $svc = Get-Service ssh-agent
    Write-Output "ssh-agent state: $($svc.Status) / $($svc.StartType)"
} catch {
    Write-Output "start failed (admin rights required?): $($_.Exception.Message)"
}

Write-Output ''
Write-Output '=== [4/5] add key to agent ==='
ssh-add $newKey 2>&1
ssh-add -l 2>&1

Write-Output ''
Write-Output '=== [5/5] public key ==='
$pub = Get-Content "$newKey.pub" -Raw
Write-Output $pub
try { $pub | Set-Clipboard; Write-Output '[copied to clipboard]' } catch { Write-Output "[clipboard failed] $($_.Exception.Message)" }

Write-Output ''
Write-Output '=== fingerprint ==='
ssh-keygen -l -f "$newKey.pub" 2>&1
