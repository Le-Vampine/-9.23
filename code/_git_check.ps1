# Quick env check: git install + ssh key/config state (ASCII only on purpose)
$git = 'C:\Program Files\Git\cmd\git.exe'
if (Test-Path $git) {
    Write-Output ('GIT OK: ' + (& $git --version))
} else {
    Write-Output 'GIT NOT READY (winget still running?)'
}

Write-Output '--- .ssh files ---'
Get-ChildItem (Join-Path $env:USERPROFILE '.ssh') | Select-Object -ExpandProperty Name

Write-Output '--- ssh-agent ---'
Get-Service ssh-agent | Select-Object Status, StartType | Format-Table -AutoSize

Write-Output '--- github.com auth test ---'
ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -T git@github.com 2>&1
