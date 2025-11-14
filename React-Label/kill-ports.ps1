$ports = @(3000, 8001)
foreach ($port in $ports) {
  $pids = netstat -ano | findstr ":$port" | ForEach-Object { ($_ -split '\s+')[-1] }
  foreach ($processId in $pids) {
    if ($processId -match '^\d+$') { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }
  }
}