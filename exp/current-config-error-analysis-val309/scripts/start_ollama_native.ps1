param(
    [string]$Model = "qwen3.5:4b-q4_K_M",
    [string]$HostUrl = "http://127.0.0.1:11434",
    [int]$ReadyTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

# Ollama is a native Windows service/daemon. Start it as a separate process;
# the project Python virtualenv is only the API client and does not host Ollama.
$candidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
    (Get-Command ollama.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

if (-not $candidates) {
    throw "Ollama native executable not found. Install Ollama for Windows or add ollama.exe to PATH."
}
$ollamaExe = [string](@($candidates)[0])
$baseUrl = $HostUrl.TrimEnd("/")

try {
    $version = Invoke-RestMethod -Uri "$baseUrl/api/version" -TimeoutSec 2
} catch {
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 500
        try {
            $version = Invoke-RestMethod -Uri "$baseUrl/api/version" -TimeoutSec 2
            break
        } catch {
            if ((Get-Date) -ge $deadline) {
                throw "Ollama did not become ready at $baseUrl within $ReadyTimeoutSeconds seconds. $($_.Exception.Message)"
            }
        }
    } while ((Get-Date) -lt $deadline)
}

$tags = Invoke-RestMethod -Uri "$baseUrl/api/tags" -TimeoutSec 5
$installed = @($tags.models | ForEach-Object { $_.name })
if ($Model -notin $installed) {
    throw "Required local model '$Model' is not installed. Installed tags: $($installed -join ', ')"
}

[pscustomobject]@{
    executable = $ollamaExe
    version = $version.version
    host = $baseUrl
    model = $Model
    server_process = "native Ollama process, outside the Python virtualenv"
} | ConvertTo-Json -Compress
