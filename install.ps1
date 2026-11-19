$ErrorActionPreference = "Stop"

$ProjectDirectory = $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectDirectory ".venv"
$PythonCandidates = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCandidates += @{ Command = "py"; Prefix = @("-3.11") }
}
$PythonCandidates += @{ Command = "python3.11"; Prefix = @() }
$PythonCandidates += @{ Command = "python"; Prefix = @() }

$PythonCommand = $null
$PythonPrefix = @()
foreach ($candidate in $PythonCandidates) {
    $commandInfo = Get-Command $candidate.Command -ErrorAction SilentlyContinue
    if (-not $commandInfo) {
        continue
    }

    $candidatePrefix = $candidate.Prefix
    $candidateExecutable = $commandInfo.Source
    $version = & $candidateExecutable @candidatePrefix -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.11") {
        $PythonCommand = $commandInfo.Source
        $PythonPrefix = $candidatePrefix
        break
    }
}

if (-not $PythonCommand) {
    throw "Python 3.11 is required. Install Python 3.11 and rerun this script. The Python Launcher (py) is supported."
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$CommandArgs
    )

    & $FilePath @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: ${FilePath} $($CommandArgs -join ' ')"
    }
}

Write-Host "Creating the virtual environment with Python 3.11..."
$venvCommandArgs = @($PythonPrefix) + @("-m", "venv", $VirtualEnvironment)
Invoke-CheckedCommand -FilePath $PythonCommand -CommandArgs $venvCommandArgs

$VenvPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
Invoke-CheckedCommand -FilePath $VenvPython -CommandArgs @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedCommand -FilePath $VenvPython -CommandArgs @("-m", "pip", "install", "-r", (Join-Path $ProjectDirectory "requirements-lock.txt"))

$EnvironmentFile = Join-Path $ProjectDirectory ".env"
if (-not (Test-Path $EnvironmentFile)) {
    Copy-Item (Join-Path $ProjectDirectory ".env.example") $EnvironmentFile
    Write-Host "Created .env from .env.example. Add an OpenAI key only if you plan to use cloud routes."
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDirectory "documents") | Out-Null

$OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if ($OllamaCommand) {
    & $OllamaCommand.Source list *> $null
    if ($LASTEXITCODE -eq 0) {
        $answer = Read-Host "Download qwen2.5:3b and nomic-embed-text now? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($answer) -or $answer -match "^(y|yes)$") {
            Invoke-CheckedCommand -FilePath $OllamaCommand.Source -CommandArgs @("pull", "qwen2.5:3b")
            Invoke-CheckedCommand -FilePath $OllamaCommand.Source -CommandArgs @("pull", "nomic-embed-text")
        }
        else {
            Write-Host "Download the models later with: ollama pull qwen2.5:3b; ollama pull nomic-embed-text"
        }
    }
    else {
        Write-Host "Ollama is installed but its service is not responding. Start Ollama, then pull qwen2.5:3b and nomic-embed-text."
    }
}
else {
    Write-Host "Ollama was not found. Install it from https://ollama.com/download, then pull qwen2.5:3b and nomic-embed-text."
}

Write-Host ""
Write-Host "Setup is complete. Add documents under $ProjectDirectory\documents and start AI Router with:"
Write-Host "  $VenvPython $ProjectDirectory\run_app.py"
