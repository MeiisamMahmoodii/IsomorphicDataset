# IsomorphicDataSet LaTeX Compilation Script for Windows
# This PowerShell script helps compile the LaTeX document to PDF

$ErrorActionPreference = "Stop"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎓 IsomorphicDataSet LaTeX PDF Compiler" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Determine script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$texFile = Join-Path $scriptDir "IsomorphicDataSet_Complete_Report.tex"
$pdfFile = Join-Path $scriptDir "IsomorphicDataSet_Complete_Report.pdf"

# Check if tex file exists
if (-not (Test-Path $texFile)) {
    Write-Host "❌ Error: $texFile not found!" -ForegroundColor Red
    Write-Host "Expected location: $scriptDir" -ForegroundColor Red
    exit 1
}

Write-Host "📄 Found LaTeX file: IsomorphicDataSet_Complete_Report.tex" -ForegroundColor Green
Write-Host "📁 Working directory: $scriptDir" -ForegroundColor Green
Write-Host ""

# Find pdflatex
Write-Host "🔍 Finding pdflatex..." -ForegroundColor Yellow

$pdflatexPaths = @(
    "C:\Program Files\MikeTeX\miktex\bin\x64\pdflatex.exe",
    "C:\Program Files (x86)\MikeTeX\miktex\bin\pdflatex.exe",
    "C:\Program Files\MikeTeX 22.12\miktex\bin\x64\pdflatex.exe",
    "C:\Program Files\MikeTeX 23.1\miktex\bin\x64\pdflatex.exe"
)

$pdflatexCmd = $null

# Check PATH first
try {
    $pdflatexCmd = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($pdflatexCmd) {
        $pdflatexCmd = $pdflatexCmd.Source
    }
}
catch {
    # Continue to check common paths
}

# Check common MikeTeX locations
if (-not $pdflatexCmd) {
    foreach ($path in $pdflatexPaths) {
        if (Test-Path $path) {
            $pdflatexCmd = $path
            break
        }
    }
}

if (-not $pdflatexCmd) {
    Write-Host "❌ Error: pdflatex not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To fix this:" -ForegroundColor Yellow
    Write-Host "  1. Download MikeTeX from: https://miktex.org/download" -ForegroundColor White
    Write-Host "  2. Install MikeTeX" -ForegroundColor White
    Write-Host "  3. During installation, enable 'Install missing packages on-the-fly'" -ForegroundColor White
    Write-Host "  4. Reboot and try again" -ForegroundColor White
    exit 1
}

Write-Host "✅ Found pdflatex: $pdflatexCmd" -ForegroundColor Green
Write-Host ""

# Compile
Write-Host "🔄 Compiling LaTeX to PDF..." -ForegroundColor Cyan
Write-Host "   Pass 1: Generating document structure..." -ForegroundColor White

& $pdflatexCmd -interaction=nonstopmode -output-directory=$scriptDir $texFile > $null 2>&1

Write-Host "   Pass 2: Generating table of contents..." -ForegroundColor White

& $pdflatexCmd -interaction=nonstopmode -output-directory=$scriptDir $texFile > $null 2>&1

# Check result
if (Test-Path $pdfFile) {
    $fileSize = (Get-Item $pdfFile).Length / 1MB
    Write-Host "✅ Success! PDF created: $pdfFile" -ForegroundColor Green
    Write-Host "📊 File size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 Your comprehensive report is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📖 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Open: $pdfFile" -ForegroundColor White
    Write-Host "   2. Read the executive summary (Chapter 1)" -ForegroundColor White
    Write-Host "   3. Follow the 'How to Run' guide (Chapter 6)" -ForegroundColor White
    Write-Host ""

    # Offer to open PDF
    $response = Read-Host "Open PDF now? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Start-Process $pdfFile
    }

    # Offer to clean auxiliary files
    $cleanResponse = Read-Host "Clean up auxiliary LaTeX files? (y/n)"
    if ($cleanResponse -eq "y" -or $cleanResponse -eq "Y") {
        $auxExtensions = @("*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", "*.brf", "*.fls", "*.fdb_latexmk")
        $removed = 0
        foreach ($ext in $auxExtensions) {
            Get-ChildItem $scriptDir -Filter $ext -File | Remove-Item -Force -ErrorAction SilentlyContinue
            $removed += @(Get-ChildItem $scriptDir -Filter $ext -File).Count
        }
        if ($removed -gt 0) {
            Write-Host "🧹 Cleaned up $removed auxiliary files" -ForegroundColor Green
        }
    }

} else {
    Write-Host "❌ Error: PDF was not created!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check that MikeTeX is installed: https://miktex.org/download" -ForegroundColor White
    Write-Host "  2. Open MikeTeX Console and update all packages" -ForegroundColor White
    Write-Host "  3. Try again" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative: Use Overleaf" -ForegroundColor Yellow
    Write-Host "  1. Go to https://www.overleaf.com" -ForegroundColor White
    Write-Host "  2. Create new project → Upload → select $texFile" -ForegroundColor White
    Write-Host "  3. Click Recompile, then Download PDF" -ForegroundColor White
    exit 1
}
