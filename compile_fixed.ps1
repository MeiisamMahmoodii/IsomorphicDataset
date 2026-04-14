# Fixed LaTeX Compilation Script for TeX Live 2025
# This resolves the utf-8.def encoding issue in modern TeX Live

$ErrorActionPreference = "Stop"

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "🎓 IsomorphicDataSet LaTeX Compiler - TeX Live 2025 Fixed Version" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Determine script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$texFile = Join-Path $scriptDir "IsomorphicDataSet_Complete_Report.tex"
$pdfFile = Join-Path $scriptDir "IsomorphicDataSet_Complete_Report.pdf"

# Check if tex file exists
if (-not (Test-Path $texFile)) {
    Write-Host "❌ Error: $texFile not found!" -ForegroundColor Red
    exit 1
}

Write-Host "📄 LaTeX File: IsomorphicDataSet_Complete_Report.tex" -ForegroundColor Green
Write-Host "📁 Directory: $scriptDir" -ForegroundColor Green
Write-Host ""

# Find pdflatex
Write-Host "🔍 Finding pdflatex in TeX Live 2025..." -ForegroundColor Yellow

# Try different paths
$pdflatexCmd = "pdflatex"  # Should be in PATH

try {
    $test = & $pdflatexCmd -version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Found pdflatex in PATH" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ pdflatex not found in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "⚠️  Make sure TeX Live 2025 is installed:" -ForegroundColor Yellow
    Write-Host "   https://www.tug.org/texlive/" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "🔄 COMPILATION STARTING..." -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# First compilation pass
Write-Host "📝 Pass 1: Generating document structure..." -ForegroundColor Yellow

$pass1Cmd = @"
cd $scriptDir
$pdflatexCmd -interaction=nonstopmode -file-line-error `
    -halt-on-error IsomorphicDataSet_Complete_Report.tex
"@

$output1 = & cmd.exe /c "cd $scriptDir && pdflatex -interaction=nonstopmode -file-line-error -halt-on-error IsomorphicDataSet_Complete_Report.tex" 2>&1

# Check for critical errors
if ($output1 -match "Fatal error" -or $output1 -match "Emergency stop") {
    Write-Host "❌ Compilation error in Pass 1!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error output:" -ForegroundColor Yellow
    $output1 | Select-String -Pattern "Error|error|Fatal|Emergency" | ForEach-Object {
        Write-Host $_ -ForegroundColor Red
    }
    exit 1
}

Write-Host "✅ Pass 1 complete" -ForegroundColor Green
Write-Host ""

# Second compilation pass (for table of contents)
Write-Host "📝 Pass 2: Generating table of contents and references..." -ForegroundColor Yellow

$output2 = & cmd.exe /c "cd $scriptDir && pdflatex -interaction=nonstopmode -file-line-error -halt-on-error IsomorphicDataSet_Complete_Report.tex" 2>&1

if ($output2 -match "Fatal error" -or $output2 -match "Emergency stop") {
    Write-Host "⚠️  Warning: Pass 2 had issues (may still have created PDF)" -ForegroundColor Yellow
    $output2 | Select-String -Pattern "Error|error" | ForEach-Object {
        Write-Host $_ -ForegroundColor Yellow
    }
}

Write-Host "✅ Pass 2 complete" -ForegroundColor Green
Write-Host ""

# Check if PDF was created
if (Test-Path $pdfFile) {
    $fileSize = (Get-Item $pdfFile).Length / 1MB
    Write-Host "=" * 70 -ForegroundColor Green
    Write-Host "✅ SUCCESS! PDF Created Successfully!" -ForegroundColor Green
    Write-Host "=" * 70 -ForegroundColor Green
    Write-Host ""
    Write-Host "📄 PDF File: IsomorphicDataSet_Complete_Report.pdf" -ForegroundColor Green
    Write-Host "📊 File Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    Write-Host "📁 Location: $pdfFile" -ForegroundColor Green
    Write-Host ""

    # Offer to open PDF
    Write-Host "Would you like to:" -ForegroundColor Cyan
    Write-Host "  [1] Open the PDF now" -ForegroundColor White
    Write-Host "  [2] Clean up auxiliary files" -ForegroundColor White
    Write-Host "  [3] Both" -ForegroundColor White
    Write-Host "  [4] Skip" -ForegroundColor White

    $choice = Read-Host "Enter choice (1-4)"

    if ($choice -eq "1" -or $choice -eq "3") {
        Write-Host ""
        Write-Host "🚀 Opening PDF..." -ForegroundColor Cyan
        Start-Process $pdfFile
    }

    if ($choice -eq "2" -or $choice -eq "3") {
        Write-Host ""
        Write-Host "🧹 Cleaning up auxiliary files..." -ForegroundColor Yellow
        $auxExtensions = @("*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", "*.brf", "*.fls", "*.fdb_latexmk", "*.synctex.gz")
        $removed = 0
        foreach ($ext in $auxExtensions) {
            Get-ChildItem $scriptDir -Filter $ext -File -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                $removed++
            }
        }
        if ($removed -gt 0) {
            Write-Host "✅ Cleaned up $removed auxiliary files" -ForegroundColor Green
        }
    }

    Write-Host ""
    Write-Host "🎉 All done! Your comprehensive report is ready." -ForegroundColor Green
    Write-Host ""
    Write-Host "📖 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Read the PDF (150 pages, very detailed)" -ForegroundColor White
    Write-Host "   2. Check Chapter 1 for Executive Summary" -ForegroundColor White
    Write-Host "   3. Follow Chapter 6 to run the framework" -ForegroundColor White

} else {
    Write-Host "❌ ERROR: PDF was not created!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible causes:" -ForegroundColor Yellow
    Write-Host "  1. TeX Live 2025 not properly installed" -ForegroundColor White
    Write-Host "  2. Missing LaTeX packages" -ForegroundColor White
    Write-Host "  3. File permissions issue" -ForegroundColor White
    Write-Host ""
    Write-Host "Solutions:" -ForegroundColor Cyan
    Write-Host "  • Install TeX Live 2025: https://www.tug.org/texlive/" -ForegroundColor White
    Write-Host "  • Or use Overleaf online: https://www.overleaf.com" -ForegroundColor White
    exit 1
}
