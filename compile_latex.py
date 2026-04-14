#!/usr/bin/env python3
"""
LaTeX Compilation Helper for IsomorphicDataSet Report

This script helps compile the LaTeX document to PDF with proper error handling.
"""

import subprocess
import os
import sys
from pathlib import Path
import shutil

def find_pdflatex():
    """Find pdflatex executable"""
    common_paths = [
        "C:\\Program Files\\MikeTeX\\miktex\\bin\\x64\\pdflatex.exe",
        "C:\\Program Files (x86)\\MikeTeX\\miktex\\bin\\pdflatex.exe",
        "C:\\Program Files\\MikeTeX 22.12\\miktex\\bin\\x64\\pdflatex.exe",
    ]

    # Try in PATH first
    if shutil.which("pdflatex"):
        return shutil.which("pdflatex")

    # Try common MikeTeX locations
    for path in common_paths:
        if os.path.exists(path):
            return path

    return None

def compile_latex(tex_file, output_dir=None):
    """
    Compile LaTeX to PDF

    Args:
        tex_file (str): Path to .tex file
        output_dir (str): Output directory for PDF
    """
    tex_path = Path(tex_file)

    if not tex_path.exists():
        print(f"❌ Error: File not found: {tex_file}")
        return False

    if not tex_path.suffix == ".tex":
        print(f"❌ Error: File must be .tex format")
        return False

    # Find pdflatex
    pdflatex_cmd = find_pdflatex()
    if not pdflatex_cmd:
        print("❌ Error: pdflatex not found!")
        print("\nTo fix:")
        print("  1. Install MikeTeX from: https://miktex.org/download")
        print("  2. During installation, enable 'Install missing packages on-the-fly'")
        print("  3. Ensure C:\\Program Files\\MikeTeX is in your PATH")
        return False

    # Set output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = tex_path.parent

    print(f"📄 Compiling: {tex_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Using: {pdflatex_cmd}\n")

    # First compilation
    print("▶️  First pass...")
    result1 = subprocess.run(
        [
            pdflatex_cmd,
            "-interaction=nonstopmode",
            f"-output-directory={output_dir}",
            str(tex_path)
        ],
        capture_output=True,
        text=True
    )

    if result1.returncode != 0:
        print("❌ First pass failed!")
        print(result1.stdout)
        print(result1.stderr)
        return False

    # Second compilation (for table of contents and references)
    print("▶️  Second pass (generating table of contents)...")
    result2 = subprocess.run(
        [
            pdflatex_cmd,
            "-interaction=nonstopmode",
            f"-output-directory={output_dir}",
            str(tex_path)
        ],
        capture_output=True,
        text=True
    )

    if result2.returncode != 0:
        print("⚠️  Warning: Second pass had issues")
        print(result2.stdout[-500:])  # Last 500 chars

    # Check for output PDF
    pdf_name = tex_path.stem + ".pdf"
    pdf_path = output_dir / pdf_name

    if pdf_path.exists():
        print(f"\n✅ Success! PDF created: {pdf_path}")
        print(f"📊 File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    else:
        print(f"❌ Error: PDF not created at {pdf_path}")
        return False

def cleanup_auxiliary_files(directory):
    """Clean up LaTeX auxiliary files"""
    aux_extensions = ['.aux', '.log', '.out', '.toc', '.bbl', '.blg', '.brf', '.fls', '.fdb_latexmk']

    count = 0
    for ext in aux_extensions:
        for file in Path(directory).glob(f"*{ext}"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file}: {e}")

    if count > 0:
        print(f"🧹 Cleaned up {count} auxiliary files")

def main():
    """Main entry point"""
    print("=" * 60)
    print("🎓 IsomorphicDataSet LaTeX Compilation Helper")
    print("=" * 60 + "\n")

    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    tex_file = script_dir / "IsomorphicDataSet_Complete_Report.tex"

    # Check if file exists
    if not tex_file.exists():
        print(f"❌ Error: {tex_file} not found!")
        print(f"\nExpected location: {script_dir}")
        return False

    print(f"📄 Found: {tex_file.name}\n")

    # Compile
    success = compile_latex(str(tex_file), script_dir)

    if success:
        # Offer to clean up
        print("\n" + "=" * 60)
        response = input("Clean up auxiliary files? (y/n): ").strip().lower()
        if response == 'y':
            cleanup_auxiliary_files(script_dir)

        print("\n✨ Done! Your PDF is ready to use.")
        pdf_path = script_dir / "IsomorphicDataSet_Complete_Report.pdf"
        print(f"\nTo view the PDF:")
        print(f"  • Windows: Double-click {pdf_path.name}")
        print(f"  • Or open with your favorite PDF reader")

        return True
    else:
        print("\n" + "=" * 60)
        print("❌ Compilation failed. Troubleshooting tips:")
        print("\n1. Ensure MikeTeX is installed:")
        print("   Download from https://miktex.org/download")
        print("\n2. Update MikeTeX packages:")
        print("   • Open MikeTeX Console")
        print("   • Click 'Check for updates'")
        print("   • Install all updates")
        print("\n3. Try online compiler (Overleaf):")
        print("   • Go to https://www.overleaf.com")
        print("   • Upload the .tex file")
        print("   • Download the PDF")

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
