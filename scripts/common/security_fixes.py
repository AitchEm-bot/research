#!/usr/bin/env python3
"""
Security Fixes for C2PA Robustness Testing Pipeline
===================================================

This script applies security fixes to existing scripts by replacing
unsafe subprocess calls with secure alternatives from utils.py.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_unsafe_subprocess_calls(file_path: Path) -> List[Tuple[int, str]]:
    """Find potentially unsafe subprocess calls in a file."""
    unsafe_patterns = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # Check for shell=True (critical security issue)
            if 'shell=True' in line:
                unsafe_patterns.append((i, line.strip()))

            # Check for subprocess calls without validation
            elif re.search(r'subprocess\.(run|call|Popen|check_output)', line):
                # Check if the line contains user input or f-strings
                if 'sys.argv' in line or 'f"' in line or "f'" in line:
                    unsafe_patterns.append((i, line.strip()))
                # Check for string concatenation in commands
                elif '+' in line and 'subprocess' in line:
                    unsafe_patterns.append((i, line.strip()))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return unsafe_patterns


def generate_secure_subprocess_replacement(unsafe_code: str) -> str:
    """Generate secure replacement for unsafe subprocess call."""

    # If it uses shell=True, strongly recommend rewriting
    if 'shell=True' in unsafe_code:
        return "# SECURITY: Replace shell=True with utils.run_command() - shell injection risk!"

    # For other subprocess calls, suggest utils.run_command
    if 'subprocess.run' in unsafe_code:
        return "# Use: result = utils.run_command(cmd, timeout=60)"
    elif 'subprocess.call' in unsafe_code:
        return "# Use: result = utils.run_command(cmd, check=False)"
    elif 'subprocess.Popen' in unsafe_code:
        return "# Use: result = utils.run_command(cmd) for simpler cases"
    elif 'subprocess.check_output' in unsafe_code:
        return "# Use: result = utils.run_command(cmd, capture_output=True)"

    return "# Review this subprocess call for security"


def scan_directory_for_issues(scripts_dir: Path) -> dict:
    """Scan all Python files for security issues."""
    issues = {}

    for py_file in scripts_dir.rglob('*.py'):
        # Skip our utilities and this script
        if 'common' in py_file.parts:
            continue

        unsafe_calls = find_unsafe_subprocess_calls(py_file)
        if unsafe_calls:
            issues[py_file] = unsafe_calls

    return issues


def generate_security_report(scripts_dir: Path):
    """Generate comprehensive security report."""
    print("=" * 80)
    print("SECURITY AUDIT REPORT - C2PA Robustness Testing Pipeline")
    print("=" * 80)
    print()

    issues = scan_directory_for_issues(scripts_dir)

    if not issues:
        print("[OK] No critical security issues found!")
        return

    # Categorize issues
    critical_issues = []  # shell=True
    high_issues = []      # User input in subprocess
    medium_issues = []    # Unvalidated subprocess calls

    for file_path, file_issues in issues.items():
        for line_num, code in file_issues:
            if 'shell=True' in code:
                critical_issues.append((file_path, line_num, code))
            elif 'sys.argv' in code or 'f"' in code or "f'" in code:
                high_issues.append((file_path, line_num, code))
            else:
                medium_issues.append((file_path, line_num, code))

    # Report critical issues
    if critical_issues:
        print("CRITICAL SECURITY ISSUES (shell injection risk):")
        print("-" * 40)
        for file_path, line_num, code in critical_issues:
            rel_path = file_path.relative_to(scripts_dir.parent)
            print(f"  {rel_path}:{line_num}")
            print(f"    Code: {code[:60]}...")
            print(f"    Fix: Remove shell=True and use utils.run_command()")
            print()

    # Report high-priority issues
    if high_issues:
        print("HIGH PRIORITY ISSUES (potential command injection):")
        print("-" * 40)
        for file_path, line_num, code in high_issues[:5]:  # Show first 5
            rel_path = file_path.relative_to(scripts_dir.parent)
            print(f"  {rel_path}:{line_num}")
            print(f"    Code: {code[:60]}...")
            print(f"    Fix: Use utils.run_command() with proper validation")
            print()

        if len(high_issues) > 5:
            print(f"  ... and {len(high_issues) - 5} more high-priority issues")
            print()

    # Summary
    print("SUMMARY:")
    print("-" * 40)
    print(f"  Critical Issues: {len(critical_issues)}")
    print(f"  High Priority: {len(high_issues)}")
    print(f"  Medium Priority: {len(medium_issues)}")
    print()

    print("RECOMMENDED FIXES:")
    print("-" * 40)
    print("  1. Replace all subprocess calls with utils.run_command()")
    print("  2. Never use shell=True in subprocess calls")
    print("  3. Always validate and sanitize command arguments")
    print("  4. Use whitelisted commands only (as implemented in utils.py)")
    print("  5. Set appropriate timeouts for all external commands")
    print()

    # Generate example fix
    print("EXAMPLE SECURE REPLACEMENT:")
    print("-" * 40)
    print("  # BEFORE (unsafe):")
    print("  cmd = f'ffmpeg -i {input_file} -o {output_file}'")
    print("  result = subprocess.run(cmd, shell=True, capture_output=True)")
    print()
    print("  # AFTER (secure):")
    print("  from scripts.common import utils")
    print("  cmd = ['ffmpeg', '-i', str(input_file), '-o', str(output_file)]")
    print("  result = utils.run_command(cmd, timeout=60)")
    print()


def apply_automated_fixes(file_path: Path, dry_run: bool = True):
    """
    Apply automated security fixes to a file.

    Args:
        file_path: Path to Python file to fix
        dry_run: If True, only show what would be changed
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Add import for utils if not present
    if 'import utils' not in content and 'subprocess' in content:
        # Add import after other imports
        import_line = "from scripts.common import utils\n"
        content = re.sub(r'(import subprocess.*\n)', r'\1' + import_line, content, count=1)

    # Replace shell=True patterns
    content = re.sub(
        r'subprocess\.run\([^)]*shell=True[^)]*\)',
        'utils.run_command(cmd)  # FIXED: Removed shell=True for security',
        content
    )

    # Replace basic subprocess.run patterns
    content = re.sub(
        r'subprocess\.run\((\[.*?\])',
        r'utils.run_command(\1',
        content
    )

    if dry_run:
        if content != original_content:
            print(f"Would modify: {file_path.name}")
            # Show first few changes
            import difflib
            diff = difflib.unified_diff(
                original_content.splitlines(keepends=True)[:20],
                content.splitlines(keepends=True)[:20],
                fromfile='original',
                tofile='fixed',
                n=1
            )
            print(''.join(list(diff)[:20]))
    else:
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path.name}")


def main():
    """Main entry point."""
    scripts_dir = Path(__file__).parent.parent

    if len(sys.argv) > 1 and sys.argv[1] == '--fix':
        print("Applying automated security fixes...")
        for py_file in scripts_dir.rglob('*.py'):
            if 'common' not in py_file.parts:
                apply_automated_fixes(py_file, dry_run=False)
    else:
        generate_security_report(scripts_dir)
        print("To apply automated fixes, run: python security_fixes.py --fix")


if __name__ == "__main__":
    main()