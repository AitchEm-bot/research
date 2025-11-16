#!/usr/bin/env python3
"""
Refactoring Report Generator
============================

Analyzes the scripts directory and generates a report of refactoring opportunities.
"""

import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def analyze_imports(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.add(f"{module}.{alias.name}")
                    else:
                        imports.add(alias.name)
    except (SyntaxError, UnicodeDecodeError):
        pass  # Skip files with syntax errors
    return imports


def find_duplicate_functions(directory: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """Find duplicate or similar function definitions."""
    function_patterns = defaultdict(list)

    for py_file in directory.rglob('*.py'):
        if 'common' in py_file.parts:  # Skip our utils module
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find function definitions
            func_pattern = re.compile(r'^def\s+(\w+)\s*\([^)]*\):', re.MULTILINE)
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                function_patterns[func_name].append((py_file, line_num))
        except (UnicodeDecodeError, IOError):
            pass

    # Filter to show only duplicated functions
    duplicates = {name: locations for name, locations in function_patterns.items()
                  if len(locations) > 1}
    return duplicates


def find_subprocess_usage(directory: Path) -> List[Tuple[Path, int, str]]:
    """Find all subprocess usage that should use safe wrappers."""
    subprocess_calls = []

    for py_file in directory.rglob('*.py'):
        if 'common' in py_file.parts:  # Skip our utils module
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if 'subprocess.' in line and any(cmd in line for cmd in ['run', 'call', 'Popen', 'check_output']):
                    subprocess_calls.append((py_file, i, line.strip()))
        except (UnicodeDecodeError, IOError):
            pass

    return subprocess_calls


def find_csv_operations(directory: Path) -> List[Tuple[Path, int, str]]:
    """Find CSV operations that could use shared utilities."""
    csv_ops = []

    for py_file in directory.rglob('*.py'):
        if 'common' in py_file.parts:
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if 'csv.DictWriter' in line or 'writeheader' in line or 'writerow' in line:
                    csv_ops.append((py_file, i, line.strip()))
        except (UnicodeDecodeError, IOError):
            pass

    return csv_ops


def find_logging_setup(directory: Path) -> List[Tuple[Path, int]]:
    """Find logging.basicConfig calls that should use shared setup."""
    logging_setups = []

    for py_file in directory.rglob('*.py'):
        if 'common' in py_file.parts:
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if 'logging.basicConfig' in line:
                    logging_setups.append((py_file, i))
        except (UnicodeDecodeError, IOError):
            pass

    return logging_setups


def generate_report():
    """Generate comprehensive refactoring report."""
    scripts_dir = Path(__file__).parent.parent

    print("=" * 80)
    print("C2PA ROBUSTNESS TESTING - CODE REFACTORING REPORT")
    print("=" * 80)
    print()

    # 1. Duplicate Functions
    print("1. DUPLICATE FUNCTION DEFINITIONS")
    print("-" * 40)
    duplicates = find_duplicate_functions(scripts_dir)
    if duplicates:
        for func_name, locations in sorted(duplicates.items()):
            print(f"\n  Function: {func_name}()")
            for path, line in locations:
                relative_path = path.relative_to(scripts_dir.parent)
                print(f"    - {relative_path}:{line}")
    else:
        print("  No duplicate functions found.")
    print()

    # 2. Subprocess Usage
    print("2. SUBPROCESS CALLS (Should use utils.run_command)")
    print("-" * 40)
    subprocess_calls = find_subprocess_usage(scripts_dir)
    if subprocess_calls:
        current_file = None
        for path, line, code in subprocess_calls[:20]:  # Limit to first 20
            relative_path = path.relative_to(scripts_dir.parent)
            if current_file != relative_path:
                print(f"\n  {relative_path}:")
                current_file = relative_path
            print(f"    Line {line}: {code[:60]}...")
        if len(subprocess_calls) > 20:
            print(f"\n  ... and {len(subprocess_calls) - 20} more occurrences")
    else:
        print("  No direct subprocess calls found.")
    print()

    # 3. CSV Operations
    print("3. CSV OPERATIONS (Should use utils CSV functions)")
    print("-" * 40)
    csv_ops = find_csv_operations(scripts_dir)
    if csv_ops:
        files_with_csv = set()
        for path, _, _ in csv_ops:
            files_with_csv.add(path.relative_to(scripts_dir.parent))

        print(f"  Found CSV operations in {len(files_with_csv)} files:")
        for file_path in sorted(files_with_csv):
            print(f"    - {file_path}")
    else:
        print("  No CSV operations found.")
    print()

    # 4. Logging Setup
    print("4. LOGGING SETUP (Should use utils.setup_logging)")
    print("-" * 40)
    logging_setups = find_logging_setup(scripts_dir)
    if logging_setups:
        print(f"  Found logging.basicConfig in {len(logging_setups)} files:")
        for path, line in logging_setups:
            relative_path = path.relative_to(scripts_dir.parent)
            print(f"    - {relative_path}:{line}")
    else:
        print("  No logging setup calls found.")
    print()

    # 5. Common Import Patterns
    print("5. COMMON IMPORTS (Candidates for utils)")
    print("-" * 40)
    all_imports = defaultdict(int)
    file_count = 0

    for py_file in scripts_dir.rglob('*.py'):
        if 'common' in py_file.parts:
            continue
        file_count += 1
        imports = analyze_imports(py_file)
        for imp in imports:
            all_imports[imp] += 1

    # Show imports used in >50% of files
    common_imports = [(imp, count) for imp, count in all_imports.items()
                      if count > file_count * 0.5 and imp not in ['sys', 'os']]

    if common_imports:
        print("  Imports used in >50% of files:")
        for imp, count in sorted(common_imports, key=lambda x: x[1], reverse=True):
            percentage = (count / file_count) * 100
            print(f"    - {imp}: {count}/{file_count} files ({percentage:.0f}%)")
    print()

    # 6. Recommendations
    print("6. REFACTORING RECOMMENDATIONS")
    print("-" * 40)
    print("  HIGH PRIORITY:")
    print("    [x] Create utils.py with shared functions")
    print("    [ ] Replace subprocess.run with utils.run_command (security)")
    print("    [ ] Replace logging.basicConfig with utils.setup_logging")
    print("    [ ] Use utils.write_csv_header and utils.append_csv_row")
    print()
    print("  MEDIUM PRIORITY:")
    print("    [ ] Consolidate duplicate functions:")
    for func_name in list(duplicates.keys())[:5]:
        print(f"        - {func_name}()")
    print("    [ ] Create utils.verify_c2pa_manifest wrapper")
    print("    [ ] Standardize file finding logic with utils.find_original_asset")
    print()
    print("  LOW PRIORITY:")
    print("    [ ] Add type hints to all functions")
    print("    [ ] Add docstrings to all modules")
    print("    [ ] Standardize error handling patterns")
    print()

    # 7. Security Issues
    print("7. SECURITY CONSIDERATIONS")
    print("-" * 40)
    security_issues = []

    for py_file in scripts_dir.rglob('*.py'):
        if 'common' in py_file.parts:
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for shell=True
            if 'shell=True' in content:
                security_issues.append((py_file, "Uses shell=True (injection risk)"))

            # Check for format strings in subprocess
            if re.search(r'subprocess.*f["\'].*{', content):
                security_issues.append((py_file, "F-strings in subprocess calls"))

            # Check for unvalidated input
            if 'sys.argv' in content and 'subprocess' in content:
                security_issues.append((py_file, "May pass unvalidated argv to subprocess"))
        except:
            pass

    if security_issues:
        print("  FOUND SECURITY ISSUES:")
        for path, issue in security_issues:
            relative_path = path.relative_to(scripts_dir.parent)
            print(f"    - {relative_path}: {issue}")
    else:
        print("  No major security issues found.")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total Python files analyzed: {file_count}")
    print(f"  Duplicate functions found: {len(duplicates)}")
    print(f"  Subprocess calls to refactor: {len(subprocess_calls)}")
    print(f"  Files with CSV operations: {len(files_with_csv) if csv_ops else 0}")
    print(f"  Files with logging setup: {len(logging_setups)}")
    print()
    print("  Next Steps:")
    print("    1. Import utils.py in scripts needing refactoring")
    print("    2. Replace redundant code with utils functions")
    print("    3. Run smoke tests to verify functionality")
    print("    4. Document changes in CHANGELOG.md")
    print()


if __name__ == "__main__":
    generate_report()