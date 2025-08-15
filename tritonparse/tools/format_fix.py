#!/usr/bin/env python3
"""
Format fix script for tritonparse project.

This script runs linters/formatters for Python or, when requested, for the website code:
- Python: usort (imports), ruff (lint), black (format)
- Website: eslint (with --fix)

Usage:
    python -m tritonparse.tools.format_fix [options]

Options:
    --check-only    Only check for issues, don't fix them
    --website       Run format/lint only for the website (website/)
    --verbose       Verbose output
    --help          Show this help message
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], verbose: bool = False, cwd: str | None = None) -> bool:
    """Run a command and return success status."""
    if verbose:
        if cwd:
            print(f"Running (cwd={cwd}): {' '.join(cmd)}")
        else:
            print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=cwd
        )

        if result.returncode != 0:
            if verbose:
                print(f"Command failed with return code {result.returncode}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            return False

        if verbose and result.stdout:
            print(result.stdout)

        return True
    except Exception as e:
        if verbose:
            print(f"Error running command: {e}")
        return False


def run_usort(check_only: bool = False, verbose: bool = False) -> bool:
    """Run usort for import sorting."""
    cmd = ["usort"]

    if check_only:
        cmd.extend(["check", "."])
    else:
        cmd.extend(["format", "."])

    return run_command(cmd, verbose)


def run_ruff_check(check_only: bool = False, verbose: bool = False) -> bool:
    """Run ruff for linting only."""
    cmd = ["ruff", "check", "."]

    if check_only:
        cmd.append("--diff")
    else:
        cmd.append("--fix")

    return run_command(cmd, verbose)


def run_black(check_only: bool = False, verbose: bool = False) -> bool:
    """Run black for code formatting."""
    cmd = ["black"]

    if check_only:
        cmd.extend(["--check", "--diff", "."])
    else:
        cmd.append(".")

    return run_command(cmd, verbose)


def run_website_eslint(check_only: bool = False, verbose: bool = False) -> bool:
    """Run ESLint for website (TypeScript/React) code.

    Uses local dev dependency via npx under the website directory.
    """
    # Resolve repo root from this file location: .../tritonparse/tritonparse/tools/format_fix.py -> repo root at parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    website_dir = repo_root / "website"

    # Use npx to run the eslint shipped with the website package.json
    cmd: list[str] = [
        "npx",
        "--yes",
        "eslint",
        ".",
    ]

    if check_only:
        # No --fix in check-only mode
        pass
    else:
        cmd.append("--fix")

    return run_command(cmd, verbose, cwd=str(website_dir))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Format fix script for tritonparse project",
        epilog="""
Examples:
    # Fix all formatting issues
    python -m tritonparse.tools.format_fix
    
    # Check for issues without fixing
    python -m tritonparse.tools.format_fix --check-only
    
    # Verbose output
    python -m tritonparse.tools.format_fix --verbose
        """,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for issues, don't fix them",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--website",
        action="store_true",
        help="Run website (website/) ESLint formatting instead of Python formatters",
    )

    args = parser.parse_args()

    # Run formatters
    success = True

    if args.website:
        print("Running ESLint for website formatting...")
        if not run_website_eslint(args.check_only, args.verbose):
            print("❌ website eslint failed")
            success = False
        else:
            print("✅ website eslint completed")
    else:
        # Python formatters
        print("Running usort for import sorting...")
        if not run_usort(args.check_only, args.verbose):
            print("❌ usort failed")
            success = False
        else:
            print("✅ usort completed")

        print("Running ruff for linting...")
        if not run_ruff_check(args.check_only, args.verbose):
            print("❌ ruff linting failed")
            success = False
        else:
            print("✅ ruff linting completed")

        print("Running black for code formatting...")
        if not run_black(args.check_only, args.verbose):
            print("❌ black failed")
            success = False
        else:
            print("✅ black completed")

    if success:
        print("\n🎉 All formatting tools completed successfully!")
        return 0
    else:
        print("\n❌ Some formatting tools failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
