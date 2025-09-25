#!/usr/bin/env python3

"""
Test te_llama_phase1.py and te_llama_phase2.py to see how they compare
to our proven hybrid approach
"""

import subprocess
import sys
import time

def run_benchmark(script_name, description):
    """Run a benchmark script and capture results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}")

    try:
        # Run the script
        start_time = time.time()
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        elapsed = time.time() - start_time

        print(f"\nScript completed in {elapsed:.1f} seconds")

        # Print stdout
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)

        # Print stderr if there were issues but script still ran
        if result.stderr and result.returncode == 0:
            print("\nSTDERR (warnings):")
            print(result.stderr)

        # Handle errors
        if result.returncode != 0:
            print(f"\n❌ Script failed with return code {result.returncode}")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False, None

        return True, result.stdout

    except subprocess.TimeoutExpired:
        print(f"\n⏰ Script timed out after 5 minutes")
        return False, None
    except Exception as e:
        print(f"\n❌ Error running script: {e}")
        return False, None

def extract_performance_data(output):
    """Extract key performance metrics from benchmark output"""
    if not output:
        return {}

    metrics = {}
    lines = output.split('\n')

    for line in lines:
        # Look for performance indicators
        if 'speedup' in line.lower() or 'faster' in line.lower():
            metrics['speedup_info'] = line.strip()
        if 'ms' in line and ('average' in line.lower() or 'time' in line.lower()):
            metrics['timing_info'] = line.strip()
        if 'improvement' in line.lower() and '%' in line:
            metrics['improvement'] = line.strip()

    return metrics

def main():
    print("=== TE LLAMA PHASES BENCHMARK COMPARISON ===")
    print("Testing te_llama_phase1.py and te_llama_phase2.py")
    print("to compare against our proven hybrid approach findings")

    # List of scripts to test
    scripts = [
        ("te_llama_phase1.py", "Phase 1: Extended Surgical Compilation"),
        ("te_llama_phase2.py", "Phase 2: TE Attention Capturable")
    ]

    results = {}

    for script, description in scripts:
        print(f"\n\n" + "="*100)
        print(f"TESTING: {script}")
        print("="*100)

        success, output = run_benchmark(script, description)

        if success:
            metrics = extract_performance_data(output)
            results[script] = {
                'success': True,
                'metrics': metrics,
                'output': output
            }
            print(f"\n✅ {script} completed successfully")

            # Print key metrics
            if metrics:
                print("\nKey Performance Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
        else:
            results[script] = {
                'success': False,
                'error': "Script failed to run"
            }
            print(f"\n❌ {script} failed")

    # Summary
    print(f"\n\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")

    successful_runs = [script for script, result in results.items() if result['success']]

    if successful_runs:
        print(f"\n✅ Successful benchmarks: {len(successful_runs)}/{len(scripts)}")
        for script in successful_runs:
            print(f"  - {script}")

        print(f"\n📊 Performance Comparison:")
        print(f"Our proven findings (from comprehensive benchmarks):")
        print(f"  - Hybrid Approach: 16.7-16.8 ms (optimal)")
        print(f"  - TE TransformerLayer: 17.3-17.5 ms")
        print(f"  - TE MultiheadAttention: 19.8-20.8 ms")

        print(f"\nPhase Results:")
        for script in successful_runs:
            metrics = results[script]['metrics']
            print(f"  - {script}: {metrics.get('timing_info', 'No clear timing found')}")

    else:
        print(f"\n❌ No benchmarks completed successfully")
        print(f"This might indicate:")
        print(f"  - Compatibility issues with current TE version")
        print(f"  - Missing dependencies")
        print(f"  - API changes since these were written")

    print(f"\n🎯 Key Takeaway:")
    print(f"Based on our comprehensive analysis, the optimal approach is:")
    print(f"  Hybrid: TE Linear + cuDNN SDPA + torch.compile")
    print(f"  This beats TE's best configurations by 3-5% and standard TE by 18-24%")

if __name__ == "__main__":
    main()