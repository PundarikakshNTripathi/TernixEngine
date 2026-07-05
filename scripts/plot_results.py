import json
import matplotlib.pyplot as plt
import argparse
import os
import re

def plot_and_update(json_path, output_dir, readme_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            print("No benchmark data found.")
            return
            
        names = []
        cpu_times = []
        naive_time = None
        avx2_time = None
        
        for b in benchmarks:
            name = b['name'].replace('BM_', '')
            time = b['cpu_time']
            names.append(name)
            cpu_times.append(time)
            if 'NaiveTernary' in name:
                naive_time = time
            elif 'AVX2Ternary' in name:
                avx2_time = time
                
        # Plotting
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, cpu_times, color='#00ff9d')
        plt.ylabel(f"CPU Time ({benchmarks[0]['time_unit']})", fontweight='bold')
        plt.title('TernixEngine AVX2 SIMD Microbenchmark Results', fontweight='bold', pad=20)
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.2)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.2f}", ha='center', va='bottom')
            
        plt.tight_layout()
        out_file = os.path.join(output_dir, 'benchmark_results.png')
        # Overwrite existing
        plt.savefig(out_file, dpi=300)
        print(f"Graph successfully saved/overwritten to {out_file}")
        
        # Update README
        if naive_time and avx2_time and os.path.exists(readme_path):
            speedup = naive_time / avx2_time
            with open(readme_path, 'r') as f:
                content = f.read()
                
            # Regex to replace table rows dynamically
            naive_pattern = r"\|\s*Naive Scalar\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|"
            avx2_pattern = r"\|\s*Tiled SIMD Accumulation\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|"
            
            new_naive = f"| Naive Scalar | Branching `if (w == 0)` | Base C++ | 1.0x (Baseline) | {naive_time:.2f} ms |"
            new_avx2 = f"| Tiled SIMD Accumulation | Tiled `M -> K -> N` | AVX2 (`_mm256_srlv_epi32`) | ~{speedup:.1f}x | {avx2_time:.2f} ms |"
            
            content = re.sub(naive_pattern, new_naive, content)
            content = re.sub(avx2_pattern, new_avx2, content)
            
            with open(readme_path, 'w') as f:
                f.write(content)
            print("README.md dynamically updated with actual empirical metrics!")
            
    except Exception as e:
        print(f"Failed to process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results.json")
    parser.add_argument("--output", default="benchmarks")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    plot_and_update(args.input, args.output, args.readme)
