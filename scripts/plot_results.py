import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_benchmarks(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmarks first.")
        return
        
    try:
        # Google Benchmark CSVs contain metadata headers. We find the CSV header row.
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('name,'):
                header_idx = i
                break
                
        df = pd.read_csv(csv_path, skiprows=header_idx)
        
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 7))
        
        # We focus on real_time or cpu_time
        df = df.sort_values(by='cpu_time', ascending=False)
        bars = plt.bar(df['name'].str.replace('BM_', ''), df['cpu_time'], color='#00ff9d')
        
        plt.ylabel(f"CPU Time ({df['time_unit'].iloc[0]})", fontweight='bold', fontsize=12)
        plt.title('TernixEngine AVX2 SIMD Microbenchmark Results', fontweight='bold', fontsize=16, pad=20)
        plt.xticks(rotation=25, ha='right', fontsize=10)
        plt.grid(axis='y', alpha=0.2)
        
        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.1f}", ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        out_file = os.path.join(output_dir, 'benchmark_results.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Graph successfully saved to {out_file}")
        
    except Exception as e:
        print(f"Failed to plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results.csv")
    parser.add_argument("--output", default="benchmarks")
    args = parser.parse_args()
    plot_benchmarks(args.input, args.output)
