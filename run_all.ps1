param (
    [switch]$BuildOnly
)

Write-Host "=== TernixEngine Dev Pipeline ===" -ForegroundColor Cyan

Write-Host "`n[1/4] Configuring CMake..." -ForegroundColor Green
cmake -B build -G Ninja
if ($LASTEXITCODE -ne 0) { throw "CMake config failed" }

Write-Host "`n[2/4] Compiling (Release)..." -ForegroundColor Green
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }

if ($BuildOnly) { exit }

Write-Host "`n[3/4] Executing Unit Tests..." -ForegroundColor Green
Set-Location build
ctest --output-on-failure
Set-Location ..

Write-Host "`n[4/4] Executing Microbenchmarks & Graphing..." -ForegroundColor Green
if (-not (Test-Path "benchmarks")) { New-Item -ItemType Directory -Path "benchmarks" | Out-Null }

# Execute Google Benchmarks and dynamically output to CSV
.\build\bench_simd_math.exe --benchmark_format=csv --benchmark_out=benchmarks\results.csv
.\build\bench_end_to_end.exe --benchmark_format=csv --benchmark_out=benchmarks\results_e2e.csv

# Generate graphs
python scripts\plot_results.py --input benchmarks\results.csv --output benchmarks

Write-Host "`nPipeline Execution Successful. Artifacts generated in /benchmarks." -ForegroundColor Cyan
