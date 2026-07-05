@echo off
setlocal

echo === TernixEngine Dev Pipeline ===

echo.
echo [1/4] Configuring CMake...
cmake -B build -G Ninja
if %ERRORLEVEL% neq 0 (
    echo CMake config failed
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Compiling (Release)...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo Compilation failed
    exit /b %ERRORLEVEL%
)

if "%1"=="--build-only" exit /b 0

echo.
echo [3/4] Executing Unit Tests...
cd build
ctest --output-on-failure
cd ..

echo.
echo [4/4] Executing Microbenchmarks ^& Graphing...
if not exist benchmarks\ mkdir benchmarks

.\build\bench_simd_math.exe --benchmark_format=csv --benchmark_out=benchmarks\results.csv
.\build\bench_end_to_end.exe --benchmark_format=csv --benchmark_out=benchmarks\results_e2e.csv

python scripts\plot_results.py --input benchmarks\results.csv --output benchmarks

echo.
echo Pipeline Execution Successful. Artifacts generated in /benchmarks.
