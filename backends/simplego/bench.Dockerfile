# Dockerfile for benchmarking go-highway SIMD acceleration in GoMLX
#
# Build and run on amd64:
#   docker build --platform linux/amd64 -f backends/simplego/bench.Dockerfile -t gomlx-simd-bench .
#   docker run --platform linux/amd64 --rm gomlx-simd-bench
#
# Or with docker buildx for cross-platform:
#   docker buildx build --platform linux/amd64 -f backends/simplego/bench.Dockerfile -t gomlx-simd-bench --load .
#   docker run --platform linux/amd64 --rm gomlx-simd-bench

FROM --platform=linux/amd64 golang:1.24-bookworm

# Install Go 1.26rc1 for SIMD support
RUN go install golang.org/dl/go1.26rc1@latest && \
    go1.26rc1 download

WORKDIR /app

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum ./
RUN go1.26rc1 mod download

# Copy the source code
COPY . .

# Create benchmark script
RUN cat > /run_benchmarks.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "GoMLX SIMD Benchmark: go-highway integration"
echo "=============================================="
echo ""
echo "Architecture: $(uname -m)"
echo "CPU Info: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo 'N/A')"
echo ""

echo ">>> Running benchmarks WITHOUT SIMD (scalar baseline)..."
echo ""
go1.26rc1 test -bench="Float32_1M" -benchtime=2s ./backends/simplego/... 2>&1 | tee /tmp/no_simd.txt
echo ""

echo ">>> Running benchmarks WITH SIMD (AVX2/AVX-512)..."
echo ""
GOEXPERIMENT=simd go1.26rc1 test -bench="Float32_1M" -benchtime=2s ./backends/simplego/... 2>&1 | tee /tmp/simd.txt
echo ""

echo "=============================================="
echo "Performance Comparison (Float32 1M elements)"
echo "=============================================="
echo ""
echo "Operation        | Scalar (ns/op) | SIMD (ns/op) | Speedup"
echo "-----------------|----------------|--------------|--------"

for op in Exp Log Tanh Sin Cos Logistic Erf Sqrt; do
    scalar=$(grep "Benchmark${op}_Float32_1M" /tmp/no_simd.txt | awk '{print $3}')
    simd=$(grep "Benchmark${op}_Float32_1M" /tmp/simd.txt | awk '{print $3}')
    if [ -n "$scalar" ] && [ -n "$simd" ]; then
        speedup=$(echo "scale=2; $scalar / $simd" | bc)
        printf "%-16s | %14s | %12s | %sx\n" "$op" "$scalar" "$simd" "$speedup"
    fi
done
echo ""
EOF
RUN chmod +x /run_benchmarks.sh

CMD ["/run_benchmarks.sh"]
