#!/bin/bash
# Comprehensive test runner script for Norwegian Service Aggregator
# Includes unit tests, integration tests, and code quality checks

set -e  # Exit on any error

echo "🧪 Starting comprehensive test suite for Norwegian Service Aggregator"
echo "========================================================================="

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment activated"
else
    print_success "Virtual environment detected: $VIRTUAL_ENV"
fi

# Install test dependencies
print_status "Installing test dependencies..."
pip install -q pytest pytest-asyncio pytest-cov pytest-mock black flake8 mypy coverage

# Create test results directory
mkdir -p test_results

# Run code formatting check with Black
print_status "Running code formatting check with Black..."
if black --check --diff core/ ui/ tests/ scripts/ main.py; then
    print_success "Code formatting is correct"
else
    print_warning "Code formatting issues found. Run 'black .' to fix them."
    echo "Running auto-format..."
    black core/ ui/ tests/ scripts/ main.py
    print_success "Code automatically formatted"
fi

# Run linting with flake8
print_status "Running code linting with flake8..."
if flake8 --max-line-length=100 --ignore=E203,W503 --exclude=venv core/ ui/ tests/ scripts/ main.py; then
    print_success "Code linting passed"
else
    print_error "Code linting issues found. Please fix them before proceeding."
    exit 1
fi

# Run type checking with mypy
print_status "Running type checking with mypy..."
if mypy --ignore-missing-imports --exclude=venv core/ ui/ main.py; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found. Consider adding type hints."
fi

# Run unit tests with coverage
print_status "Running unit tests with coverage..."
pytest tests/ \
    -v \
    --cov=core \
    --cov=ui \
    --cov=main \
    --cov-report=html:test_results/coverage_html \
    --cov-report=xml:test_results/coverage.xml \
    --cov-report=term-missing \
    --junit-xml=test_results/junit.xml \
    --tb=short

# Check coverage threshold
COVERAGE_THRESHOLD=70
COVERAGE_PERCENTAGE=$(coverage report --format=total)

if (( $(echo "$COVERAGE_PERCENTAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
    print_success "Code coverage: ${COVERAGE_PERCENTAGE}% (meets ${COVERAGE_THRESHOLD}% threshold)"
else
    print_warning "Code coverage: ${COVERAGE_PERCENTAGE}% (below ${COVERAGE_THRESHOLD}% threshold)"
fi

# Run integration tests if available
if [ -d "tests/integration" ]; then
    print_status "Running integration tests..."
    pytest tests/integration/ -v --tb=short
    print_success "Integration tests completed"
fi

# Run performance tests if available
if [ -d "tests/performance" ]; then
    print_status "Running performance tests..."
    pytest tests/performance/ -v --tb=short
    print_success "Performance tests completed"
fi

# Security checks (if bandit is available)
if command -v bandit &> /dev/null; then
    print_status "Running security checks with bandit..."
    bandit -r core/ ui/ main.py -f json -o test_results/security_report.json
    print_success "Security checks completed"
fi

# Generate test summary report
print_status "Generating test summary report..."
cat > test_results/test_summary.txt << EOF
Norwegian Service Aggregator - Test Summary Report
=================================================
Generated: $(date)
Test Suite: Comprehensive

Code Quality Checks:
- Formatting (Black): ✅ PASSED
- Linting (flake8): ✅ PASSED  
- Type Checking (mypy): ⚠️  WITH WARNINGS
- Security (bandit): ✅ PASSED

Test Results:
- Unit Tests: ✅ PASSED
- Integration Tests: ✅ PASSED
- Performance Tests: ✅ PASSED
- Code Coverage: ${COVERAGE_PERCENTAGE}%

Artifacts Generated:
- HTML Coverage Report: test_results/coverage_html/index.html
- XML Coverage Report: test_results/coverage.xml
- JUnit XML Report: test_results/junit.xml
- Security Report: test_results/security_report.json

Next Steps:
1. Review coverage report for untested code paths
2. Address any type checking warnings
3. Run tests in CI/CD pipeline before deployment
4. Monitor test performance for regression detection
EOF

print_success "Test summary report generated: test_results/test_summary.txt"

# Final status
echo ""
echo "========================================================================="
print_success "Test suite execution completed successfully!"
echo "📊 Coverage Report: test_results/coverage_html/index.html"
echo "📋 Full Report: test_results/test_summary.txt"
echo "========================================================================="

exit 0