#!/bin/bash

echo "🩸 Blood Cell Analyzer - Complete Test Suite"
echo "============================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $2 -eq 0 ]; then
        echo -e "${GREEN}✅ $1 PASSED${NC}"
    else
        echo -e "${RED}❌ $1 FAILED${NC}"
    fi
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}❌ Please run from the project root directory${NC}"
    exit 1
fi

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0

echo -e "${YELLOW}🔧 Setting up test environment...${NC}"

# 1. Backend Tests
echo -e "\n${YELLOW}🐍 Running Backend Tests...${NC}"
cd backend
source venv/bin/activate 2>/dev/null || echo "Virtual environment not activated"

# Run backend unit tests
python -m pytest tests/ -v --tb=short
BACKEND_UNIT_STATUS=$?
print_status "Backend Unit Tests" $BACKEND_UNIT_STATUS
TOTAL_TESTS=$((TOTAL_TESTS + 1))
[ $BACKEND_UNIT_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

# Run backend API tests
python run_all_tests.py
BACKEND_API_STATUS=$?
print_status "Backend API Tests" $BACKEND_API_STATUS
TOTAL_TESTS=$((TOTAL_TESTS + 1))
[ $BACKEND_API_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

cd ..

# 2. Frontend Tests
echo -e "\n${YELLOW}🎨 Running Frontend Tests...${NC}"
cd frontend

# Run frontend tests
npm test -- --coverage --watchAll=false --ci 2>/dev/null
FRONTEND_STATUS=$?
print_status "Frontend Tests" $FRONTEND_STATUS
TOTAL_TESTS=$((TOTAL_TESTS + 1))
[ $FRONTEND_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

# Run frontend test runner
node test_runner.js
FRONTEND_RUNNER_STATUS=$?
print_status "Frontend Test Runner" $FRONTEND_RUNNER_STATUS
TOTAL_TESTS=$((TOTAL_TESTS + 1))
[ $FRONTEND_RUNNER_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

cd ..

# 3. Integration Tests
echo -e "\n${YELLOW}🔗 Running Integration Tests...${NC}"
cd tests

# Check if services are running
echo "Checking if services are running..."
python integration_test.py
INTEGRATION_STATUS=$?
print_status "Integration Tests" $INTEGRATION_STATUS
TOTAL_TESTS=$((TOTAL_TESTS + 1))
[ $INTEGRATION_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

cd ..

# 4. End-to-End Tests (if available)
if [ -f "tests/e2e_test.py" ]; then
    echo -e "\n${YELLOW}🎭 Running End-to-End Tests...${NC}"
    cd tests
    python e2e_test.py
    E2E_STATUS=$?
    print_status "End-to-End Tests" $E2E_STATUS
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    [ $E2E_STATUS -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))
    cd ..
fi

# Final Summary
echo -e "\n============================================="
echo -e "${YELLOW}📊 Final Test Summary:${NC}"
echo -e "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$((TOTAL_TESTS - PASSED_TESTS))${NC}"

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo -e "Success Rate: ${GREEN}$SUCCESS_RATE%${NC}"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "\n${GREEN}🎉 All tests passed! Your Blood Cell Analyzer is working perfectly!${NC}"
    exit 0
else
    echo -e "\n${RED}⚠️  Some tests failed. Please check the output above.${NC}"
    exit 1
fi
