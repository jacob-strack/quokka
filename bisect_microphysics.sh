#!/bin/bash

# Git bisect script for finding the Microphysics commit that broke HydroContact test
# This script helps bisect the Microphysics submodule to find the commit that 
# caused the HydroContact test to fail due to CODATA constants changes.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GOOD_COMMIT="05b09dc"    # Known good commit in Microphysics
BAD_COMMIT="860ef38"     # Known bad commit in Microphysics
BUILD_DIR="build_bisect"
TEST_NAME="HydroContact"
INPUT_FILE="inputs/contact_wave.in"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    start     - Initialize submodules and start git bisect process
    test      - Build and test the current commit
    good      - Mark current commit as good (test passes)
    bad       - Mark current commit as bad (test fails)  
    status    - Show current bisect status
    manual    - Show manual bisect instructions
    reset     - Reset and clean up the bisect process
    help      - Show this help message

Example workflow:
    ./bisect_microphysics.sh start    # Start bisect
    ./bisect_microphysics.sh test     # Test current commit
    ./bisect_microphysics.sh good     # If test passed
    ./bisect_microphysics.sh test     # Test next commit  
    ./bisect_microphysics.sh bad      # If test failed
    # Repeat until bisect completes
    ./bisect_microphysics.sh reset    # Clean up when done

The script will systematically test commits between:
- GOOD: $GOOD_COMMIT (HydroContact test passes)
- BAD:  $BAD_COMMIT (HydroContact test fails)
EOF
}

# Function to initialize submodules
init_submodules() {
    print_status "Initializing and updating submodules..."
    git submodule update --init --recursive
    
    # Ensure we're in the correct directory
    if [ ! -d "extern/Microphysics" ]; then
        print_error "Microphysics submodule not found at extern/Microphysics"
        exit 1
    fi
    
    print_success "Submodules initialized"
}

# Function to start git bisect
start_bisect() {
    print_status "Starting git bisect for Microphysics submodule..."
    
    # Initialize submodules first
    init_submodules
    
    # Enter Microphysics directory and start bisect
    cd extern/Microphysics
    
    # Reset any existing bisect
    git bisect reset 2>/dev/null || true
    
    # Start bisect
    git bisect start
    git bisect bad $BAD_COMMIT
    git bisect good $GOOD_COMMIT
    
    cd ../..
    
    print_success "Git bisect started. Current commit in Microphysics:"
    cd extern/Microphysics && git log --oneline -1 && cd ../..
    print_status "Run './bisect_microphysics.sh test' to test this commit"
}

# Function to build the project
build_project() {
    print_status "Building project..."
    
    # Clean previous build
    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure with cmake
    # Use AMReX_SPACEDIM=1 for the 1D HydroContact test
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
        -DAMReX_SPACEDIM=1
    
    # Build with ninja
    ninja -j6 test_hydro_contact
    
    cd ..
    print_success "Build completed"
}

# Function to run the HydroContact test
run_test() {
    print_status "Running HydroContact test..."
    
    cd $BUILD_DIR
    
    # Check if the test executable exists
    if [ ! -f "src/problems/HydroContact/test_hydro_contact" ]; then
        print_error "HydroContact test executable not found"
        cd ..
        return 1
    fi
    
    # Run the test
    print_status "Executing: ./src/problems/HydroContact/test_hydro_contact ../$INPUT_FILE"
    
    # Capture output and check for success
    if timeout 300 ./src/problems/HydroContact/test_hydro_contact ../$INPUT_FILE > test_output.log 2>&1; then
        # Check if the test passed by looking for error norm
        if grep -q "ERROR NORM.*= 0" test_output.log 2>/dev/null; then
            print_success "Test PASSED - Error norm is exactly zero"
            cd ..
            return 0
        elif grep -q "ERROR NORM" test_output.log; then
            local error_norm=$(grep "ERROR NORM" test_output.log | tail -1)
            print_error "Test FAILED - $error_norm"
            cd ..
            return 1
        else
            print_warning "Test completed but error norm not found in output"
            print_status "Last few lines of output:"
            tail -10 test_output.log
            cd ..
            return 1
        fi
    else
        print_error "Test execution failed or timed out"
        print_status "Last few lines of output:"
        tail -10 test_output.log 2>/dev/null || echo "No output available"
        cd ..
        return 1
    fi
}

# Function to test current commit
test_commit() {
    print_status "Testing current Microphysics commit..."
    
    # Show current commit
    cd extern/Microphysics
    local current_commit=$(git log --oneline -1)
    print_status "Current Microphysics commit: $current_commit"
    cd ../..
    
    # Update submodules to current commit
    git submodule update --recursive
    
    # Build and test
    if build_project && run_test; then
        print_success "Current commit PASSED the test"
        print_status "Run './bisect_microphysics.sh good' to mark as good"
        return 0
    else
        print_error "Current commit FAILED the test"  
        print_status "Run './bisect_microphysics.sh bad' to mark as bad"
        return 1
    fi
}

# Function to mark commit as good
mark_good() {
    print_status "Marking current commit as good..."
    cd extern/Microphysics
    git bisect good
    cd ../..
    
    # Update submodule to the next commit identified by bisect
    git submodule update --recursive
    
    check_bisect_status
}

# Function to mark commit as bad  
mark_bad() {
    print_status "Marking current commit as bad..."
    cd extern/Microphysics
    git bisect bad
    cd ../..
    
    # Update submodule to the next commit identified by bisect
    git submodule update --recursive
    
    check_bisect_status
}

# Function to check bisect status
check_bisect_status() {
    cd extern/Microphysics
    
    if git bisect log >/dev/null 2>&1; then
        print_status "Current bisect status:"
        git bisect log | head -20
        
        # Check if bisect is complete
        if git status | grep -q "You are currently bisecting"; then
            local current_commit=$(git log --oneline -1)
            print_status "Next commit to test: $current_commit"
            print_status "Run './bisect_microphysics.sh test' to test this commit"
        else
            print_success "Bisect may be complete! Check the output above."
        fi
    else
        print_warning "No active bisect found"
    fi
    
    cd ../..
}

# Function to show bisect status
show_status() {
    print_status "Checking git bisect status..."
    check_bisect_status
}

# Function to reset bisect
reset_bisect() {
    print_status "Resetting git bisect..."
    
    cd extern/Microphysics
    git bisect reset 2>/dev/null || true
    cd ../..
    
    # Clean build directory
    rm -rf $BUILD_DIR
    
    print_success "Bisect reset and build directory cleaned"
}

# Function to show manual instructions
show_manual() {
    cat << EOF
Manual Git Bisect Instructions for Microphysics:

1. Start the bisect process:
   cd extern/Microphysics
   git bisect start
   git bisect bad $BAD_COMMIT
   git bisect good $GOOD_COMMIT
   cd ../..

2. For each commit to test:
   # Update submodules
   git submodule update --recursive
   
   # Build the project
   rm -rf $BUILD_DIR
   mkdir $BUILD_DIR && cd $BUILD_DIR
   cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DAMReX_SPACEDIM=1
   ninja -j6
   cd ..
   
   # Run the test
   cd $BUILD_DIR
   ./src/problems/HydroContact/test_hydro_contact ../$INPUT_FILE
   cd ..
   
   # Mark the commit based on test result
   cd extern/Microphysics
   git bisect good    # if error norm = 0 (test passed)
   git bisect bad     # if error norm > 0 (test failed)
   cd ../..

3. When done:
   cd extern/Microphysics
   git bisect reset
   cd ../..
   rm -rf $BUILD_DIR

Expected outcome:
- GOOD commits: Error norm should be exactly 0
- BAD commits: Error norm will be > 0 (even very small values like 1e-15)
EOF
}

# Main script logic
case "${1:-help}" in
    start)
        start_bisect
        ;;
    test)
        test_commit
        ;;
    good)
        mark_good
        ;;
    bad)
        mark_bad
        ;;
    status)
        show_status
        ;;
    manual)
        show_manual
        ;;
    reset)
        reset_bisect
        ;;
    help|*)
        show_usage
        ;;
esac
