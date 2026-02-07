#!/bin/bash
# Fast determinism pattern scanner
# Usage: ./bin/check_determinism_patterns.sh [file_or_dir...]
#        (accepts multiple files/dirs as arguments)

set -e

# Load Tier A file list from SST source
TIER_A_PATTERN_FILE="${TIER_A_PATTERN_FILE:-TRAINING/common/determinism_policy.py}"
# Extract Tier A files pattern (implementation: grep TIER_A_FILES from policy file)
# For now, hardcode but note this should come from SST
TIER_A_FILES="target_ranker.py|composite_score.py|training_plan_generator.py|target_routing.py|decision_engine.py|feature_selector.py"

# Collect all targets (handle multiple args)
TARGETS=("${@:-TRAINING}")

VIOLATIONS_FOUND=0

for TARGET in "${TARGETS[@]}"; do
    echo "Scanning: $TARGET"
    echo ""
    
    # Pattern 1: Filesystem operations without sorting
    echo "=== Filesystem Operations (unsorted) ==="
    MATCHES=$(rg -n "\.iterdir\(\)|\.glob\(|\.rglob\(|os\.listdir\(" "$TARGET" 2>/dev/null | \
        grep -v "sorted(" | \
        grep -v "DETERMINISM_OK" | \
        grep -v "iterdir_sorted\|glob_sorted\|rglob_sorted" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        VIOLATIONS_FOUND=1
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 2: Set iterations (NEVER acceptable in Tier A)
    echo "=== Set Iterations ==="
    MATCHES=$(rg -n "for .* in set\(|for .* in \{" "$TARGET" 2>/dev/null | \
        grep -v "sorted(" | \
        grep -v "DETERMINISM_OK" | \
        grep -v "sorted_unique" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        VIOLATIONS_FOUND=1
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 3: Dictionary iterations (Tier A files only)
    echo "=== Dictionary Iterations (Tier A files) ==="
    MATCHES=$(rg -n "\.items\(\)|\.keys\(\)" "$TARGET" 2>/dev/null | \
        grep -E "$TIER_A_FILES" | \
        grep -v "sorted(" | \
        grep -v "sorted_items\|sorted_keys" | \
        grep -v "DETERMINISM_OK" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        VIOLATIONS_FOUND=1
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 4: Time/UUID usage
    echo "=== Time/UUID Usage ==="
    MATCHES=$(rg -n "time\.time\(\)|datetime\.now\(\)|uuid\.uuid4\(\)|st_mtime" "$TARGET" 2>/dev/null | \
        grep -v "DETERMINISM_OK" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        # Note: This may be acceptable in some contexts, so don't fail on this alone
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 5: Concurrency patterns
    echo "=== Concurrency Patterns ==="
    MATCHES=$(rg -n "ThreadPoolExecutor|ProcessPoolExecutor|as_completed|imap_unordered" "$TARGET" 2>/dev/null | \
        grep -v "DETERMINISM_OK" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        # Note: May be acceptable if results are sorted after collection
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 6: Entry points (note: full verification requires Python script)
    echo "=== Entry Points (verification needed) ==="
    MATCHES=$(rg -n "if __name__.*==.*__main__|def main\(" "$TARGET" 2>/dev/null || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        echo "NOTE: Run bin/verify_determinism_init.py for full verification"
    else
        echo "None found"
    fi
    echo ""
    
    # Pattern 7: Score-based sorting without tie-breakers
    echo "=== Score-Based Sorting (check for tie-breakers) ==="
    MATCHES=$(rg -n "sorted\(.*key.*=.*lambda.*\.(score|auc|composite|ic|r2)" "$TARGET" 2>/dev/null | \
        grep -v "DETERMINISM_OK" | \
        grep -v "tie.*breaker\|-.*score.*target\|-.*auc.*target" || true)
    
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        echo "WARNING: Score-based sorting may need tie-breakers for deterministic ordering"
        # Don't fail on this - it's a warning, not a hard violation
    else
        echo "None found"
    fi
    echo ""
done

echo "Scan complete."

# Exit with error if Tier A violations found
if [ $VIOLATIONS_FOUND -eq 1 ]; then
    echo "ERROR: Non-deterministic patterns detected in Tier A files."
    echo "Fix or add DETERMINISM_OK(TierX, ISSUE-XXX): reason comment."
    exit 1
fi

exit 0
