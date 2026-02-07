# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Leakage Assessment Dataclass

Single source of truth for leakage assessment flags.
Prevents contradictory reason strings like "overfit_likely; cv_not_suspicious".
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LeakageAssessment:
    """
    Comprehensive leakage assessment with all flags computed once.
    
    This prevents contradictory reason strings by computing all flags
    from a single assessment rather than building strings ad-hoc.
    """
    leak_scan_pass: bool
    cv_suspicious: bool
    overfit_likely: bool
    auc_too_high_models: List[str]
    cv_metric_name: Optional[str] = None  # Actual metric name from scoring source (e.g., "r2", "roc_auc", "accuracy")
    cv_metric_value: Optional[float] = None  # Maximum CV score across all models (finite only)
    primary_metric_tstat: Optional[float] = None  # Universal T-stat (finite only)
    
    def reason(self) -> str:
        """
        Generate reason string from flags.
        
        Returns:
            Semicolon-separated list of flags, or "none" if all flags are False
        """
        flags = []
        
        if self.cv_suspicious:
            flags.append("cv_suspicious")
        
        if self.overfit_likely:
            flags.append("overfit_likely")
        
        if self.auc_too_high_models:
            flags.append(f"auc>0.90:{','.join(self.auc_too_high_models)}")
        
        return "; ".join(flags) if flags else "none"
    
    def _auto_fix_decision(self) -> Tuple[bool, List[str]]:
        """
        Shared gate evaluator: computes both should_fix and failed_gates from same logic.
        
        This ensures should_auto_fix() and auto_fix_reason() never drift.
        
        Returns:
            Tuple of (should_fix: bool, failed_gates: List[str])
        """
        # should_auto_fix() = not self.leak_scan_pass or self.cv_suspicious
        # This is an OR gate: auto-fix runs if EITHER condition is true
        
        should_fix = not self.leak_scan_pass or self.cv_suspicious
        failed_gates = []
        
        if not should_fix:
            # Auto-fix skipped: both conditions failed
            # leak_scan_pass=True AND cv_suspicious=False
            if self.leak_scan_pass:
                failed_gates.append("no_leak_signals")
            if not self.cv_suspicious:
                failed_gates.append("cv_not_suspicious")
        
        return should_fix, failed_gates
    
    def should_auto_fix(self) -> bool:
        """
        Determine if auto-fix should run based on assessment.
        
        Auto-fix should run if:
        - Leak scan failed (leaky features detected)
        - CV is suspicious (suggests real leakage, not just overfitting)
        
        Auto-fix should NOT run if:
        - Only overfit_likely (classic overfitting, not leakage)
        - CV is normal (suggests legitimate signal)
        """
        should_fix, _ = self._auto_fix_decision()
        return should_fix
    
    def auto_fix_reason(self) -> Optional[str]:
        """
        Generate reason for why auto-fix was skipped (if applicable).
        
        Returns:
            Reason string if should_auto_fix() is False, None otherwise
        """
        should_fix, failed_gates = self._auto_fix_decision()
        
        if should_fix:
            return None
        
        # Add contextual info (not gates, but useful for diagnosis)
        reasons = failed_gates.copy()  # Start with failed gates
        
        if self.overfit_likely and not self.cv_suspicious:
            reasons.append("overfit_likely")
        
        # Add metric values for better comparability
        metric_parts = []
        if self.cv_metric_value is not None and self.cv_metric_name:
            metric_parts.append(f"cv_{self.cv_metric_name}={self.cv_metric_value:.4f}")
        
        if self.primary_metric_tstat is not None:
            metric_parts.append(f"tstat={self.primary_metric_tstat:.3f}")
        
        if metric_parts:
            reason_str = "; ".join(reasons) if reasons else "none"
            return f"{reason_str} ({', '.join(metric_parts)})"
        
        return "; ".join(reasons) if reasons else "none"
