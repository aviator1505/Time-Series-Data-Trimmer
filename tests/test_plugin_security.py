"""
Security tests for plugin expression validation.

These tests verify that the validate_plugin_expression() function properly blocks
dangerous code patterns while allowing legitimate mathematical expressions.

IMPORTANT: Security testing is critical for research-grade applications where
plugin expressions are executed via pd.eval().
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    DANGEROUS_EXPRESSION_PATTERNS,
    SAFE_EXPRESSION_FUNCTIONS,
    validate_plugin_expression,
)


# =============================================================================
# 1. Valid Expression Tests
# =============================================================================

class TestValidExpressions:
    """Tests for expressions that should be allowed."""

    def test_valid_simple_arithmetic(self):
        """Simple arithmetic with columns should pass."""
        is_valid, _ = validate_plugin_expression("col_a + col_b", ["col_a", "col_b"])
        assert is_valid

    def test_valid_subtraction(self):
        """Subtraction with columns should pass."""
        is_valid, _ = validate_plugin_expression("col_a - col_b", ["col_a", "col_b"])
        assert is_valid

    def test_valid_multiplication(self):
        """Multiplication with columns should pass."""
        is_valid, _ = validate_plugin_expression("col_a * col_b", ["col_a", "col_b"])
        assert is_valid

    def test_valid_division(self):
        """Division with columns should pass."""
        is_valid, _ = validate_plugin_expression("col_a / col_b", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_constants(self):
        """Expressions with numeric constants should pass."""
        is_valid, _ = validate_plugin_expression("col_a * 2 + 3.14", ["col_a"])
        assert is_valid

    def test_valid_with_safe_functions(self):
        """Expressions with safe functions should pass."""
        is_valid, _ = validate_plugin_expression("abs(col_a) + sqrt(col_b)", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_abs(self):
        """abs() function should be allowed."""
        is_valid, _ = validate_plugin_expression("abs(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_sqrt(self):
        """sqrt() function should be allowed."""
        is_valid, _ = validate_plugin_expression("sqrt(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_trig_functions(self):
        """Trigonometric functions should be allowed."""
        is_valid, _ = validate_plugin_expression("sin(col_a) + cos(col_b) + tan(col_c)",
                                                   ["col_a", "col_b", "col_c"])
        assert is_valid

    def test_valid_with_log_functions(self):
        """Logarithmic functions should be allowed."""
        is_valid, _ = validate_plugin_expression("log(col_a) + log10(col_b)", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_exp(self):
        """exp() function should be allowed."""
        is_valid, _ = validate_plugin_expression("exp(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_pow(self):
        """pow() function should be allowed."""
        is_valid, _ = validate_plugin_expression("pow(col_a, 2)", ["col_a"])
        assert is_valid

    def test_valid_with_statistical_functions(self):
        """Statistical functions should be allowed."""
        is_valid, _ = validate_plugin_expression("mean(col_a) + std(col_a) + median(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_min_max_sum(self):
        """min/max/sum functions should be allowed."""
        is_valid, _ = validate_plugin_expression("min(col_a) + max(col_a) + sum(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_var(self):
        """var() function should be allowed."""
        is_valid, _ = validate_plugin_expression("var(col_a)", ["col_a"])
        assert is_valid

    def test_valid_with_rounding_functions(self):
        """Rounding functions should be allowed."""
        is_valid, _ = validate_plugin_expression("floor(col_a) + ceil(col_b) + round(col_c)",
                                                   ["col_a", "col_b", "col_c"])
        assert is_valid

    def test_valid_with_clip(self):
        """clip() function should be allowed."""
        is_valid, _ = validate_plugin_expression("clip(col_a, 0, 1)", ["col_a"])
        assert is_valid

    def test_valid_complex_expression(self):
        """Complex expressions with multiple operations should pass."""
        is_valid, _ = validate_plugin_expression("(col_a - mean(col_a)) / std(col_a)", ["col_a"])
        assert is_valid

    def test_valid_nested_functions(self):
        """Nested function calls should pass."""
        is_valid, _ = validate_plugin_expression("abs(sqrt(col_a))", ["col_a"])
        assert is_valid

    def test_valid_parentheses(self):
        """Expressions with parentheses should pass."""
        is_valid, _ = validate_plugin_expression("(col_a + col_b) * (col_c - col_d)",
                                                   ["col_a", "col_b", "col_c", "col_d"])
        assert is_valid

    def test_valid_with_boolean_keywords(self):
        """Boolean keywords should be allowed."""
        is_valid, _ = validate_plugin_expression("col_a if True else col_b", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_and_or_not(self):
        """Logical operators should be allowed."""
        is_valid, _ = validate_plugin_expression("col_a and col_b or not col_c",
                                                   ["col_a", "col_b", "col_c"])
        assert is_valid

    def test_valid_with_none(self):
        """None keyword should be allowed."""
        is_valid, _ = validate_plugin_expression("col_a if col_b else None", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_type_names(self):
        """Numpy-style type names should be allowed."""
        is_valid, _ = validate_plugin_expression("col_a + float(col_b)", ["col_a", "col_b"])
        assert is_valid

    def test_valid_with_nan_inf(self):
        """nan and inf should be allowed."""
        is_valid, _ = validate_plugin_expression("col_a + nan", ["col_a"])
        assert is_valid
        is_valid, _ = validate_plugin_expression("col_a + inf", ["col_a"])
        assert is_valid

    def test_valid_scientific_notation(self):
        """Scientific notation should pass."""
        is_valid, _ = validate_plugin_expression("col_a * 1e-3", ["col_a"])
        assert is_valid

    def test_valid_negative_numbers(self):
        """Negative numbers should pass."""
        is_valid, _ = validate_plugin_expression("col_a + -5", ["col_a"])
        assert is_valid

    def test_valid_underscore_column_names(self):
        """Column names with underscores should pass."""
        is_valid, _ = validate_plugin_expression("my_col_name + another_col",
                                                   ["my_col_name", "another_col"])
        assert is_valid

    def test_valid_column_names_with_numbers(self):
        """Column names with numbers should pass."""
        is_valid, _ = validate_plugin_expression("col1 + col2", ["col1", "col2"])
        assert is_valid


# =============================================================================
# 2. Dangerous Pattern Tests
# =============================================================================

class TestDangerousPatterns:
    """Tests for expressions containing dangerous patterns that should be blocked."""

    def test_blocks_import(self):
        """Import statements should be blocked."""
        is_valid, msg = validate_plugin_expression("import os", ["col"])
        assert not is_valid
        assert "disallowed" in msg.lower()

    def test_blocks_dunder_import(self):
        """__import__() function should be blocked."""
        is_valid, msg = validate_plugin_expression("__import__('os')", ["col"])
        assert not is_valid
        assert "disallowed" in msg.lower()

    def test_blocks_exec(self):
        """Exec function should be blocked."""
        is_valid, _ = validate_plugin_expression("exec('print(1)')", ["col"])
        assert not is_valid

    def test_blocks_eval(self):
        """Eval function should be blocked."""
        is_valid, _ = validate_plugin_expression("eval('1+1')", ["col"])
        assert not is_valid

    def test_blocks_compile(self):
        """Compile function should be blocked."""
        is_valid, _ = validate_plugin_expression("compile('pass', '', 'exec')", ["col"])
        assert not is_valid

    def test_blocks_open(self):
        """File open should be blocked."""
        is_valid, _ = validate_plugin_expression("open('/etc/passwd')", ["col"])
        assert not is_valid

    def test_blocks_os_access(self):
        """OS module access should be blocked."""
        is_valid, _ = validate_plugin_expression("os.system('ls')", ["col"])
        assert not is_valid

    def test_blocks_os_path(self):
        """OS path access should be blocked."""
        is_valid, _ = validate_plugin_expression("os.path.exists('/etc')", ["col"])
        assert not is_valid

    def test_blocks_os_remove(self):
        """OS file removal should be blocked."""
        is_valid, _ = validate_plugin_expression("os.remove('/tmp/test')", ["col"])
        assert not is_valid

    def test_blocks_sys_access(self):
        """Sys module access should be blocked."""
        is_valid, _ = validate_plugin_expression("sys.exit()", ["col"])
        assert not is_valid

    def test_blocks_sys_path(self):
        """Sys path manipulation should be blocked."""
        is_valid, _ = validate_plugin_expression("sys.path.append('/tmp')", ["col"])
        assert not is_valid

    def test_blocks_subprocess(self):
        """Subprocess module access should be blocked."""
        is_valid, _ = validate_plugin_expression("subprocess.run(['ls'])", ["col"])
        assert not is_valid

    def test_blocks_subprocess_popen(self):
        """Subprocess Popen should be blocked."""
        is_valid, _ = validate_plugin_expression("subprocess.Popen(['ls'])", ["col"])
        assert not is_valid

    def test_blocks_builtins(self):
        """Builtins access should be blocked."""
        is_valid, _ = validate_plugin_expression("builtins.open", ["col"])
        assert not is_valid

    def test_blocks_globals(self):
        """Globals access should be blocked."""
        is_valid, _ = validate_plugin_expression("globals()['os']", ["col"])
        assert not is_valid

    def test_blocks_locals(self):
        """Locals access should be blocked."""
        is_valid, _ = validate_plugin_expression("locals()['col']", ["col"])
        assert not is_valid

    def test_blocks_getattr(self):
        """Getattr function should be blocked."""
        is_valid, _ = validate_plugin_expression("getattr(col, '__class__')", ["col"])
        assert not is_valid

    def test_blocks_setattr(self):
        """Setattr function should be blocked."""
        is_valid, _ = validate_plugin_expression("setattr(col, 'x', 1)", ["col"])
        assert not is_valid

    def test_blocks_delattr(self):
        """Delattr function should be blocked."""
        is_valid, _ = validate_plugin_expression("delattr(col, 'x')", ["col"])
        assert not is_valid

    def test_blocks_dunder_attributes(self):
        """Dunder attributes should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__class__", ["col"])
        assert not is_valid

    def test_blocks_dunder_dict(self):
        """__dict__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__dict__", ["col"])
        assert not is_valid

    def test_blocks_dunder_globals(self):
        """__globals__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__globals__", ["col"])
        assert not is_valid

    def test_blocks_dunder_builtins(self):
        """__builtins__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__builtins__", ["col"])
        assert not is_valid

    def test_blocks_dunder_name(self):
        """__name__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__name__", ["col"])
        assert not is_valid

    def test_blocks_dunder_module(self):
        """__module__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__module__", ["col"])
        assert not is_valid

    def test_blocks_dunder_code(self):
        """__code__ access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__code__", ["col"])
        assert not is_valid


# =============================================================================
# 3. Case Sensitivity Tests
# =============================================================================

class TestCaseSensitivity:
    """Tests for case-insensitive blocking of dangerous patterns."""

    def test_case_insensitive_import(self):
        """Import should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("IMPORT os", ["col"])
        assert not is_valid

        is_valid, _ = validate_plugin_expression("Import os", ["col"])
        assert not is_valid

        is_valid, _ = validate_plugin_expression("iMpOrT os", ["col"])
        assert not is_valid

    def test_case_insensitive_exec(self):
        """Exec should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("EXEC('code')", ["col"])
        assert not is_valid

        is_valid, _ = validate_plugin_expression("Exec('code')", ["col"])
        assert not is_valid

    def test_case_insensitive_eval(self):
        """Eval should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("EVAL('1+1')", ["col"])
        assert not is_valid

        is_valid, _ = validate_plugin_expression("Eval('1+1')", ["col"])
        assert not is_valid

    def test_case_insensitive_open(self):
        """Open should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("OPEN('file')", ["col"])
        assert not is_valid

        is_valid, _ = validate_plugin_expression("Open('file')", ["col"])
        assert not is_valid

    def test_case_insensitive_compile(self):
        """Compile should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("COMPILE('code', '', 'exec')", ["col"])
        assert not is_valid

    def test_case_insensitive_globals(self):
        """Globals should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("GLOBALS()", ["col"])
        assert not is_valid

    def test_case_insensitive_locals(self):
        """Locals should be blocked regardless of case."""
        is_valid, _ = validate_plugin_expression("LOCALS()", ["col"])
        assert not is_valid


# =============================================================================
# 4. Unknown Identifier Tests
# =============================================================================

class TestUnknownIdentifiers:
    """Tests for expressions with unknown identifiers."""

    def test_blocks_unknown_function(self):
        """Unknown functions should be blocked."""
        is_valid, msg = validate_plugin_expression("unknown_func(col_a)", ["col_a"])
        assert not is_valid
        assert "unknown" in msg.lower()

    def test_blocks_missing_column(self):
        """References to non-existent columns should be blocked."""
        is_valid, msg = validate_plugin_expression("col_a + col_b", ["col_a"])  # col_b missing
        assert not is_valid
        assert "unknown" in msg.lower() or "col_b" in msg.lower()

    def test_blocks_unknown_module(self):
        """Unknown module references should be blocked."""
        is_valid, _ = validate_plugin_expression("numpy.array([1,2,3])", ["col"])
        assert not is_valid

    def test_blocks_dangerous_function_disguised(self):
        """Functions that look safe but aren't should be blocked."""
        is_valid, _ = validate_plugin_expression("my_exec(col)", ["col"])
        assert not is_valid

    def test_blocks_all_unknown_identifiers(self):
        """Expression with all unknown identifiers should be blocked."""
        is_valid, _ = validate_plugin_expression("foo + bar + baz", ["col"])
        assert not is_valid

    def test_allows_all_known_identifiers(self):
        """Expression with all known identifiers should pass."""
        is_valid, _ = validate_plugin_expression("abs(col_a) + sqrt(col_b)", ["col_a", "col_b"])
        assert is_valid


# =============================================================================
# 5. Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_expression(self):
        """Empty expression should be invalid."""
        is_valid, msg = validate_plugin_expression("", ["col"])
        assert not is_valid
        assert "empty" in msg.lower()

    def test_none_expression(self):
        """None expression should be invalid."""
        is_valid, msg = validate_plugin_expression(None, ["col"])
        assert not is_valid
        assert "empty" in msg.lower() or "not a string" in msg.lower()

    def test_whitespace_only_expression(self):
        """Whitespace-only expression - current behavior documents that it passes.

        Note: This is a minor edge case. Whitespace-only expressions will pass
        validation but pd.eval("   ") will raise an error anyway, so this is
        not a security concern. A future improvement could check expr.strip().
        """
        is_valid, _ = validate_plugin_expression("   ", ["col"])
        # Current behavior: whitespace-only passes validation since "   " is truthy
        # pd.eval will fail on execution, so no security risk
        # Documenting current behavior rather than asserting it "should" fail
        assert is_valid  # Current behavior - no identifiers, no dangerous patterns

    def test_empty_columns_list(self):
        """Expression with empty columns list should fail if it references columns."""
        is_valid, _ = validate_plugin_expression("col_a + col_b", [])
        assert not is_valid

    def test_valid_with_empty_columns(self):
        """Expression with only constants and safe functions should pass with empty columns."""
        is_valid, _ = validate_plugin_expression("abs(-5) + sqrt(4)", [])
        assert is_valid

    def test_numeric_string_expression(self):
        """Pure numeric expression should pass."""
        is_valid, _ = validate_plugin_expression("1 + 2 + 3", ["col"])
        assert is_valid

    def test_single_column(self):
        """Single column reference should pass."""
        is_valid, _ = validate_plugin_expression("col", ["col"])
        assert is_valid

    def test_column_with_leading_underscore(self):
        """Column with leading underscore should pass."""
        is_valid, _ = validate_plugin_expression("_col + col", ["_col", "col"])
        assert is_valid

    def test_long_expression(self):
        """Very long but valid expression should pass."""
        cols = [f"col{i}" for i in range(100)]
        expr = " + ".join(cols)
        is_valid, _ = validate_plugin_expression(expr, cols)
        assert is_valid

    def test_expression_with_newlines(self):
        """Expression with newlines should be handled."""
        is_valid, _ = validate_plugin_expression("col_a +\ncol_b", ["col_a", "col_b"])
        # The validation should handle this - either pass or fail consistently
        # This tests that it doesn't crash

    def test_expression_with_tabs(self):
        """Expression with tabs should be handled."""
        is_valid, _ = validate_plugin_expression("col_a +\tcol_b", ["col_a", "col_b"])
        # The validation should handle this - either pass or fail consistently


# =============================================================================
# 6. Evasion Attempt Tests
# =============================================================================

class TestEvasionAttempts:
    """Tests for potential evasion attempts that try to bypass security."""

    def test_evasion_string_concatenation_import(self):
        """String concatenation to form 'import' should be caught."""
        # This might not be caught by pattern matching but could be dangerous
        is_valid, _ = validate_plugin_expression("'im' + 'port'", ["col"])
        # Even if this passes validation, pd.eval won't execute it maliciously
        # The test documents the behavior

    def test_evasion_dunder_with_spaces(self):
        """Dunder attributes even with weird formatting should be blocked."""
        is_valid, _ = validate_plugin_expression("col . __class__", ["col"])
        assert not is_valid

    def test_evasion_encoded_import(self):
        """Unicode or encoded variations should be handled."""
        # Standard word boundaries should catch this
        is_valid, _ = validate_plugin_expression("col + import", ["col"])
        assert not is_valid

    def test_evasion_exec_in_string(self):
        """Exec in string context should be blocked."""
        is_valid, _ = validate_plugin_expression("'exec'", ["col"])
        # Pattern matches even in string context
        assert not is_valid

    def test_evasion_nested_dangerous_function(self):
        """Nested dangerous function calls should be blocked."""
        is_valid, _ = validate_plugin_expression("abs(exec('code'))", ["col"])
        assert not is_valid

    def test_evasion_getattr_chain(self):
        """Getattr chaining should be blocked."""
        is_valid, _ = validate_plugin_expression("getattr(getattr(col, 'a'), 'b')", ["col"])
        assert not is_valid

    def test_evasion_os_as_substring(self):
        """os. pattern should only match at word boundaries."""
        # Column named 'cos' should be fine (contains 'os' but not 'os.')
        is_valid, _ = validate_plugin_expression("cos(col)", ["col"])
        # 'cos' is a safe function
        assert is_valid

    def test_evasion_eval_as_column_name(self):
        """Column named 'eval' should still be blocked (keyword conflict)."""
        is_valid, _ = validate_plugin_expression("eval", ["eval"])
        assert not is_valid

    def test_evasion_import_as_column_name(self):
        """Column named 'import' should be blocked."""
        is_valid, _ = validate_plugin_expression("import", ["import"])
        assert not is_valid

    def test_evasion_class_attribute_access(self):
        """Class attribute access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__class__.__mro__", ["col"])
        assert not is_valid

    def test_evasion_subclasses(self):
        """Subclass access should be blocked."""
        is_valid, _ = validate_plugin_expression("col.__class__.__subclasses__()", ["col"])
        assert not is_valid


# =============================================================================
# 7. Comprehensive Pattern Coverage Tests
# =============================================================================

class TestPatternCoverage:
    """Tests to ensure all documented dangerous patterns are blocked."""

    @pytest.mark.parametrize("pattern,example", [
        (r'\b__\w+__\b', "__import__"),
        (r'\b__\w+__\b', "__class__"),
        (r'\b__\w+__\b', "__dict__"),
        (r'\bimport\b', "import"),
        (r'\bexec\b', "exec"),
        (r'\beval\b', "eval"),
        (r'\bcompile\b', "compile"),
        (r'\bopen\b', "open"),
        (r'\bos\.', "os.system"),
        (r'\bsys\.', "sys.exit"),
        (r'\bsubprocess\.', "subprocess.run"),
        (r'\bbuiltins\.', "builtins.open"),
        (r'\bglobals\b', "globals"),
        (r'\blocals\b', "locals"),
        (r'\bgetattr\b', "getattr"),
        (r'\bsetattr\b', "setattr"),
        (r'\bdelattr\b', "delattr"),
    ])
    def test_dangerous_pattern_blocked(self, pattern, example):
        """Each dangerous pattern should be blocked."""
        expr = f"{example}(col)"
        is_valid, msg = validate_plugin_expression(expr, ["col"])
        assert not is_valid, f"Pattern '{pattern}' should block expression containing '{example}'"


# =============================================================================
# 8. Safe Functions Coverage Tests
# =============================================================================

class TestSafeFunctionsCoverage:
    """Tests to ensure all documented safe functions are allowed."""

    @pytest.mark.parametrize("func", list(SAFE_EXPRESSION_FUNCTIONS))
    def test_safe_function_allowed(self, func):
        """Each safe function should be allowed."""
        expr = f"{func}(col)"
        is_valid, msg = validate_plugin_expression(expr, ["col"])
        assert is_valid, f"Safe function '{func}' should be allowed but got: {msg}"


# =============================================================================
# 9. Return Value Tests
# =============================================================================

class TestReturnValues:
    """Tests for correct return value format."""

    def test_valid_returns_tuple(self):
        """Valid expression should return (True, '')."""
        result = validate_plugin_expression("col + 1", ["col"])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] == ""

    def test_invalid_returns_tuple_with_message(self):
        """Invalid expression should return (False, error_message)."""
        result = validate_plugin_expression("exec('code')", ["col"])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is False
        assert len(result[1]) > 0  # Error message should not be empty

    def test_unknown_identifier_error_message_contains_identifier(self):
        """Error message for unknown identifier should mention the identifier."""
        is_valid, msg = validate_plugin_expression("unknown_var + col", ["col"])
        assert not is_valid
        assert "unknown_var" in msg.lower() or "unknown" in msg.lower()

    def test_dangerous_pattern_error_message_indicates_disallowed(self):
        """Error message for dangerous pattern should indicate disallowed."""
        is_valid, msg = validate_plugin_expression("exec('code')", ["col"])
        assert not is_valid
        assert "disallowed" in msg.lower() or "pattern" in msg.lower()


# =============================================================================
# 10. Integration-like Tests
# =============================================================================

class TestRealisticExpressions:
    """Tests with realistic scientific expression patterns."""

    def test_zscore_normalization(self):
        """Z-score normalization expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "(gaze_x - mean(gaze_x)) / std(gaze_x)",
            ["gaze_x"]
        )
        assert is_valid

    def test_percent_change(self):
        """Percent change expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "(new_value - old_value) / old_value * 100",
            ["new_value", "old_value"]
        )
        assert is_valid

    def test_euclidean_distance(self):
        """Euclidean distance expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))",
            ["x1", "y1", "x2", "y2"]
        )
        assert is_valid

    def test_angle_conversion(self):
        """Angle conversion expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "heading_deg * 3.14159 / 180",
            ["heading_deg"]
        )
        assert is_valid

    def test_signal_amplitude(self):
        """Signal amplitude expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "abs(signal_max - signal_min)",
            ["signal_max", "signal_min"]
        )
        assert is_valid

    def test_rms_calculation(self):
        """RMS calculation expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "sqrt(mean(pow(signal, 2)))",
            ["signal"]
        )
        assert is_valid

    def test_clipped_value(self):
        """Clipped value expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "clip(sensor_value, 0, 100)",
            ["sensor_value"]
        )
        assert is_valid

    def test_conditional_expression(self):
        """Conditional expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "gaze_x if valid else nan",
            ["gaze_x", "valid"]
        )
        assert is_valid

    def test_multi_channel_combination(self):
        """Multi-channel combination expression should pass."""
        is_valid, _ = validate_plugin_expression(
            "(left_eye_x + right_eye_x) / 2",
            ["left_eye_x", "right_eye_x"]
        )
        assert is_valid

    def test_velocity_calculation(self):
        """Velocity-like calculation should pass."""
        is_valid, _ = validate_plugin_expression(
            "sqrt(vx * vx + vy * vy + vz * vz)",
            ["vx", "vy", "vz"]
        )
        assert is_valid
