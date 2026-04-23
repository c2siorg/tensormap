"""Unit tests for retry decorator."""

from unittest.mock import patch

import pytest

from app.utils.retry import retry


class TestRetryDecorator:
    def test_succeeds_on_first_attempt(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def succeed_once():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_once()
        assert result == "success"
        assert call_count == 1

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient failure")
            return "success"

        result = fail_twice_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_stops_after_max_retries(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            always_fails()
        assert call_count == 3

    @patch("app.utils.retry.time.sleep")
    def test_no_real_sleeping(self, mock_sleep):
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient failure")
            return "success"

        result = fail_twice_then_succeed()
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("app.utils.retry.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        call_count = 0

        @retry(max_attempts=4, delay=1.0, backoff=2.0)
        def fail_three_times():
            nonlocal call_count
            call_count += 1
            raise ValueError("Transient failure")

        with pytest.raises(ValueError):
            fail_three_times()

        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0
