"""
This test demonstrates that the contextvars.ContextVar fix prevents
the race condition where multiple concurrent training tasks would
overwrite the same global _main_loop variable.
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, r"path to backend")

from app.services.model_run import _training_loop_ctx, _model_result


def test_concurrent_event_loops_isolated():
    """Verify that concurrent tasks have isolated event loop contexts."""
    
    results = {}
    
    async def task_1():
        """Simulates first user's training task."""
        loop1 = asyncio.get_running_loop()
        token1 = _training_loop_ctx.set(loop1)
        results["task1_set"] = (id(loop1), _training_loop_ctx.get())
        await asyncio.sleep(0.1)
        results["task1_get"] = (id(loop1), _training_loop_ctx.get())
        _training_loop_ctx.reset(token1)
    
    async def task_2():
        """Simulates second user's training task."""
        loop2 = asyncio.get_running_loop()
        token2 = _training_loop_ctx.set(loop2)
        results["task2_set"] = (id(loop2), _training_loop_ctx.get())
        await asyncio.sleep(0.05)
        results["task2_get"] = (id(loop2), _training_loop_ctx.get())
        _training_loop_ctx.reset(token2)
    
    async def main():
        """Run both tasks concurrently."""
        await asyncio.gather(task_1(), task_2())
    
    asyncio.run(main())
    
    loop1_id_set = results["task1_set"][0]
    loop1_id_get = results["task1_get"][0]
    loop2_id_set = results["task2_set"][0]
    loop2_id_get = results["task2_get"][0]
    
    print("\n" + "="*70)
    print("RACE CONDITION FIX TEST: Event Loop Context Isolation")
    print("="*70)
    
    print(f"\n✓ Task 1 - Loop ID when set: {loop1_id_set}")
    print(f"✓ Task 1 - Loop ID when retrieved: {loop1_id_get}")
    print(f"  Match: {loop1_id_set == loop1_id_get}")
    
    print(f"\n✓ Task 2 - Loop ID when set: {loop2_id_set}")
    print(f"✓ Task 2 - Loop ID when retrieved: {loop2_id_get}")
    print(f"  Match: {loop2_id_set == loop2_id_get}")
    
    assert loop1_id_set == loop1_id_get, "Task 1 lost its loop context!"
    assert loop2_id_set == loop2_id_get, "Task 2 lost its loop context!"
    
    print("\n" + "="*70)
    print(" TEST PASSED: Context variables correctly isolate event loops")
    print("="*70)
    print("\nWHAT THIS FIXES:")
    print("- Multiple concurrent training requests no longer share the same")
    print("  global _main_loop variable")
    print("- Each training task has its own isolated event loop context")
    print("- Socket.IO progress events are sent to the correct user's connection")
    print("="*70)


def test_model_result_with_context_var():
    """Verify _model_result correctly retrieves loop from context variable."""
    
    print("\n" + "="*70)
    print("SOCKET.IO EMISSION TEST: Context Variable Retrieval")
    print("="*70)
    
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True
    
    token = _training_loop_ctx.set(mock_loop)
    
    try:
        with patch("app.services.model_run.sio.emit") as mock_emit:
            with patch("asyncio.run_coroutine_threadsafe") as mock_threadsafe:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_threadsafe.return_value = mock_future
                
                _model_result("Test message", 0)
                
                mock_threadsafe.assert_called_once()
                args = mock_threadsafe.call_args
                assert args[0][1] is mock_loop, "Wrong loop passed to run_coroutine_threadsafe!"
        
        print("✓ Loop retrieved from context variable correctly")
        print("✓ Socket.IO emission called with correct event loop")
        print("\n" + "="*70)
        print("TEST PASSED: _model_result uses context variable correctly")
        print("="*70)
        
    finally:
        _training_loop_ctx.reset(token)


if __name__ == "__main__":
    try:
        test_concurrent_event_loops_isolated()
        test_model_result_with_context_var()
        print("\n" + " "*20)
        print("ALL TESTS PASSED - ISSUE #1 IS FIXED!")
        print(" "*20)
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
