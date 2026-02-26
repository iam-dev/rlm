"""Security hardening tests for LocalREPL.

Covers:
- Safe import allowlist enforcement
- Restricted open() sandbox enforcement
- Blocked dangerous builtins (eval, exec, compile, input)
- Thread-safe output capture via contextlib.redirect_stdout
"""

import inspect
import io
import os
import sys

import pytest

from rlm.environments.local_repl import (
    _ALLOWED_MODULES,
    LocalREPL,
    _make_restricted_open,
    _safe_import,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repl():
    """Provide a fresh LocalREPL instance, cleaned up after each test."""
    env = LocalREPL()
    yield env
    env.cleanup()


# =============================================================================
# 1. Safe imports
# =============================================================================


class TestSafeImports:
    """_safe_import enforces the _ALLOWED_MODULES allowlist."""

    # --- Allowed modules ---

    @pytest.mark.parametrize(
        "module_name",
        [
            "math",
            "json",
            "re",
            "collections",
            "itertools",
            "datetime",
            "random",
            "copy",
        ],
    )
    def test_allowed_module_imports_without_error(self, repl, module_name):
        """Importing an allowed module must succeed with no stderr."""
        result = repl.execute_code(f"import {module_name}")
        assert result.stderr == "", (
            f"Expected no error importing '{module_name}', got: {result.stderr}"
        )

    def test_math_functions_usable_after_import(self, repl):
        """math module is fully functional after import."""
        result = repl.execute_code("import math\nx = math.sqrt(16)")
        assert result.stderr == ""
        assert repl.locals["x"] == 4.0

    def test_json_module_usable_after_import(self, repl):
        """json module encodes and decodes correctly after import."""
        result = repl.execute_code(
            "import json\nencoded = json.dumps({'a': 1})\ndecoded = json.loads(encoded)"
        )
        assert result.stderr == ""
        assert repl.locals["decoded"] == {"a": 1}

    def test_re_module_usable_after_import(self, repl):
        """re module matches patterns correctly after import."""
        code = "import re\nm = re.search(r'\\d+', 'abc123')\nfound = m.group()"
        result = repl.execute_code(code)
        assert result.stderr == ""
        assert repl.locals["found"] == "123"

    def test_collections_module_usable_after_import(self, repl):
        """collections.Counter works after import."""
        result = repl.execute_code("from collections import Counter\ncounts = Counter('aabb')")
        assert result.stderr == ""
        assert repl.locals["counts"]["a"] == 2

    def test_itertools_module_usable_after_import(self, repl):
        """itertools.chain works after import."""
        result = repl.execute_code("import itertools\nresult = list(itertools.chain([1, 2], [3]))")
        assert result.stderr == ""
        assert repl.locals["result"] == [1, 2, 3]

    def test_datetime_module_usable_after_import(self, repl):
        """datetime.datetime constructor works without error after import.

        Note: datetime.date.today() internally imports 'time' which is not on
        the allowlist, so we test with a fixed-value constructor instead.
        """
        result = repl.execute_code("import datetime\nd = datetime.datetime(2024, 1, 1)")
        assert result.stderr == ""
        import datetime as dt

        assert isinstance(repl.locals["d"], dt.datetime)
        assert repl.locals["d"].year == 2024

    def test_random_module_usable_after_import(self, repl):
        """random.randint returns an integer in the expected range after import."""
        result = repl.execute_code("import random\nrandom.seed(0)\nv = random.randint(1, 10)")
        assert result.stderr == ""
        assert 1 <= repl.locals["v"] <= 10

    def test_copy_module_usable_after_import(self, repl):
        """copy.deepcopy produces an independent copy after import."""
        result = repl.execute_code("import copy\norig = [1, 2, 3]\ncloned = copy.deepcopy(orig)")
        assert result.stderr == ""
        assert repl.locals["cloned"] == [1, 2, 3]

    # --- Disallowed modules ---

    @pytest.mark.parametrize(
        "module_name",
        [
            "os",
            "sys",
            "subprocess",
            "socket",
            "shutil",
            "pathlib",
        ],
    )
    def test_disallowed_module_raises_import_error(self, repl, module_name):
        """Importing a disallowed module must produce an ImportError in stderr."""
        result = repl.execute_code(f"import {module_name}")
        assert "ImportError" in result.stderr, (
            f"Expected ImportError for '{module_name}', got stderr: {result.stderr!r}"
        )

    def test_disallowed_os_import_message_mentions_module(self, repl):
        """The ImportError message for 'os' specifically names the blocked module."""
        result = repl.execute_code("import os")
        assert "os" in result.stderr

    def test_disallowed_subprocess_blocked_in_from_import(self, repl):
        """'from subprocess import run' is also blocked."""
        result = repl.execute_code("from subprocess import run")
        assert "ImportError" in result.stderr

    def test_safe_import_directly_raises_import_error(self):
        """_safe_import raises ImportError for blocked modules when called directly."""
        with pytest.raises(ImportError, match="os"):
            _safe_import("os")

    def test_safe_import_directly_allows_math(self):
        """_safe_import returns the math module when called directly."""
        import math as stdlib_math

        result = _safe_import("math")
        assert result is stdlib_math

    def test_allowed_modules_frozenset_contains_expected_entries(self):
        """_ALLOWED_MODULES includes the documented safe set."""
        expected = {
            "math",
            "json",
            "re",
            "collections",
            "itertools",
            "datetime",
            "random",
            "copy",
        }
        assert expected.issubset(_ALLOWED_MODULES), (
            f"Missing from _ALLOWED_MODULES: {expected - _ALLOWED_MODULES}"
        )

    def test_disallowed_modules_absent_from_allowlist(self):
        """Dangerous modules are absent from _ALLOWED_MODULES."""
        dangerous = {"os", "sys", "subprocess", "socket", "shutil", "pathlib"}
        present = dangerous & _ALLOWED_MODULES
        assert not present, f"Dangerous modules found in allowlist: {present}"


# =============================================================================
# 2. Restricted open()
# =============================================================================


class TestRestrictedOpen:
    """_make_restricted_open enforces sandbox-directory-only file access."""

    def test_write_and_read_file_within_temp_dir(self, repl):
        """open() can create and read a file inside temp_dir."""
        result = repl.execute_code(
            "with open('hello.txt', 'w') as f:\n"
            "    f.write('hello')\n"
            "with open('hello.txt', 'r') as f:\n"
            "    content = f.read()"
        )
        assert result.stderr == "", f"Unexpected error: {result.stderr}"
        assert repl.locals["content"] == "hello"

    def test_open_absolute_path_outside_temp_dir_raises_permission_error(self, repl):
        """open() with an absolute path outside temp_dir must raise PermissionError."""
        result = repl.execute_code("f = open('/tmp/malicious.txt', 'w')")
        assert "PermissionError" in result.stderr, (
            f"Expected PermissionError, got: {result.stderr!r}"
        )

    def test_open_etc_passwd_raises_permission_error(self, repl):
        """open('/etc/passwd') must be denied."""
        result = repl.execute_code("f = open('/etc/passwd', 'r')")
        assert "PermissionError" in result.stderr

    def test_path_traversal_attack_raises_permission_error(self, repl):
        """open('../../etc/passwd') path-traversal must be denied."""
        result = repl.execute_code("f = open('../../etc/passwd', 'r')")
        assert "PermissionError" in result.stderr

    def test_path_traversal_via_dotdot_sequence_denied(self, repl):
        """A path that escapes temp_dir via multiple .. segments is denied."""
        result = repl.execute_code("f = open('../../../etc/hosts', 'r')")
        assert "PermissionError" in result.stderr

    def test_absolute_path_inside_temp_dir_is_allowed(self, repl):
        """An absolute path explicitly inside temp_dir is permitted."""
        temp_dir = repl.temp_dir
        target = os.path.join(temp_dir, "allowed.txt")
        result = repl.execute_code(
            f"with open(r'{target}', 'w') as f:\n"
            "    f.write('ok')\n"
            f"with open(r'{target}', 'r') as f:\n"
            "    v = f.read()"
        )
        assert result.stderr == ""
        assert repl.locals["v"] == "ok"

    def test_restricted_open_factory_directly_denies_outside_path(self):
        """_make_restricted_open's returned callable raises PermissionError directly."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            restricted = _make_restricted_open(td)
            with pytest.raises(PermissionError):
                restricted("/etc/passwd", "r")

    def test_restricted_open_factory_directly_allows_inside_path(self):
        """_make_restricted_open's returned callable allows a path inside temp_dir."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            restricted = _make_restricted_open(td)
            target = os.path.join(td, "test.txt")
            with restricted(target, "w") as fh:
                fh.write("data")
            with restricted(target, "r") as fh:
                assert fh.read() == "data"

    def test_file_written_in_code_persists_to_read_in_next_execution(self, repl):
        """Files written in one execute_code call are readable in the next."""
        repl.execute_code("with open('state.txt', 'w') as f:\n    f.write('persistent')")
        result = repl.execute_code("with open('state.txt', 'r') as f:\n    data = f.read()")
        assert result.stderr == ""
        assert repl.locals["data"] == "persistent"


# =============================================================================
# 3. Blocked builtins
# =============================================================================


class TestBlockedBuiltins:
    """eval, exec, compile, and input are set to None in _SAFE_BUILTINS."""

    def test_eval_is_none_in_builtins(self, repl):
        """eval is None so calling it raises TypeError (not callable)."""
        result = repl.execute_code("result = eval('1+1')")
        # eval is None; calling None() raises TypeError
        assert "TypeError" in result.stderr or "NoneType" in result.stderr, (
            f"Expected TypeError from calling None, got: {result.stderr!r}"
        )

    def test_exec_is_none_in_builtins(self, repl):
        """exec is None so calling it raises TypeError."""
        result = repl.execute_code("exec('x=1')")
        assert "TypeError" in result.stderr or "NoneType" in result.stderr, (
            f"Expected TypeError from calling None, got: {result.stderr!r}"
        )

    def test_compile_is_none_in_builtins(self, repl):
        """compile is None so calling it raises TypeError."""
        result = repl.execute_code("compile('x=1', '<string>', 'exec')")
        assert "TypeError" in result.stderr or "NoneType" in result.stderr, (
            f"Expected TypeError from calling None, got: {result.stderr!r}"
        )

    def test_input_is_none_in_builtins(self, repl):
        """input is None so calling it raises TypeError."""
        result = repl.execute_code("input('prompt: ')")
        assert "TypeError" in result.stderr or "NoneType" in result.stderr, (
            f"Expected TypeError from calling None, got: {result.stderr!r}"
        )

    def test_eval_name_resolves_to_none(self, repl):
        """The name 'eval' resolves to None inside the sandbox."""
        result = repl.execute_code("v = eval")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_exec_name_resolves_to_none(self, repl):
        """The name 'exec' resolves to None inside the sandbox."""
        result = repl.execute_code("v = exec")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_compile_name_resolves_to_none(self, repl):
        """The name 'compile' resolves to None inside the sandbox."""
        result = repl.execute_code("v = compile")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_input_name_resolves_to_none(self, repl):
        """The name 'input' resolves to None inside the sandbox."""
        result = repl.execute_code("v = input")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_globals_name_resolves_to_none(self, repl):
        """globals() is also blocked — set to None in safe builtins."""
        result = repl.execute_code("v = globals")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_locals_name_resolves_to_none(self, repl):
        """locals() is also blocked — set to None in safe builtins."""
        result = repl.execute_code("v = locals")
        assert result.stderr == ""
        assert repl.locals["v"] is None

    def test_safe_builtins_remain_accessible_after_blocking(self, repl):
        """Blocking dangerous builtins does not break safe ones like len and sum."""
        result = repl.execute_code("v = len([1, 2, 3]) + sum([4, 5])")
        assert result.stderr == ""
        assert repl.locals["v"] == 12

    def test_blocked_builtins_cannot_be_used_for_dynamic_code_execution(self, repl):
        """Attempting dynamic code execution via eval, exec, or compile all fail."""
        for snippet in [
            "eval('1+1')",
            "exec('x=1')",
            "compile('x=1', '<string>', 'exec')",
        ]:
            result = repl.execute_code(snippet)
            assert result.stderr != "", f"Expected an error for: {snippet!r}, but stderr was empty"


# =============================================================================
# 4. Output capture thread safety
# =============================================================================


class TestOutputCaptureThreadSafety:
    """_capture_output uses contextlib.redirect_stdout, not direct sys.stdout mutation."""

    def test_capture_output_does_not_mutate_sys_stdout_after_context_exits(self, repl):
        """sys.stdout identity must be unchanged after _capture_output context exits."""
        original_stdout = sys.stdout
        with repl._capture_output():
            pass
        assert sys.stdout is original_stdout, (
            "sys.stdout was not restored after _capture_output context exit"
        )

    def test_capture_output_redirects_stdout_inside_block_to_buffer(self, repl):
        """Inside _capture_output, sys.stdout is the yielded buffer; it is restored on exit."""
        original_stdout = sys.stdout
        with repl._capture_output() as (stdout_buf, _stderr_buf):
            inside_stdout = sys.stdout
        # Restored after exit
        assert sys.stdout is original_stdout, "sys.stdout not restored after _capture_output exit"
        # Inside the block, sys.stdout was re-pointed to the buffer by redirect_stdout
        assert inside_stdout is stdout_buf

    def test_capture_output_source_uses_redirect_stdout(self):
        """Inspect the source of _capture_output to confirm contextlib.redirect_stdout usage."""
        source = inspect.getsource(LocalREPL._capture_output)
        assert "redirect_stdout" in source, "_capture_output must use contextlib.redirect_stdout"
        assert "redirect_stderr" in source, "_capture_output must use contextlib.redirect_stderr"

    def test_capture_output_source_does_not_directly_assign_sys_stdout(self):
        """_capture_output must not contain 'sys.stdout =' (direct mutation)."""
        source = inspect.getsource(LocalREPL._capture_output)
        assert "sys.stdout =" not in source, "_capture_output must not directly assign sys.stdout"
        assert "sys.stderr =" not in source, "_capture_output must not directly assign sys.stderr"

    def test_sequential_executions_in_separate_repls_capture_correct_output(self):
        """Separate REPL instances run sequentially each capture only their own output.

        contextlib.redirect_stdout modifies process-global sys.stdout, so it is NOT
        safe for concurrent threads running simultaneously — only one redirect can be
        active at a time. This test verifies the sequential (non-overlapping) case,
        which is the contract LocalREPL provides.
        """
        for i in range(5):
            env = LocalREPL()
            try:
                res = env.execute_code(f"print('instance-{i}')")
                assert res.stdout.strip() == f"instance-{i}", (
                    f"Instance {i} captured wrong output: {res.stdout!r}"
                )
            finally:
                env.cleanup()

    def test_stdout_not_leaked_between_sequential_executions(self, repl):
        """Output from one execute_code call does not bleed into the next."""
        repl.execute_code("print('first')")
        result = repl.execute_code("print('second')")
        assert "first" not in result.stdout
        assert "second" in result.stdout

    def test_capture_output_yields_string_io_buffers(self, repl):
        """_capture_output yields two StringIO buffers: (stdout, stderr)."""
        with repl._capture_output() as (stdout_buf, stderr_buf):
            assert isinstance(stdout_buf, io.StringIO)
            assert isinstance(stderr_buf, io.StringIO)

    def test_capture_output_collects_print_into_buffer(self, repl):
        """Output printed inside the context appears in the yielded stdout buffer."""
        with repl._capture_output() as (stdout_buf, _):
            print("captured")
        assert stdout_buf.getvalue() == "captured\n"
