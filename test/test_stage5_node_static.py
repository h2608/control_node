#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static checks over ``stage5_node.py``.

``stage5_node.py`` cannot be imported without ``rclpy``, so none of the pure
unit tests reach it — roughly seven thousand lines of state machine with no
automated guard at all.  Everything here works on the parsed AST instead, which
needs no ROS and runs in milliseconds.

This exists because of a measured cost.  A one-line ``self.p5_send_velocity_
command(vx=0.0, vy=0.0, wz=0.0)`` — missing the required ``step_height`` —
compiled cleanly, passed all 150 unit tests, and then killed the node on the
first run of a sixteen-run batch that had already been restarted twice.  The
whole class of error is "an internal call that does not match its own
definition", and the AST can see all of it.
"""

import ast
import os

import pytest

SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'control_node', 'stage5_node.py')


def _tree():
    with open(SOURCE, encoding='utf-8') as handle:
        return ast.parse(handle.read(), filename=SOURCE)


def _methods(tree):
    """Return {name: FunctionDef} for every method of every class in the file."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[item.name] = item
    return out


def _decorators(func):
    return {d.id for d in func.decorator_list if isinstance(d, ast.Name)}


def _arity(func):
    """Return (min_positional, max_positional_or_None) as seen by the caller.

    The implicit first parameter is dropped for plain methods and for
    ``classmethod`` (``cls`` is bound the same way ``self`` is) but not for
    ``staticmethod``, which has none.  Getting this wrong produces a false
    positive on every classmethod in the file, and a static check that reports
    phantom problems is worse than no check.
    """
    args = func.args
    positional = list(getattr(args, 'posonlyargs', [])) + list(args.args)
    if 'staticmethod' not in _decorators(func) and positional:
        positional = positional[1:]              # drop self / cls
    required = len(positional) - len(args.defaults)
    if args.vararg is not None:
        return max(0, required), None
    return max(0, required), len(positional)


def _self_calls(tree):
    """Yield (call_node, method_name) for every ``self.method(...)`` call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == 'self'):
            yield node, func.attr


def test_every_internal_call_satisfies_its_own_signature():
    """No ``self.foo(...)`` may omit an argument ``def foo`` requires.

    Keyword-only arguments and ``**kwargs`` call sites are skipped rather than
    guessed at; the check is deliberately conservative, because a static test
    that cries wolf gets switched off and then catches nothing at all.
    """
    tree = _tree()
    methods = _methods(tree)
    problems = []
    for call, name in _self_calls(tree):
        func = methods.get(name)
        if func is None:                        # inherited or dynamic
            continue
        if any(k.arg is None for k in call.keywords):        # **kwargs
            continue
        if any(isinstance(a, ast.Starred) for a in call.args):
            continue
        required, maximum = _arity(func)
        supplied = len(call.args) + len({k.arg for k in call.keywords})
        if supplied < required:
            problems.append(
                'line %d: self.%s(...) supplies %d of %d required args'
                % (call.lineno, name, supplied, required))
        elif maximum is not None and len(call.args) > maximum:
            problems.append(
                'line %d: self.%s(...) supplies %d positional args, max %d'
                % (call.lineno, name, len(call.args), maximum))
    assert not problems, '\n'.join(problems)


def test_velocity_commands_always_declare_a_step_height():
    """The specific regression: a stop that forgets the gait's foot lift.

    Kept alongside the general check because this one call is on the control
    path for every state, and a stop command with the wrong foot lift is a
    different failure from one that will not run at all.
    """
    tree = _tree()
    missing = []
    for call, name in _self_calls(tree):
        if name != 'p5_send_velocity_command':
            continue
        keywords = {k.arg for k in call.keywords if k.arg}
        if len(call.args) < 4 and 'step_height' not in keywords:
            missing.append(call.lineno)
    assert not missing, 'no step_height at lines %r' % (missing,)


def test_the_checker_would_actually_catch_the_regression():
    """Guard the guard: a test that cannot fail proves nothing.

    Feeds the checker the exact shape of the bug it was written for and
    asserts it complains.
    """
    tree = ast.parse(
        'class N:\n'
        '    def send(self, vx, vy, wz, step_height):\n'
        '        pass\n'
        '    def caller(self):\n'
        '        self.send(vx=0.0, vy=0.0, wz=0.0)\n')
    methods = _methods(tree)
    required, _ = _arity(methods['send'])
    call = next(c for c, n in _self_calls(tree) if n == 'send')
    supplied = len(call.args) + len({k.arg for k in call.keywords})
    assert required == 4
    assert supplied < required


if __name__ == '__main__':
    pytest.main([__file__])
