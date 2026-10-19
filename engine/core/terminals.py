"""Typed execution boundaries; no terminal is a normal guest return."""


def from_stop(stop):
    info = stop.info
    terminal = dict(kind=stop.kind, site=stop.site)
    if 'code' in info:
        terminal['code'] = info['code']
    event = info.get('event', {})
    if event:
        terminal['args'] = dict(event.get('args', ()))
        terminal['stack_args'] = dict(event.get('stack_args', ()))
    for name in ('callee', 'reason', 'source'):
        if name in info:
            terminal[name] = info[name]
    return terminal


def compile_terminal(compiler, block):
    native = getattr(block, 'terminal', {})
    if not native:
        return {}
    result = dict(native)
    if 'code' in result:
        result['code'] = compiler.expr(result['code'])
    for name in ('args', 'stack_args'):
        if name in result:
            result[name] = {k: compiler.expr(v) for k, v in result[name].items()}
    return result


def roots(terminal):
    result = [terminal['code']] if 'code' in terminal else []
    for name in ('args', 'stack_args'):
        result.extend(terminal.get(name, {}).values())
    return result
