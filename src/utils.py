import platform

if platform == 'linux':
    import resource
else:
    import psutil

def get_memory_usage() -> float:
    if platform == 'linux':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        return psutil.Process().memory_info().rss

def print_elapsed(start: float, end: float) -> None:
    duration = end - start

    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    milliseconds = str(duration - int(duration))[2:6]
    print("\tRuntime:", f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds}")

def format_node_amount(n: int) -> str:
    if n < 1000000:
        return f'{int(n / 1000)}k'
    return f'{int(n / 1000000)}m'

def color_by_value(value, cap=10):
    value = min(max(value, 0), cap)
    r = value / cap
    color = [r, 0.0, 0.0]
    return color