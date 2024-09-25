def is_subdict(small: dict, big: dict) -> bool:
    for key, value in small.items():
        if key not in big or big[key] != value:
            return False
    return True
