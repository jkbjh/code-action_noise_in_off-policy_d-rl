def empty_to_none(converter):
    def empty_to_none_(arg):
        return converter(arg) if arg else None

    return empty_to_none_
