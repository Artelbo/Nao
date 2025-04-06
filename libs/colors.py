class Options:
    RESET = '\x1b[0m'
    BRIGHT = '\x1b[1m'
    DARK = '\x1b[2m'
    UNDERSCORE = '\x1b[4m'
    BLINK = '\x1b[5m'
    REVERSE = '\x1b[7m'
    HIDDEN = '\x1b[8m'


class OPS(Options):
    pass


class ForegroundColors:
    BLACK = '\x1b[30m'
    GRAY = '\x1b[90m'
    LIGHT_GRAY = '\x1b[37m'
    WHITE = '\x1b[97m'

    DARK_RED = '\x1b[31m'
    LIGHT_RED = '\x1b[91m'

    DARK_GREEN = '\x1b[32m'
    LIGHT_GREEN = '\x1b[92m'

    DARK_YELLOW = '\x1b[33m'
    LIGHT_YELLOW = '\x1b[93m'

    DARK_BLUE = '\x1b[34m'
    LIGHT_BLUE = '\x1b[94m'

    DARK_MAGENTA = '\x1b[35m'
    LIGHT_MAGENTA = '\x1b[95m'

    DARK_CYAN = '\x1b[36m'
    LIGHT_CYAN = '\x1b[96m'


class FC(ForegroundColors):
    pass


class BackgroundColors:
    BLACK = '\x1b[40m'
    GRAY = '\x1b[100m'
    LIGHT_GRAY = '\x1b[47m'
    WHITE = '\x1b[107m'

    DARK_RED = '\x1b[41m'
    LIGHT_RED = '\x1b[101m'

    DARK_GREEN = '\x1b[42m'
    LIGHT_GREEN = '\x1b[102m'

    DARK_YELLOW = '\x1b[43m'
    LIGHT_YELLOW = '\x1b[103m'

    DARK_BLUE = '\x1b[44m'
    LIGHT_BLUE = '\x1b[104m'

    DARK_MAGENTA = '\x1b[45m'
    LIGHT_MAGENTA = '\x1b[105m'

    DARK_CYAN = '\x1b[46m'
    LIGHT_CYAN = '\x1b[106m'


class BC(BackgroundColors):
    pass
