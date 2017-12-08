import time
import __main__ as main

__all__ = ['ContextTimer', 'ProgressBar']


class ContextTimer:
    """Clocks using python context to meassure execution time"""
    @property
    def duration(self) -> float:
        """Time elapsed from last context enter

        :return: Current elapsed time if still running,
            or elapsed time at the moment of context exit if not running
        """
        if self._running:
            return time.process_time() - self._begin

        return self._end - self._begin

    def __init__(self) -> None:
        """Creates timer, timer is not running after creation"""
        self._begin = 0
        self._end = 0
        self._running = False

    def __enter__(self) -> 'ContextTimer':
        """Resets timer and starts it"""
        self._running = True
        self._begin = time.process_time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Captures elapsed time and stops measuring"""
        self._end = time.process_time()
        self._running = False


# TODO better outputting - with \r
# TODO USE logging
class ProgressBar:
    def __init__(self,
                 lower_bound: float = 0.0,
                 upper_bound: float = 1.0,
                 bar_width: int = 60,
                 bar_char: str = '=',
                 header: str = 'Progress',
                 footer: str ='Finished!',
                 verbosity: int = 1):
        self.width = bar_width
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.header = header
        self.footer = footer
        self.bar = 0
        self.value = lower_bound
        self.bar_char = bar_char
        self.verbosity = verbosity
        self.is_tracking = False
        self.is_interactive = not hasattr(main, '__file__')

    def __enter__(self):
        if self.is_interactive:
            print(
                '| {header} {fill}|'.format(
                    header=self.header, fill=' ' * (self.width - len(self.header) - 4)),
                sep='')

        self.is_tracking = True
        self.value = 0
        self.bar = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self.set(self.upper_bound)

            if self.verbosity > 0:
                print(self.footer)

        if self.is_interactive:
            print()

        self.is_tracking = False


    def set(self, current):
        if not self.is_tracking:
            raise RuntimeError('Progress tracking permitted only inside context')

        current = current if current > self.lower_bound else self.lower_bound
        current = current if current < self.upper_bound else self.upper_bound

        # TODO maybe warn about over/underflow

        curr_progress = (current - self.lower_bound) / (self.upper_bound - self.lower_bound)
        target_bar = int(curr_progress * self.width)
        to_add = int(target_bar - self.bar)
        if to_add > 0:
            if self.verbosity > 0:
                if self.is_interactive:
                    print(self.bar_char * to_add, end='', flush=True)
                else:
                    print('\r| {header} | '.format(header=self.header),
                          self.bar_char * target_bar,
                          ' ' * (self.width - target_bar), ' | ',
                          sep='', end='', flush=True)

            self.bar = target_bar

        self.value = current
