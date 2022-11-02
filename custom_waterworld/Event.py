from typing import Callable, Optional, TypeVar

TArgs = TypeVar("TArgs")
TResult = TypeVar("TResult")
CallbackType = Callable[[TArgs], TResult]


class Event:
    __slots__ = ("_callbacks", "callback_type")

    def __init__(self, callback_type: Optional[CallbackType] = None):
        self._callbacks = set()
        self.callback_type = callback_type

    def __call__(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)

    def subscribe(self, callback: CallbackType):
        # I'd like to change this to determine that the callback is of the correct type
        if self.callback_type is not None and not callable(callback):
            raise TypeError(
                f"Expected callback to be of type {self.callback_type},"
                f" got {type(callback)}"
            )
        self._callbacks.add(callback)

    def unsubscribe(self, callback: CallbackType):
        self._callbacks.remove(callback)

    def clear(self):
        self._callbacks.clear()

    def __iadd__(self, other):
        self.subscribe(other)
        return self

    def __isub__(self, other):
        self.unsubscribe(other)
        return self

    def __len__(self):
        return len(self._callbacks)
