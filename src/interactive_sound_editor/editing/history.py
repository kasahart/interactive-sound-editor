"""Application-level immutable edit history."""
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class EditHistory(Generic[T]):
    past: tuple[T, ...]
    current: T
    future: tuple[T, ...] = ()

    def commit(self, state: T) -> "EditHistory[T]":
        return EditHistory(self.past + (self.current,), state, ())

    def undo(self) -> "EditHistory[T]":
        if not self.past:
            return self
        return EditHistory(self.past[:-1], self.past[-1], (self.current,) + self.future)

    def redo(self) -> "EditHistory[T]":
        if not self.future:
            return self
        return EditHistory(self.past + (self.current,), self.future[0], self.future[1:])
