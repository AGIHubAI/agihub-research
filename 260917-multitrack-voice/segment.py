"""
Cut a token stream into sentences as soon as they are complete.

The LLM yields fragments, roughly a word at a time. Waiting for the whole answer
before speaking is what makes the agent slow, so sentences are emitted the
moment they are provably finished.

The rule that makes this safe is to hold back the last sentence. A boundary is
only trusted once at least one further token has arrived past it, so the
segmenter re-decides on a longer string each time and never commits to a
boundary it might revise. `Dr.` is not flushed early because when the boundary
is considered it is still the final sentence of the buffer.

Segmentation is spaCy's rule-based `sentencizer` on a blank English pipeline: no
model weights, ~0.2 ms per parse, and it inherits spaCy's tokenizer exceptions
so abbreviations, decimals and `e.g.` are already handled. Kokoro's own spaCy
pipeline cannot be reused — misaki loads it with only `tok2vec` and `tagger`
enabled, so it has neither a parser nor a senter.

The failure mode is under-segmentation, which only delays audio and never emits
a truncated sentence. The one case that must not stall is a model ignoring the
"no lists" instruction: list items carry no terminal punctuation, so the buffer
would grow without a boundary forever. `max_chars` bounds that.
"""

from __future__ import annotations

import re

# Fall back to these if no sentence boundary appears within max_chars, in order
# of preference. A clause break is a better place to cut than a word boundary.
_FALLBACK = re.compile(r"[,;:]\s|\n")


class Segmenter:
    """
    Feed fragments in, take complete sentences out.

    Not thread-safe; drive it from the one thread that reads the token stream.
    """

    def __init__(self, max_chars: int = 200):
        # Imported lazily: spaCy costs ~860 ms on first import and a caller
        # that never segments should not pay it.
        import spacy

        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")
        self._max_chars = max_chars
        self._buffer = ""

    def feed(self, fragment: str) -> list[str]:
        """Add a fragment; return whatever sentences are now complete."""
        if not fragment:
            return []

        self._buffer += fragment
        out: list[str] = []

        while True:
            sentence = self._take()
            if sentence is None:
                break
            out.append(sentence)

        return out

    def flush(self) -> list[str]:
        """
        End of stream: emit whatever is left.

        The held-back final sentence lives here, so this must be called or the
        last sentence of every answer is silently dropped.
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []

    # -- internals --------------------------------------------------------

    def _take(self) -> str | None:
        """Pop one provably-complete sentence, or None if none is ready."""
        if not self._buffer.strip():
            return None

        sentences = list(self._nlp(self._buffer).sents)

        # More than one sentence means everything before the last is finished:
        # a further token has already arrived past that boundary.
        if len(sentences) > 1:
            first = sentences[0].text_with_ws
            self._buffer = self._buffer[len(first):]
            return first.strip() or None

        # One sentence and no boundary in sight. Normally we wait, but a model
        # emitting a list never produces terminal punctuation at all.
        if len(self._buffer) >= self._max_chars:
            return self._force()

        return None

    def _force(self) -> str | None:
        """Cut an over-long buffer at the last clause break, else at a space."""
        window = self._buffer[: self._max_chars]

        cut = max((m.end() for m in _FALLBACK.finditer(window)), default=0)
        if cut == 0:
            cut = window.rfind(" ") + 1
        if cut <= 0:
            cut = self._max_chars

        head, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return head.strip() or None


def split(text: str, max_chars: int = 200) -> list[str]:
    """Segment a complete string. The same code path a stream takes."""
    segmenter = Segmenter(max_chars=max_chars)
    out = segmenter.feed(text)
    return out + segmenter.flush()
