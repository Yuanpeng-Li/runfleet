"""[WP1] Comment-preserving edits of config.toml (tomlkit): add / remove / reorder [[host]] tables, set [general] /
[notify] keys.

Settings (§7.13) and the auto-remember of hosts named on the command line (§8.4) change the user's own file, so every
edit keeps comments, blank lines, key order and formatting; only the touched tables change. Each operation is one
read-modify-write of the file as it is *now* (an edit made in $EDITOR meanwhile is never lost), validated with the
same parser as startup before anything is written, and written atomically (temp file in the same directory, fsync,
rename; the file mode is kept and a symlinked config is written through to its target). Two RunFleet processes never
interleave a read-modify-write: each operation holds an flock on a sidecar `.config.toml.lock` next to the file
(present only while held). A file that is not valid TOML is never rewritten, and every failure (unreadable or
unwritable file, read-only file system, disk full, a lock held too long) raises ConfigError with the file untouched.

Comments above a `[[host]]` header belong to that host: tomlkit stores them as the trailing lines of whatever comes
before the header, so moving or removing a host moves or removes "its" comment block with it, and the comment lines
after the last host (a commented-out example) stay at the end of the file. Two things stay in place instead: the blank
lines between tables (by position, so a file that starts with `[[host]]` keeps starting with it and every table keeps
its separator), and a section heading above the first host (comment lines up to the last `# ---`/`# ===`-style
marker), which stays above whichever host is first.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.container import Container, OutOfOrderTableProxy
from tomlkit.items import AoT, Comment, InlineTable, Item, Table, Whitespace
from tomlkit.toml_document import TOMLDocument

from runfleet.config import (GENERAL_SCHEMA, LOCAL_NAMES, NOTIFY_SCHEMA, UI_SCHEMA, HostConfig, Reader, dump_hosts_toml,
                             format_duration, host_table_toml, parse_config, parse_duration)
from runfleet.errors import ConfigError

_Body = list[tuple[Any, Item]]
LOCK_TIMEOUT = 2.0          # seconds to wait for another RunFleet's edit (each holds the lock for milliseconds)


# ======================================================================================= file level
class ConfigFile:
    """One config.toml; every method re-reads the file, so concurrent edits by hand are preserved."""

    def __init__(self, path: Path) -> None:
        self.path = path

    # ------------------------------------------------------------ read
    def read_text(self) -> str | None:
        try:
            with self.path.open(encoding="utf-8", newline="") as f:     # raw: CRLF files must stay CRLF
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise ConfigError("", f"{self.path}: {e.strerror}") from None

    def document(self) -> TOMLDocument:
        """The parsed file; a missing file is the commented template new configs start from. A layout tomlkit
        cannot reproduce byte for byte (e.g. [[host]] tables split by other tables, which it regroups) is refused:
        an edit must never reformat the rest of the user's file."""
        text = self.read_text()
        doc = parse_document(dump_hosts_toml(()) if text is None else text, self.path)
        if text is not None and tomlkit.dumps(doc) != text:
            raise ConfigError("", f"{self.path}: RunFleet cannot edit this layout without reformatting it "
                                  "(are the [[host]] tables split by other tables?); edit it by hand")
        return doc

    def hosts(self) -> list[HostConfig]:
        """Configured hosts in file order (validated like at startup); [] when the file does not exist."""
        text = self.read_text()
        if text is None:
            return []
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("", f"{self.path}: invalid TOML: {e}") from None
        cfg, _warnings = parse_config(data, source=self.path)
        return list(cfg.hosts)

    def host_keys(self, name: str) -> list[str]:
        """Keys of the saved [[host]] table `name` other than `name` / `ssh` (hand-tuned settings); [] if absent."""
        text = self.read_text()
        if text is None:
            return []
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("", f"{self.path}: invalid TOML: {e}") from None
        hosts = data.get("host")
        for table in hosts if isinstance(hosts, list) else []:
            if isinstance(table, dict) and table.get("name") == name:
                return [k for k in table if k not in ("name", "ssh")]
        return []

    # ------------------------------------------------------------ operations (each one locked read-modify-write)
    def add_hosts(self, entries: Sequence[HostConfig], restore: Mapping[str, str] | None = None) -> list[str]:
        """Append a minimal [[host]] table for every entry whose name (or ssh alias) is not configured yet; returns
        the names written. `restore`: name -> text returned by an earlier `remove_host`, written back instead of the
        minimal table (same name and ssh target only), so an uncheck + check keeps the host's own settings."""
        with self._lock():
            doc = self.document()
            added = add_host_tables(doc, entries, restore)
            if added:
                self.write(doc)
            return added

    def remove_host(self, name: str) -> str | None:
        """Remove the [[host]] table `name`; returns its text (comment block + table, for `add_hosts(restore=)`) or
        None when it is not in the file."""
        with self._lock():
            doc = self.document()
            removed = remove_host_table(doc, name)
            if removed is None:
                return None
            self.write(doc)
            return removed

    def reorder_hosts(self, names: Sequence[str]) -> bool:
        with self._lock():
            doc = self.document()
            if not reorder_host_tables(doc, names):
                return False
            self.write(doc)
            return True

    def set_general(self, values: Mapping[str, object]) -> bool:
        return self.set_table("general", values)

    def set_table(self, name: str, values: Mapping[str, object]) -> bool:
        """Set keys of the top-level table `name` ([general], [notify], …); False (nothing written) when unchanged."""
        with self._lock():
            doc = self.document()
            if not set_table_keys(doc, name, values):
                return False
            self.write(doc)
            return True

    # ------------------------------------------------------------ write
    def write(self, doc: TOMLDocument) -> None:
        text = tomlkit.dumps(doc)
        old = self.read_text() or ""
        if "\r\n" in old and "\n" not in old.replace("\r\n", ""):      # a CRLF file stays CRLF
            text = text.replace("\r\n", "\n").replace("\n", "\r\n")
        validate_text(text, self.path)
        try:
            atomic_write(self.path, text)
        except OSError as e:            # read-only file system / dotfiles dir, permission denied, disk full
            raise ConfigError("", f"{self.path}: cannot write: {e.strerror or e}") from None

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """flock on `.<name>.lock` next to the (symlink-resolved) file around one read-modify-write. The file itself
        cannot be locked: the atomic rename swaps its inode. The lock file exists only while held (unlinked before
        the lock is released; a waiter that then holds a stale inode retries). Without a writable directory there is
        no lock, and the write that follows reports the real problem; any other lock failure is an error."""
        import fcntl

        target = self.path.resolve() if self.path.is_symlink() else self.path
        lock = target.parent / f".{target.name}.lock"
        deadline = time.monotonic() + LOCK_TIMEOUT
        fd: int | None = None
        while fd is None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as e:
                if os.access(target.parent, os.W_OK):       # a writable directory but no lock: never edit unlocked
                    raise ConfigError("", f"{self.path}: cannot lock: {e.strerror or e}") from None
                break
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.fstat(fd).st_ino == os.stat(lock).st_ino:
                    break                                   # ours, and still the file at that path
            except (BlockingIOError, FileNotFoundError):
                pass
            os.close(fd)
            fd = None
            if time.monotonic() >= deadline:
                raise ConfigError("", f"{self.path}: another RunFleet is saving it — try again")
            time.sleep(0.02)
        try:
            yield
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.unlink(lock)
                os.close(fd)                                # closing releases the lock


def parse_document(text: str, path: Path | None = None) -> TOMLDocument:
    try:
        return tomlkit.parse(text)
    except Exception as e:          # tomlkit raises its own ParseError subclasses (and occasionally ValueError)
        where = f"{path}: " if path is not None else ""
        raise ConfigError("", f"{where}invalid TOML: {e}") from None


def validate_text(text: str, path: Path | None = None) -> None:
    """The edited document must load exactly like at startup; otherwise nothing is written."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:    # pragma: no cover - tomlkit output is valid TOML
        raise ConfigError("", f"{path}: edit would produce invalid TOML: {e}") from None
    parse_config(data, source=path)


def atomic_write(path: Path, text: str) -> None:
    """Replace `path` with `text`: temp file next to the real target, fsync, rename. Keeps the mode of an existing
    file (new files: 0666 minus the umask, like any editor) and writes through a symlink instead of replacing it."""
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o7777
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ======================================================================================= document level
def _is_trivia(item: Item) -> bool:
    return isinstance(item, (Comment, Whitespace))


def _hosts_aot(doc: TOMLDocument) -> AoT | None:
    occurrences = [item for key, item in doc.body if key is not None and key.key == "host"]
    if not occurrences:
        return None
    if len(occurrences) > 1:
        raise ConfigError("host", "the [[host]] tables are split by other tables; keep them together to edit them "
                                  "from RunFleet")
    aot = occurrences[0]
    if not isinstance(aot, AoT):
        raise ConfigError("host", "hosts are not written as [[host]] tables; edit this file by hand")
    return aot


def _host_name(table: Table) -> str:
    name = table.get("name")
    return str(name) if name is not None else ""


def _split_trailer(body: _Body) -> tuple[_Body, _Body]:
    """(content, trailing comments/blank lines) of a table body."""
    i = len(body)
    while i > 0 and body[i - 1][0] is None and _is_trivia(body[i - 1][1]):
        i -= 1
    return body[:i], body[i:]


def _set_trailer(container: Container, trailer: _Body) -> None:
    body = container.body
    content, _old = _split_trailer(body)
    del body[len(content):]         # trailing trivia sit after every key: no index in container._map moves
    body.extend(trailer)


class _Head:
    """Where the comment block above the first [[host]] header lives: the trailing lines of the deepest last table
    before it, or top-level lines of the document itself."""

    def __init__(self, doc: TOMLDocument) -> None:
        self.doc = doc
        self.index = _host_index(doc)
        before = [item for _key, item in doc.body[: self.index] if not isinstance(item, tomlkit.items.Null)]
        node = before[-1] if before else None
        self.container: Container | None = None
        while isinstance(node, (Table, AoT)):     # descend into [a] -> [a.b] -> ...: the last table holds the lines
            table = node.body[-1] if isinstance(node, AoT) else node
            self.container = table.value
            last = self.container.body[-1][1] if self.container.body else None
            node = last if isinstance(last, (Table, AoT)) else None

    def _start(self) -> int:
        i = self.index
        while i > 0 and self.doc.body[i - 1][0] is None and _is_trivia(self.doc.body[i - 1][1]):
            i -= 1
        return i

    def trailer(self) -> _Body:
        if self.container is not None:
            return list(_split_trailer(self.container.body)[1])
        return list(self.doc.body[self._start(): self.index])

    def set(self, trailer: _Body) -> None:
        if self.container is not None:
            _set_trailer(self.container, trailer)
            return
        start = self._start()
        shift = len(trailer) - (self.index - start)
        self.doc.body[start: self.index] = trailer
        if shift:                   # keys at or after the [[host]] array moved: keep tomlkit's index map in step
            m = self.doc._map
            for k, v in list(m.items()):
                if isinstance(v, tuple):
                    m[k] = tuple(i + shift if i >= self.index else i for i in v)
                elif v >= self.index:
                    m[k] = v + shift
            self.index += shift


def _host_index(doc: TOMLDocument) -> int:
    return next(i for i, (key, _item) in enumerate(doc.body) if key is not None and key.key == "host")


def _split_lead(trailer: _Body) -> tuple[_Body, _Body]:
    """(rest, lead) of trailing lines: the lead is the comment run directly above the next header (with the blank
    line before it); comment blocks separated from the header by a blank line belong to what is above."""
    j = len(trailer)
    while j > 0 and isinstance(trailer[j - 1][1], Comment):
        j -= 1
    if j == len(trailer):                     # ends in blank lines: only those separate the next header
        while j > 0 and isinstance(trailer[j - 1][1], Whitespace):
            j -= 1
    elif j > 0 and isinstance(trailer[j - 1][1], Whitespace):
        j -= 1
    return trailer[:j], trailer[j:]


def _split_sep(lead: _Body) -> tuple[_Body, _Body]:
    """(blank lines, comments) of a lead."""
    i = 0
    while i < len(lead) and isinstance(lead[i][1], Whitespace):
        i += 1
    return lead[:i], lead[i:]


_MARKER = re.compile(r"^\s*#\s*[-=*~#_]{3,}")


def _split_heading(comments: _Body) -> tuple[_Body, _Body]:
    """(section heading, the host's own comments): the heading runs up to the last `# ---`-style marker line."""
    last = max((i for i, (_k, item) in enumerate(comments)
                if isinstance(item, Comment) and _MARKER.match(item.as_string())), default=-1)
    return comments[: last + 1], comments[last + 1:]


@dataclass
class _Block:
    lead: _Body                 # comment block directly above the [[host]] header (moves with the host)
    table: Table
    rest: _Body                 # the table's own trailing lines that are not the next header's lead


@dataclass
class _Layout:
    """What stays in place while hosts move: the head, the separators by position, the heading, the file tail."""
    head: _Head
    head_rest: _Body            # the head's own lines (not part of the first host's lead)
    heading: _Body              # section heading above the first host (stays above whichever host is first)
    seps: list[_Body]           # blank lines before host i's comment block (slot i, not host i)
    tail: _Body                 # lines after the last host (a commented-out example)


def _blocks(doc: TOMLDocument, aot: AoT) -> tuple[list[_Block], _Layout]:
    head = _Head(doc)
    head_rest, lead = _split_lead(head.trailer())
    blocks: list[_Block] = []
    seps: list[_Body] = []
    heading: _Body = []
    for i, table in enumerate(aot.body):
        sep, comments = _split_sep(lead)
        if i == 0:
            heading, comments = _split_heading(comments)
        seps.append(sep)
        rest, lead = _split_lead(list(_split_trailer(table.value.body)[1]))
        blocks.append(_Block(comments, table, rest))
    return blocks, _Layout(head, head_rest, heading, seps, lead)


def _rebuild(doc: TOMLDocument, aot: AoT, blocks: list[_Block], layout: _Layout, seps: list[_Body]) -> None:
    """Write `blocks` back in this order; slot i gets `seps[i]` (len(seps) == len(blocks))."""
    if not blocks:
        heading = (layout.seps[0] if layout.seps else []) + layout.heading if layout.heading else []
        layout.head.set(layout.head_rest + heading + layout.tail)
        doc.remove("host")
        return
    layout.head.set(layout.head_rest + seps[0] + layout.heading + blocks[0].lead)
    leads = [sep + b.lead for sep, b in zip(seps[1:], blocks[1:], strict=True)]
    for block, nxt in zip(blocks, [*leads, layout.tail], strict=True):
        _set_trailer(block.table.value, block.rest + nxt)
        content, _trailer = _split_trailer(block.table.value.body)
        if content and "\n" not in content[-1][1].trivia.trail:    # last line of a file without a final newline
            content[-1][1].trivia.trail += "\n"
    tables = [b.table for b in blocks]
    del aot[:]
    for table in tables:
        aot.append(table)


def configured_names(doc: TOMLDocument) -> list[str]:
    aot = _hosts_aot(doc)
    return [_host_name(t) for t in aot.body] if aot is not None else []


def _ssh_of(table: Table) -> str:
    ssh = table.get("ssh")
    if ssh is not None:
        return str(ssh)
    name = _host_name(table)
    return "local" if name in LOCAL_NAMES else name


def add_host_tables(doc: TOMLDocument, entries: Sequence[HostConfig],
                    restore: Mapping[str, str] | None = None) -> list[str]:
    """Append minimal tables (`name`, plus `ssh` when it differs) after the last host, before the file tail; an entry
    in `restore` (text from `remove_host_table`, same name and ssh) comes back with its keys and comments."""
    aot = _hosts_aot(doc)
    present = set() if aot is None else {_host_name(t) for t in aot.body} | {_ssh_of(t) for t in aot.body}
    new: list[_Block] = []
    names: list[str] = []
    for e in entries:
        if e.name in present or e.ssh in present and e.ssh != "local":
            continue
        present |= {e.name, e.ssh}
        block = _restored((restore or {}).get(e.name), e)
        new.append(block or _Block([], _parse_table(host_table_toml(HostConfig(name=e.name, ssh=e.ssh))), []))
        names.append(e.name)
    if not new:
        return []
    if aot is None:                           # the first host: after a blank line, its comment block above it
        _ensure_blank_line(doc)
        aot = AoT([], parsed=True)            # "parsed": tomlkit adds no blank lines of its own on append
        aot.append(new[0].table)
        doc.append("host", aot)
        if new[0].lead:
            head = _Head(doc)
            head.set(head.trailer() + new[0].lead)
        new = new[1:]
        if not new:
            return names
    blocks, layout = _blocks(doc, aot)
    _rebuild(doc, aot, blocks + new, layout, layout.seps + [[(None, tomlkit.nl())] for _ in new])
    return names


def _restored(text: str | None, entry: HostConfig) -> _Block | None:
    if text is None:
        return None
    try:
        doc = tomlkit.parse(text)
        table = doc["host"][0]
    except Exception:               # not what remove_host_table wrote: fall back to a minimal table
        return None
    if not isinstance(table, Table) or _host_name(table) != entry.name or _ssh_of(table) != entry.ssh:
        return None
    lead: _Body = [(k, item) for k, item in doc.body if k is None and isinstance(item, Comment)]
    return _Block(lead, table, _own_trailer(_split_trailer(table.value.body)[1]))


def _parse_table(text: str) -> Table:
    table = tomlkit.parse(text)["host"][0]
    assert isinstance(table, Table)
    _set_trailer(table.value, [])           # separators come from the layout, not from the snippet
    return table


def _ensure_blank_line(doc: TOMLDocument) -> None:
    text = tomlkit.dumps(doc)
    if text and not text.endswith("\n\n"):
        doc.add(tomlkit.nl() if text.endswith("\n") else tomlkit.ws("\n\n"))


def remove_host_table(doc: TOMLDocument, name: str) -> str | None:
    """Remove host `name` with its comment block; returns their text (for `add_host_tables(restore=)`), None when
    absent. The separator slots stay: removing the first host keeps the first slot's blank lines."""
    aot = _hosts_aot(doc)
    if aot is None:
        return None
    blocks, layout = _blocks(doc, aot)
    index = next((i for i, b in enumerate(blocks) if _host_name(b.table) == name), None)
    if index is None:
        return None
    removed = blocks[index]
    seps = list(layout.seps)
    del seps[index if index > 0 or len(seps) == 1 else 1]
    _rebuild(doc, aot, blocks[:index] + blocks[index + 1:], layout, seps)
    return _block_text(removed)


def _own_trailer(rest: _Body) -> _Body:
    """A host's own trailing comments (a note under its keys) without the blank lines that separate the next host."""
    own = list(rest)
    while own and isinstance(own[-1][1], Whitespace):
        own.pop()
    return own


def _block_text(block: _Block) -> str:
    _set_trailer(block.table.value, _own_trailer(block.rest))
    one = tomlkit.document()
    aot = tomlkit.aot()
    aot.append(block.table)
    one.append("host", aot)
    lead = "".join(item.as_string() for _k, item in block.lead if isinstance(item, Comment))
    return lead + tomlkit.dumps(one)


def reorder_host_tables(doc: TOMLDocument, names: Sequence[str]) -> bool:
    """Hosts named in `names` first, in that order; the others keep their relative order after them."""
    aot = _hosts_aot(doc)
    if aot is None:
        return False
    blocks, layout = _blocks(doc, aot)
    rank = {n: i for i, n in enumerate(dict.fromkeys(names))}
    order = sorted(range(len(blocks)), key=lambda i: (rank.get(_host_name(blocks[i].table), len(rank)), i))
    if order == list(range(len(blocks))):
        return False
    _rebuild(doc, aot, [blocks[i] for i in order], layout, layout.seps)
    return True


TABLE_SCHEMAS: Mapping[str, Mapping[str, Reader]] = {"general": GENERAL_SCHEMA, "ui": UI_SCHEMA,
                                                     "notify": NOTIFY_SCHEMA}


def table_value(table: str, key: str, value: object) -> Any:
    """Python value of a settings field -> what the file holds (durations as "2h" strings, tuples as lists)."""
    reader = TABLE_SCHEMAS.get(table, {}).get(key)
    if reader is None:
        raise ConfigError(f"{table}.{key}", "unknown setting")
    if reader is parse_duration and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = format_duration(float(value))
    if isinstance(value, tuple):
        value = list(value)
    reader(value, f"{table}.{key}")         # raises ConfigError for a wrong type
    return value


def general_value(key: str, value: object) -> Any:
    """`table_value` of a GeneralSettings field."""
    return table_value("general", key, value)


def set_general_keys(doc: TOMLDocument, values: Mapping[str, object]) -> bool:
    """`set_table_keys` of [general]."""
    return set_table_keys(doc, "general", values)


def set_table_keys(doc: TOMLDocument, name: str, values: Mapping[str, object]) -> bool:
    """Set keys of the top-level table `name` in place (the inline comment of a replaced key stays); False when every
    value is already there. A missing table is created with just these keys: [general] before the first table of the
    file, any other one above the [[host]] tables (and the comment block of the first host), else at the end."""
    if not values:
        return False
    converted = {k: table_value(name, k, v) for k, v in values.items()}
    current = doc.get(name)
    if current is None:
        table = tomlkit.table()
        for k, v in converted.items():
            table.add(k, v)
        if name == "general":
            _insert_first(doc, name, table)
        else:
            _insert_before_hosts(doc, name, table)
        return True
    if not isinstance(current, (Table, InlineTable, OutOfOrderTableProxy)):
        raise ConfigError(name, "expected a table")
    changed = False
    # new keys go after the last key, above the comments of what follows (a split table: of whichever part gets them)
    parts: list[Table] = []
    if isinstance(current, Table):
        parts = [current]
    elif isinstance(current, OutOfOrderTableProxy):
        parts = [t for t in getattr(current, "_tables", ()) if isinstance(t, Table)]
    trailers: list[_Body] = []
    for part in parts:
        trailers.append(list(_split_trailer(part.value.body)[1]))
        _set_trailer(part.value, [])
    for k, v in converted.items():
        if k in current and _same(current[k], v):
            continue
        _fragment(current, k)[k] = v
        changed = True
    for part, trailer in zip(parts, trailers, strict=True):
        part.value.body.extend(trailer)
    return changed


def _fragment(table: Table | InlineTable | OutOfOrderTableProxy, key: str) -> Any:
    """Where `key` is set in place. A table split over several places (`notify.a = 1` twice at the top level, or
    `[notify]` plus a later `[notify.x]`) is a proxy whose own `__setitem__` moves an existing key to its end."""
    if isinstance(table, OutOfOrderTableProxy):
        for part in getattr(table, "_tables", ()):
            if key in part:
                return part
    return table


def _insert_first(doc: TOMLDocument, name: str, table: Table) -> None:
    """Before the first table, above the comment block that introduces it, unless that block starts the file (then
    it is the file's header)."""
    table.add(tomlkit.nl())
    index = next((i for i, (key, item) in enumerate(doc.body) if key is not None and isinstance(item, (Table, AoT))),
                 None)
    if index is None:
        _ensure_blank_line(doc)
        doc.append(name, table)
        return
    start = index
    while start and doc.body[start - 1][0] is None and isinstance(doc.body[start - 1][1], Comment):
        start -= 1
    doc._insert_at(start if start > 0 else index, name, table)


def _insert_before_hosts(doc: TOMLDocument, name: str, table: Table) -> None:
    """Above the [[host]] tables: the comment block directly above the first host (tomlkit keeps it as the trailing
    lines of whatever precedes the header) moves below the new table, so it stays with its host."""
    if _hosts_aot(doc) is None:
        _ensure_blank_line(doc)
        doc.append(name, table)
        return
    head = _Head(doc)
    rest, lead = _split_lead(head.trailer())
    blanks, comments = _split_sep(lead)
    if not blanks and (rest or head.container is not None):     # never glued to what is above
        blanks = [(None, tomlkit.nl())]
    head.set(rest + blanks)
    doc._insert_at(head.index, name, table)
    _set_trailer(table.value, [(None, tomlkit.nl()), *comments])


def _same(current: Any, value: object) -> bool:
    plain = current.unwrap() if hasattr(current, "unwrap") else current
    return type(plain) is type(value) and plain == value
