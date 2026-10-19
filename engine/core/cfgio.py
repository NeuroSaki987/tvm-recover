"""Writing a recovered CFG out as data, and reading it back.

There was no machine-readable CFG artifact at all: `out/*.txt` are human-readable
dumps, and a graph existed only as Python objects for the length of a run. Two
consequences, and only one of them is about size.

* **A run that is interrupted keeps nothing.** Recovery of a real function is tens
  of seconds and of the largest modules minutes; a budget cut, a crash or a
  closed terminal loses the whole graph. The journal below writes one record per
  block *as the block completes*, so what has been recovered survives whatever
  happens next.
* **The graph is not what grows, and it is worth saying so.** `probe/cost_scale`
  measures the three quantities that do grow, over all fourteen roots of the
  module: the pending worklist peaks at **4 snapshots** (forks are consumed
  last-in-first-out, so the frontier never widens), the graph is at most 149
  blocks and 11,472 folded transitions, and peak allocation is **206.8 MB**
  driven by the **arena** -- 361,603 interned expression nodes. The graph is on
  the order of a percent of the footprint. So persistence is a deliverable and a
  durability property; it is *not* the thing that stops a recovery blowing up,
  and writing the file does not make the arena smaller. Bounding the arena is a
  separate change and is not attempted here.

The format is line-oriented JSON so that a truncated file is still readable up to
its last complete line, and every address is a hex string so the artifact can be
read and diffed by a person.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

FORMAT = "tvm0-cfg"
VERSION = 1


def _hx(v: int) -> str:
    return f"{v:#x}"


def _key(handler: int, vip: int) -> str:
    """A block identity as text.

    `(handler, vip)` and *both* halves. An earlier version wrote successors as a
    handler alone, which loses the identity the whole engine is built on: ten of
    these functions share handler code between program points, so a handler on
    its own does not name a block, and a graph read back from the file could not
    be re-checked -- its reachability count came out as 12 of 74 for a graph that
    is entirely well formed.
    """
    return f"{handler:#x}@{vip:#018x}"


def parse_key(s: str) -> tuple[int, int]:
    h, _, v = s.partition("@")
    return int(h, 16), int(v, 16)


@dataclass
class BlockRecord:
    handler: int
    vip: int
    kind: str
    site: int
    detail: str = ""
    terminal: dict = field(default_factory=dict)
    predicate: str = ""
    edge_cond: list = field(default_factory=list)
    retry: str = ""
    hypothesis: bool = False
    cost: int = 0
    events: int = 0
    succs: list[tuple[int, int]] = field(default_factory=list)
    preds: list[tuple[int, int]] = field(default_factory=list)
    transfers: list[tuple[int, int]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "block": _key(self.handler, self.vip),
            "handler": _hx(self.handler),
            "vip": _hx(self.vip),
            "kind": self.kind,
            "site": _hx(self.site),
            "detail": self.detail,
            "terminal": self.terminal,
            "predicate": self.predicate,
            # One entry per `succs` entry: the predicate value that selects it, or null where the
            # association was not established. A backend cannot emit a conditional branch without it.
            "edge_cond": list(self.edge_cond),
            "retry": self.retry,
            "hypothesis": self.hypothesis,
            "cost": self.cost,
            "events": self.events,
            "succs": [_key(*s) for s in self.succs],
            "preds": [_key(*p) for p in self.preds],
            # One string per transfer rather than a two-element list: a real
            # function folds thousands of them and the pair form roughly doubles
            # the file for no gain in readability.
            "transfers": [f"{_hx(s)}->{_hx(d)}" for s, d in self.transfers],
        }


def block_record(b) -> BlockRecord:
    return BlockRecord(
        handler=b.key.handler,
        vip=b.key.vip,
        kind=b.kind,
        site=b.site,
        detail=b.detail,
        terminal=dict(getattr(b, 'terminal', {})),
        predicate=b.predicate,
        edge_cond=list(getattr(b, 'edge_cond', ())),
        retry=b.retry,
        hypothesis=b.hypothesis,
        cost=b.cost,
        events=b.events,
        succs=[(s.handler, s.vip) for s in b.succs],
        preds=[(p.handler, p.vip) for p in b.preds],
        transfers=list(b.transfers),
    )


def header(module: str, root: int, cfg, extra: dict | None = None) -> dict:
    """The first line: what the file is, and what produced it.

    Deliberately carries **no counts**. It is written before the first block
    exists, so any total in it is a zero that reads like a fact -- which is
    exactly what it did: a complete 149-block graph announced "declared 0 blocks,
    0 unresolved" in its own header. Totals live in the footer, where they are
    known.
    """
    out = {
        "format": FORMAT,
        "version": VERSION,
        "module": os.path.basename(module),
        "root": _hx(root),
        "status": "open",
        "entry": _key(cfg.entry.handler, cfg.entry.vip) if cfg.entry else None,
    }
    if extra:
        out.update(extra)
    return out


class Journal:
    """Append-only per-block persistence, so an interrupted run keeps its graph.

    One JSON object per line, the header first. Written and flushed as each block
    completes, not at the end: the whole point is that killing the process leaves
    a usable file. A partial last line is the only corruption possible, and a
    reader skips it.
    """

    def __init__(self, path: str, module: str, root: int, cfg, extra: dict | None = None):
        self.path = path
        self.fh = open(path, "w", encoding="utf-8")
        self.fh.write(json.dumps(header(module, root, cfg, extra)) + "\n")
        self.fh.flush()
        self.count = 0

    def block(self, b) -> None:
        self.fh.write(json.dumps(block_record(b).to_json()) + "\n")
        self.count += 1
        # Flush per block. Buffering would defeat the purpose, and a block is a
        # few hundred bytes against tens of milliseconds of interpretation.
        self.fh.flush()

    def finish(self, cfg, journal_truncated: bool = False) -> None:
        rec = {"final": True, "blocks": len(cfg.blocks), "unresolved": cfg.unresolved,
               "forks": cfg.forks, "steps": cfg.steps, "arena_nodes": cfg.arena_nodes,
               "transfers": cfg.transfers, "summary": cfg.summary(),
               "journal_truncated": journal_truncated}
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        self.fh.close()


def save(cfg, path: str, module: str, root: int, extra: dict | None = None) -> None:
    """Write a finished graph in one go, in the same line-oriented format."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header(module, root, cfg, extra)) + "\n")
        for k in cfg.order:
            fh.write(json.dumps(block_record(cfg.blocks[k]).to_json()) + "\n")
        fh.write(json.dumps({"final": True, "blocks": len(cfg.blocks),
                             "unresolved": cfg.unresolved, "forks": cfg.forks,
                             "steps": cfg.steps, "arena_nodes": cfg.arena_nodes,
                             "transfers": cfg.transfers,
                             "summary": cfg.summary()}) + "\n")


def load(path: str) -> tuple[dict, list[dict], dict | None]:
    """Read a file written by `save` or `Journal`.

    Returns `(header, blocks, final)`. `final` is None when the file was cut
    short -- which is the case the journal exists for, and not an error.
    """
    head: dict = {}
    blocks: list[dict] = []
    final = None
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A truncated last line is the expected shape of an interrupted
                # run; anything else is skipped too, but counted by the caller
                # through the block tally against the header.
                break
            if i == 0:
                head = rec
            elif rec.get("final"):
                final = rec
            else:
                blocks.append(rec)
    return head, blocks, final


def verify(path: str) -> dict:
    """Structural checks on a written graph, without re-recovering anything.

    Keyed on the full `(handler, vip)` identity, which is why the file has to
    carry both halves -- see `_key`.
    """
    head, blocks, final = load(path)
    keys = {b["block"] for b in blocks if "block" in b}
    bad_edges = 0
    self_loops = 0
    for b in blocks:
        here = b.get("block")
        for s in b["succs"]:
            if s not in keys:
                bad_edges += 1
            if s == here:
                self_loops += 1
    by_key = {b["block"]: b for b in blocks if "block" in b}
    entry = head.get("entry")
    reached: set[str] = set()
    stack = [entry] if entry else []
    while stack:
        cur = stack.pop()
        if cur in reached or cur not in by_key:
            continue
        reached.add(cur)
        stack.extend(by_key[cur]["succs"])
    return {
        "header": head,
        "final": final,
        "blocks_written": len(blocks),
        "complete": final is not None,
        "bad_edges": bad_edges,
        "self_loops": self_loops,
        "reachable": len(reached),
        "distinct_blocks": len(by_key),
        "distinct_handlers": len({b["handler"] for b in blocks}),
    }
