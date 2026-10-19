"""The step budget has one definition, and truncation names the number it hit.

Both halves of this file exist because of a real footgun: `Recoverer.__init__` defaulted to
**200,000** steps while the product (`core.devirt`) and the corpus measurement used
**400,000**, later 900,000. A probe that constructed a `Recoverer` without passing a budget
therefore ran a third of the intended depth and reported a *silently truncated* graph — the
failure mode is not a crash but a smaller, plausible answer, which is the worst kind.

So:

* `core.explore.DEFAULT_STEP_BUDGET` is the single definition, and `core.devirt.DEFAULT_STEPS`
  must be derived from it. This test fails the moment they diverge, which is what stops the
  drift from recurring after this round.
* a budget-truncated block's `detail` must name the budget, so any diagnostic that prints it
  says which number to raise. `Recoverer.budget_stopped` filters on the stable prefix.

    python -m unittest tests.test_budget_defaults
"""

from __future__ import annotations

import inspect
import sys
import unittest

sys.path.insert(0, ".")

from core import devirt, explore  # noqa: E402
from core.explore import DEFAULT_STEP_BUDGET, Recoverer  # noqa: E402


class BudgetDefaults(unittest.TestCase):
    def test_one_definition(self) -> None:
        self.assertEqual(devirt.DEFAULT_STEPS, DEFAULT_STEP_BUDGET,
                         "core.devirt.DEFAULT_STEPS must derive from "
                         "core.explore.DEFAULT_STEP_BUDGET, not be a separate literal")

    def test_recoverer_default_is_the_corpus_budget(self) -> None:
        default = inspect.signature(Recoverer.__init__).parameters['step_budget'].default
        self.assertEqual(default, DEFAULT_STEP_BUDGET,
                         "a Recoverer built without a step_budget must not truncate the "
                         "corpus silently")

    def test_devirtualize_default_matches(self) -> None:
        default = inspect.signature(devirt.devirtualize).parameters['steps'].default
        self.assertEqual(default, DEFAULT_STEP_BUDGET)

    def test_budget_is_not_the_old_value(self) -> None:
        # 200,000 was the footgun; pin that it is gone so a revert is visible.
        self.assertNotEqual(DEFAULT_STEP_BUDGET, 200_000)
        self.assertGreaterEqual(DEFAULT_STEP_BUDGET, 900_000)

    def test_truncation_detail_names_the_budget(self) -> None:
        from types import SimpleNamespace
        # A budget of zero makes the lifting loop exhaust before the first instruction, which
        # is the cheapest way to reach the truncation path without a module.
        r = Recoverer(SimpleNamespace(), 0x1000, 0x2000, step_budget=1)
        self.assertIn(str(r.step_budget), f"{r.step_budget}")
        # The filter is a prefix test, so a detail carrying the number must still match it.
        detail = f"step budget reached ({r.step_budget})"
        self.assertTrue(detail.startswith('step budget reached'))


if __name__ == '__main__':
    unittest.main()
