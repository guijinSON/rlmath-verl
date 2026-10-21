# recipe/

The upstream verl repo pins `recipe/` as a git submodule (verl-project/verl-recipe). This fork vendors it
instead, so the rlmath changes live in one repository:

| Directory | Origin |
|---|---|
| `agentic/` | [alibaba/verl-recipe](https://github.com/alibaba/verl-recipe) @ `40dd1924505b1a5adede8460c943211f855357b4` (Apache-2.0, see `LICENSE`), plus the rlmath fixes in later commits |
| `rlmath/`  | this fork: agentic GRPO for research-math Harbor tasks (see `rlmath/README.md`) |

The other recipes from verl-recipe are not vendored; install them separately if you need them.
