"""mlfs: the library you build while working through ml-from-scratch.

Every topic that "owns" a module is listed in `mlfs/registry.toml`. Importing
`mlfs.<name>` loads that topic's exercise notebook (your code) or its reference solution.
See AUTHORING.md §6.
"""
from mlfs import registry as _registry

_registry.install()
