"""NiftyOne plugin discovery based on 'niftyone_{plugin_name}' naming convention."""

import importlib
import pkgutil

PLUGIN_PREFIX = "niftyone_"

# https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-naming-convention
PLUGINS = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules()
    if name.startswith(PLUGIN_PREFIX)
}
