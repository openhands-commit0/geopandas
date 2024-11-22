"""
Lightweight options machinery.

Based on https://github.com/topper-123/optioneer, but simplified (don't deal
with nested options, deprecated options, ..), just the attribute-style dict
like holding the options and giving a nice repr.
"""

def _validate_display_precision(value):
    if value is not None:
        if not isinstance(value, int):
            raise ValueError("display_precision must be an integer or None")
        if value < 0:
            raise ValueError("display_precision must be non-negative")

def _validate_io_engine(value):
    if value is not None and value not in ('pyogrio', 'fiona'):
        raise ValueError("io_engine must be one of 'pyogrio', 'fiona', or None")

def _warn_use_pygeos_deprecated(value):
    if value:
        warnings.warn(
            "The use_pygeos option is deprecated and will be removed in GeoPandas 1.1.",
            FutureWarning,
            stacklevel=2
        )
import textwrap
import warnings
from collections import namedtuple
Option = namedtuple('Option', 'key default_value doc validator callback')

class Options(object):
    """Provide attribute-style access to configuration dict."""

    def __init__(self, options):
        super().__setattr__('_options', options)
        config = {}
        for key, option in options.items():
            config[key] = option.default_value
        super().__setattr__('_config', config)

    def __setattr__(self, key, value):
        if key in self._config:
            option = self._options[key]
            if option.validator:
                option.validator(value)
            self._config[key] = value
            if option.callback:
                option.callback(key, value)
        else:
            msg = 'You can only set the value of existing options'
            raise AttributeError(msg)

    def __getattr__(self, key):
        try:
            return self._config[key]
        except KeyError:
            raise AttributeError('No such option')

    def __dir__(self):
        return list(self._config.keys())

    def __repr__(self):
        cls = self.__class__.__name__
        description = ''
        for key, option in self._options.items():
            descr = '{key}: {cur!r} [default: {default!r}]\n'.format(key=key, cur=self._config[key], default=option.default_value)
            description += descr
            if option.doc:
                doc_text = '\n'.join(textwrap.wrap(option.doc, width=70))
            else:
                doc_text = 'No description available.'
            doc_text = textwrap.indent(doc_text, prefix='    ')
            description += doc_text + '\n'
        space = '\n  '
        description = description.replace('\n', space)
        return '{}({}{})'.format(cls, space, description)
display_precision = Option(key='display_precision', default_value=None, doc='The precision (maximum number of decimals) of the coordinates in the WKT representation in the Series/DataFrame display. By default (None), it tries to infer and use 3 decimals for projected coordinates and 5 decimals for geographic coordinates.', validator=_validate_display_precision, callback=None)
io_engine = Option(key='io_engine', default_value=None, doc="The default engine for ``read_file`` and ``to_file``. Options are 'pyogrio' and 'fiona'.", validator=_validate_io_engine, callback=None)
use_pygeos = Option(key='use_pygeos', default_value=False, doc='Deprecated option previously used to enable PyGEOS. It will be removed in GeoPandas 1.1.', validator=_warn_use_pygeos_deprecated, callback=None)
options = Options({'display_precision': display_precision, 'use_pygeos': use_pygeos, 'io_engine': io_engine})