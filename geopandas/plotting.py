import warnings
from packaging.version import Version
import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.plotting import PlotAccessor
import geopandas
from ._decorator import doc

def _sanitize_geoms(geoms, prefix='Multi'):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out.  Maintains 1:1 matching of geometry to value.

    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    components = []
    component_index = []

    if isinstance(geoms, (pd.Series, pd.DataFrame)):
        geoms_index = np.arange(len(geoms))
    else:
        geoms_index = np.arange(len(list(geoms)))

    for geom, idx in zip(geoms, geoms_index):
        if geom is None or geom.is_empty:
            continue
        if prefix == 'Multi' and geom.type.startswith('Multi'):
            for component in geom.geoms:
                components.append(component)
                component_index.append(idx)
        elif prefix == 'Geom' and geom.type == 'GeometryCollection':
            for component in geom.geoms:
                components.append(component)
                component_index.append(idx)
        else:
            components.append(geom)
            component_index.append(idx)

    return components, np.array(component_index)

def _expand_kwargs(kwargs, multiindex):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    for key, value in kwargs.items():
        if isinstance(value, (np.ndarray, pd.Series)):
            if value.shape[0] != len(multiindex):
                raise ValueError(
                    f"Length of {key} sequence must match length of geometries"
                )
            kwargs[key] = value.take(multiindex)
        elif isinstance(value, list):
            if len(value) != len(multiindex):
                raise ValueError(
                    f"Length of {key} sequence must match length of geometries"
                )
            kwargs[key] = np.array(value).take(multiindex)

def _PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a Polygon geometry

    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes)::

        b = shapely.geometry.Point(0, 0).buffer(1.0)
        patch = _PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
        ax.add_patch(patch)

    GeoPandas originally relied on the descartes package by Sean Gillies
    (BSD license, https://pypi.org/project/descartes) for PolygonPatch, but
    this dependency was removed in favor of the below matplotlib code.
    """
    try:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by asShape() have
        # coordinates in the form [(x, y), (x, y), ...]
        vertices = np.concatenate(
            [np.asarray(polygon.exterior.coords)[:, :2]]
            + [np.asarray(r.coords)[:, :2] for r in polygon.interiors]
        )
        codes = np.concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors]
        )
        return Path(vertices, codes)

    path = pathify(polygon)
    return PathPatch(path, **kwargs)

def _plot_polygon_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, autolim=True, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs.
    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons
    facecolor : single color or sequence of `N` colors
        Color to fill the polygons. Cannot be used together with `values`.
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    try:
        from matplotlib.collections import PatchCollection
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    geoms, multiindex = _sanitize_geoms(geoms)
    if not geoms:
        return None

    # Process values and colors
    if values is not None:
        if color is not None:
            warnings.warn("'color' keyword is ignored when using 'values'")
        if cmap is None:
            cmap = "viridis"
        if isinstance(values, pd.Series):
            values = values.values
        values = np.take(values, multiindex)
        kwargs["array"] = values
        kwargs["cmap"] = cmap
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
    elif color is not None:
        if isinstance(color, (pd.Series, np.ndarray, list)):
            kwargs["facecolor"] = np.take(color, multiindex)
        else:
            kwargs["facecolor"] = color
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"

    # Create patches and collections
    patches = [_PolygonPatch(poly) for poly in geoms]
    collection = PatchCollection(patches, **kwargs)
    ax.add_collection(collection, autolim=autolim)

    return collection

def _plot_linestring_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, autolim=True, **kwargs):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` LineStrings and/or MultiLineStrings (can be
            mixed)
    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
    color : single color or sequence of `N` colors
        Cannot be used together with `values`.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    try:
        from matplotlib.collections import LineCollection
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    geoms, multiindex = _sanitize_geoms(geoms)
    if not geoms:
        return None

    # Extract line segments
    segments = []
    for line in geoms:
        if line.type == 'LineString':
            segments.append(np.array(line.coords))
        elif line.type == 'MultiLineString':
            segments.extend(np.array(ls.coords) for ls in line.geoms)

    # Process values and colors
    if values is not None:
        if color is not None:
            warnings.warn("'color' keyword is ignored when using 'values'")
        if cmap is None:
            cmap = "viridis"
        if isinstance(values, pd.Series):
            values = values.values
        values = np.take(values, multiindex)
        kwargs["array"] = values
        kwargs["cmap"] = cmap
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
    elif color is not None:
        if isinstance(color, (pd.Series, np.ndarray, list)):
            kwargs["color"] = np.take(color, multiindex)
        else:
            kwargs["color"] = color

    collection = LineCollection(segments, **kwargs)
    ax.add_collection(collection, autolim=autolim)

    return collection

def _plot_point_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, marker='o', markersize=None, **kwargs):
    """
    Plots a collection of Point and MultiPoint geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : sequence of `N` Points or MultiPoints

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    geoms, multiindex = _sanitize_geoms(geoms)
    if not geoms:
        return None

    # Convert points to coordinates array
    x = []
    y = []
    for point in geoms:
        if point.type == 'Point':
            x.append(point.x)
            y.append(point.y)
        elif point.type == 'MultiPoint':
            for p in point.geoms:
                x.append(p.x)
                y.append(p.y)

    # Process values and colors
    if values is not None:
        if color is not None:
            warnings.warn("'color' keyword is ignored when using 'values'")
        if cmap is None:
            cmap = "viridis"
        if isinstance(values, pd.Series):
            values = values.values
        values = np.take(values, multiindex)
        kwargs["c"] = values
        kwargs["cmap"] = cmap
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
    elif color is not None:
        if isinstance(color, (pd.Series, np.ndarray, list)):
            kwargs["c"] = np.take(color, multiindex)
        else:
            kwargs["c"] = color

    # Process marker size
    if markersize is not None:
        if isinstance(markersize, (pd.Series, np.ndarray, list)):
            markersize = np.take(markersize, multiindex)
        kwargs["s"] = markersize

    kwargs["marker"] = marker
    collection = ax.scatter(x, y, **kwargs)

    return collection

def plot_series(s, cmap=None, color=None, ax=None, figsize=None, aspect='auto', autolim=True, **style_kwds):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString, Point and MultiPoint
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str, np.array, pd.Series, List (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(s_y * pi/180) with s_y the y coordinate of the middle of
        the GeoSeries (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if aspect == 'auto':
        if s.crs is not None and s.crs.is_geographic:
            bounds = s.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(np.deg2rad(y_coord)))
        else:
            ax.set_aspect('equal')
    elif aspect is not None:
        ax.set_aspect(aspect)

    # Process style keywords
    if color is not None:
        style_kwds['color'] = color

    # Group geometries by type
    geoms = s.geometry.values
    polygons = []
    lines = []
    points = []

    for geom in geoms:
        if geom is None:
            continue
        if geom.type.startswith('Multi'):
            geom_type = geom.type[5:]  # Remove 'Multi' prefix
        else:
            geom_type = geom.type
        
        if geom_type == 'Polygon':
            polygons.append(geom)
        elif geom_type == 'LineString':
            lines.append(geom)
        elif geom_type == 'Point':
            points.append(geom)

    # Plot each geometry type
    if polygons:
        _plot_polygon_collection(ax, polygons, cmap=cmap, autolim=autolim, **style_kwds)
    if lines:
        _plot_linestring_collection(ax, lines, cmap=cmap, autolim=autolim, **style_kwds)
    if points:
        _plot_point_collection(ax, points, cmap=cmap, **style_kwds)

    return ax

def plot_dataframe(df, column=None, cmap=None, color=None, ax=None, cax=None, categorical=False, legend=False, scheme=None, k=5, vmin=None, vmax=None, markersize=None, figsize=None, legend_kwds=None, categories=None, classification_kwds=None, missing_kwds=None, aspect='auto', autolim=True, **style_kwds):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as
        dataframe. Values are used to color the plot. Ignored if `color` is
        also set.
    kind: str
        The kind of plots to produce. The default is to create a map ("geo").
        Other supported kinds of plots from pandas:

        - 'line' : line plot
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : BoxPlot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot
        - 'hexbin' : hexbin plot.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str, np.array, pd.Series (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    cax : matplotlib.pyplot Artist (default None)
        axes on which to draw the legend in case of color map.
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns, this
        will be set to True.
    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    k : int (default 5)
        Number of classes (ignored if scheme is None)
    vmin : None or float (default None)
        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.
    markersize : str or float or sequence (default None)
        Only applies to point geometries within a frame.
        If a str, will use the values in the column of the frame specified
        by markersize to set the size of markers. Otherwise can be a value
        to apply to all points, or a sequence of the same length as the
        number of points.
    figsize : tuple of integers (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        axes is given explicitly, figsize is ignored.
    legend_kwds : dict (default None)
        Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or
        :func:`matplotlib.pyplot.colorbar`.
        Additional accepted keywords when `scheme` is specified:

        fmt : string
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`).
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    missing_kwds : dict (default None)
        Keyword arguments specifying color options (as style_kwds)
        to be passed on to geometries with missing values in addition to
        or overwriting other style kwds. If None, geometries with missing
        values are not plotted.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(df_y * pi/180) with df_y the y coordinate of the middle of
        the GeoDataFrame (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(geodatasets.get_path("nybb"))
    >>> df.head()  # doctest: +SKIP
       BoroCode  ...                                           geometry
    0         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
    1         4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
    2         3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
    3         1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
    4         2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...

    >>> df.plot("BoroName", cmap="Set1")  # doctest: +SKIP

    See the User Guide page :doc:`../../user_guide/mapping` for details.

    """
    pass

@doc(plot_dataframe)
class GeoplotAccessor(PlotAccessor):
    _pandas_kinds = PlotAccessor._all_kinds

    def __call__(self, *args, **kwargs):
        data = self._parent.copy()
        kind = kwargs.pop('kind', 'geo')
        if kind == 'geo':
            return plot_dataframe(data, *args, **kwargs)
        if kind in self._pandas_kinds:
            return PlotAccessor(data)(kind=kind, **kwargs)
        else:
            raise ValueError(f'{kind} is not a valid plot kind')