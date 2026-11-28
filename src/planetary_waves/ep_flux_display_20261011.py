"""Shared EP display convention; never used to compute physical divergence."""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.text import Text
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

VERTICAL_FACTOR = 100.
ARROW_SCALE = 3500.  # reduced component units per inch; identical in both figures
READABLE_MIN_INCHES = .055
READABLE_MAX_INCHES = .130
READABLE_REFERENCE = 600.
READABLE_WIDTH_INCHES = .0072
UNIFORM_HEAD_POINTS = 6.
UNIFORM_SHAFT_POINTS = .9
UNIFORM_MIN_INCHES = .075
UNIFORM_MAX_INCHES = .160
ARROW_COLOURS = ['#2865bb','#008d9d','#2d9751','#b28708','#d96b16','#d24234','#b5233f']


def reference_arrow_cmap():
    """Moderately bright blue-to-red palette without near-white yellow/green."""
    return LinearSegmentedColormap.from_list('EP_high_contrast_white_background',ARROW_COLOURS,N=512)


def reduced_components(P, W, latitude):
    """F/(rho0*a), where the native diagnostic uses rho0=p/1000."""
    cosine = np.cos(np.deg2rad(latitude))[:, None]
    meridional, vertical = cosine * P, cosine * W
    return meridional, vertical, np.hypot(meridional, vertical)


def shown_levels(heights, minimum_spacing_km=4.):
    centre = np.nanmedian(heights, axis=tuple(range(heights.ndim - 1)))
    mask = np.zeros(len(centre), dtype=bool)
    last = -np.inf
    for j, altitude in enumerate(centre):
        if 20 <= altitude <= 90 and altitude - last >= minimum_spacing_km:
            mask[j] = True
            last = altitude
    return mask


def shown_levels_with_01hpa(heights, pressure):
    """Retain the native 0.1hPa node at the retrieval-grid spacing change."""
    mask = shown_levels(heights)
    pressure = np.asarray(pressure)
    j = int(np.argmin(abs(pressure - .1)))
    assert np.isclose(pressure[j], .1, rtol=1e-5)
    mask[j] = True
    return mask


def draw_arrows(ax, latitude, height, P, W, level_mask, cmap=None, norm=None,
                readable=False, uniform_heads=False, length_bounds_inches=None,
                head_points=None, shaft_points=None):
    """Common EP direction convention, vertical x100.

    Direction is displayed in screen coordinates: the physical components have
    the same units, then only the vertical component receives the stated stretch.
    The unstretched vector magnitude controls colour. With readable=True, use a
    visible minimum and square-root compression of display length; otherwise
    lengths are linear. Neither option smooths/interpolates or tests significance.
    """
    head_points = UNIFORM_HEAD_POINTS if head_points is None else float(head_points)
    shaft_points = UNIFORM_SHAFT_POINTS if shaft_points is None else float(shaft_points)
    mer, vert, magnitude = reduced_components(P, W, latitude)
    stretched=np.hypot(mer,VERTICAL_FACTOR*vert)
    if readable:
        # Preserve direction but compress length and give weak nonzero vectors a
        # visible minimum. Colour remains the true unstretched magnitude. This
        # mapping is not a linear flux-length scale and is recorded in captions.
        minimum=UNIFORM_MIN_INCHES if uniform_heads else READABLE_MIN_INCHES
        maximum=UNIFORM_MAX_INCHES if uniform_heads else READABLE_MAX_INCHES
        if length_bounds_inches is not None:
            minimum, maximum = map(float, length_bounds_inches)
            assert 0 < minimum <= maximum
        length=minimum+(maximum-minimum)*np.sqrt(
            np.clip(stretched/READABLE_REFERENCE,0,1))
        unit_mer=np.divide(mer,stretched,out=np.zeros_like(mer),where=stretched>0)
        unit_vert=np.divide(VERTICAL_FACTOR*vert,stretched,out=np.zeros_like(vert),where=stretched>0)
        vx,vy=length*unit_mer,length*unit_vert
        scale=1.
    else:
        vx,vy=mer,VERTICAL_FACTOR*vert
        scale=ARROW_SCALE
    xx = np.broadcast_to(latitude[:, None], height.shape)
    bbox = ax.get_position()
    width_inches, height_inches = bbox.width * ax.figure.get_figwidth(), bbox.height * ax.figure.get_figheight()
    # Exclude centres whose half-arrow would extend outside the plot frame.
    margin_x = np.abs(vx) / scale / 2 * 60 / width_inches + .1
    margin_y = np.abs(vy) / scale / 2 * 70 / height_inches + .1
    if uniform_heads:
        # Include the fixed head's transverse width and half the shaft stroke,
        # so a short, nearly horizontal arrow near the lower frame is not cut.
        head_half_width=.2*head_points/72
        stroke_padding=(shaft_points+.16)/144
        margin_x += (head_half_width*np.abs(unit_vert)+stroke_padding)*60/width_inches
        margin_y += (head_half_width*np.abs(unit_mer)+stroke_padding)*70/height_inches
    keep = ((xx >= 27.5) & (xx <= 77.5) & (xx >= 20 + margin_x)
            & (xx <= 80 - margin_x) & (height >= 20 + margin_y)
            & (height <= 90 - margin_y) & level_mask[None, :]
            & np.isfinite(magnitude) & (magnitude > 0))
    if uniform_heads:
        assert readable, 'Fixed head variant requires the documented readable length mapping.'
        # Convert the same screen-space directions to data endpoints. Only the
        # arrow head and shaft styling change; native vectors are not resampled.
        # Fixed mutation_scale gives every triangle the same physical head size,
        # irrespective of magnitude or shaft length, including weak vectors.
        dx = vx[keep] * np.diff(ax.get_xlim())[0] / width_inches
        dy = vy[keep] * np.diff(ax.get_ylim())[0] / height_inches
        patches=[]
        for x,y,sx,sy,value in zip(xx[keep],height[keep],dx,dy,magnitude[keep]):
            colour = cmap(norm(value)) if cmap is not None else '#222222'
            arrow=FancyArrowPatch((x-sx/2,y-sy/2),(x+sx/2,y+sy/2),
                arrowstyle='-|>,head_length=0.4,head_width=0.2',
                mutation_scale=head_points,
                linewidth=shaft_points,shrinkA=0,shrinkB=0,
                facecolor=colour,edgecolor=colour,transform=ax.transData,zorder=5,
                capstyle='round',joinstyle='miter')
            arrow.set_path_effects([pe.withStroke(linewidth=shaft_points+.16,
                                    foreground='#28323a'),pe.Normal()])
            ax.add_patch(arrow)
            patches.append(arrow)
        assert all(a.get_mutation_scale()==head_points for a in patches)
        assert all(a.get_linewidth()==shaft_points for a in patches)
        assert np.allclose(np.hypot(vx[keep],vy[keep]),length[keep])
        assert (length[keep]>=minimum).all() and (length[keep]<=maximum).all()
        assert np.allclose(vx[keep]*(VERTICAL_FACTOR*vert[keep])-vy[keep]*mer[keep],0,atol=1e-10)
        artist=ScalarMappable(norm=norm,cmap=cmap)
        artist.set_array(magnitude[keep])
        artist.fixed_head_patches=patches
        artist.native_keep=keep.copy()
        return artist,keep
    arguments = [xx[keep], height[keep], vx[keep], vy[keep]]
    options = dict(angles='uv', scale_units='inches', scale=scale,
        units='inches', width=READABLE_WIDTH_INCHES if readable else .0035, pivot='middle', headwidth=3.6,
        headlength=4.4, headaxislength=3.8, minshaft=.6, minlength=.25, zorder=5)
    if cmap is not None:
        arguments.append(magnitude[keep])
        options.update(cmap=cmap, norm=norm)
    else:
        options['color'] = '#222222'
    if readable:
        options.update(edgecolors='#28323a',linewidths=.10)
    artist = ax.quiver(*arguments, **options)
    if not readable:
        artist.set_path_effects([pe.withStroke(linewidth=.25,
            foreground='#777777' if cmap is not None else 'white')])
    if readable:
        assert np.allclose(np.hypot(vx[keep],vy[keep]),length[keep])
        assert (length[keep]>=READABLE_MIN_INCHES).all() and (length[keep]<=READABLE_MAX_INCHES).all()
        assert np.allclose(vx[keep]*(VERTICAL_FACTOR*vert[keep])-vy[keep]*mer[keep],0,atol=1e-10)
    return artist, keep


def add_key(ax, artist, reference=300., readable=False, show_vertical=True):
    if not readable:
        ax.quiverkey(artist, X=.72, Y=.95, U=reference, label=f'{reference:g}',
            labelpos='W', coordinates='axes', fontproperties={'size': 7},
            labelsep=.025, color='#222222')
    if show_vertical:
        ax.text(.98, .025, 'vertical ×100', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, color='#222222',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.72, pad=.3), zorder=6)


def validate_text_frame(fig):
    """Check visible text against the full canvas, not just the axes rectangle."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.bbox.width,fig.bbox.height
    clipped=[]
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box=artist.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > width+.5 or box.y1 > height+.5:
            clipped.append(artist.get_text())
    assert not clipped, f'Text outside canvas: {clipped}'
