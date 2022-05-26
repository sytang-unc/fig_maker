# used to render graphics
from typing import List, Tuple
import cairo
# unused curently
import csv
# used to import Rsvg
import gi

gi.require_version('Rsvg', '2.0')

# used to render svg's into cairo
from gi.repository import Rsvg

# used to call pdf2svg, pdflatex
import subprocess

# used to query pdf page size. Probably an easier way. I just took this from stackoverflow.
import pdfminer
import pdfminer.pdfparser
import pdfminer.pdfdocument
import pdfminer.pdfpage

# need value of pi
import math

# need to compute arctan2 to draw arrows
import numpy

"""
Drawable's maintain dimensions. Should output a cached surface for dynamic updates and a published surface for a finalized version
"""


class Drawable:
    def __init__(self, height_input: float, width_input: float, name_input: str =None):
        self.name = name_input

        self.height = height_input
        self.width = width_input

    def is_cached(self):
        print("Fell back to abstract method in Drawable!")
        assert (False)

    def update_cached_surface(self) -> None:
        print("Fell back to abstract method in Drawable!")
        assert (False)

    def draw_cached(self, ctx, x, y) -> None:
        print("Fell back to abstract method in Drawable!")
        assert (False)

    def is_published(self) -> None:
        print("Fell back to abstract method in Drawable!")
        assert (False)

    def update_published_surface(self) -> None:
        print("Fell back to abstract method in Drawable!")
        assert (False)

    def draw_published(self, ctx, x, y, alpha=1.0) -> None:
        print("Fell back to abstract method in Drawable!")
        assert (False)


"""
Sprites are immutable Drawables
"""


class Sprite(Drawable):
    def __init__(self, height_input: float, width_input: float, surface_input, name_input=None):
        Drawable.__init__(self, height_input, width_input, name_input)
        self.surface = surface_input

    def is_cached(self):
        return True

    def update_cached_surface(self):
        return

    def draw_cached(self, ctx, x, y):
        ctx.set_source_surface(self.surface, x, y)
        ctx.paint()

    def is_published(self):
        return True

    def update_published_surface(self):
        return

    def draw_published(self, ctx, x, y, alpha=1.0):
        ctx.set_source_surface(self.surface, x, y)
        if alpha == 1.0:
            ctx.paint()
        else:
            ctx.paint_with_alpha(alpha)


"""
For artificially extending bounding boxes
"""


class Blank(Drawable):
    def __init__(self, height_input, width_input):
        Drawable.__init__(self, height_input, width_input);

    def is_cached(self):
        return True

    def update_cached_surface(self):
        return

    def draw_cached(self, ctx, x, y):
        return

    def is_published(self):
        return True

    def update_published_surface(self):
        return

    def draw_published(self, ctx, x, y, alpha=1.0):
        return


preview_scale = 9.0

"""
Bounding box info is updated automatically. Drawables impose an acyclic tree structure
"""


class CompositeGraphic(Drawable):
    def __init__(self, name_input=None):
        Drawable.__init__(self, 0, 0, name_input)

        self.cached_surface = None
        self.pub_surface = None

        # may contain repeat instances
        self.elements = []
        # unique instances. This information is only kept so we know to redraw parents when we change
        self.parents = []

    # may lead to stackoverflow
    def _is_inbreeding(self, child_drawing):
        for parent in self.parents:
            if child_drawing is parent:
                return True
            if parent._is_inbreeding(child_drawing):
                return True
        return False

    def _add_element(self, draw_src, x, y):
        assert(x >= 0)
        assert(y >= 0)

        # confirm that this doesn't introduce cycles
        if isinstance(draw_src, CompositeGraphic) and self._is_inbreeding(draw_src):
            print("Child is also ancestor")
            return False

        if isinstance(draw_src, CompositeGraphic) and self not in draw_src.parents:
            draw_src.parents.append(self)

        self.elements.append((draw_src, x, y))

        self.width = max([x + draw_src.width, self.width])
        self.height = max([y + draw_src.height, self.height])
        self._alert_change()

        return True
    

    def _alert_change(self):
        self.cached_surface = None
        self.pub_surface = None
        for parent in self.parents:
            parent._alert_change()

    def remove_element(self, index):
        assert (0 <= index and index < len(self.elements))

        # get the drawable to be removed
        element = self.elements[index]
        el_drawable = element[0]
        assert (self in el_drawable.parents)

        # remove the drawable from elements
        last = len(self.elements)
        self.elements = self.elements[0:index] + self.elements[index + 1:last]

        # remove ourself from drawables parents if this was our only instance
        children = [el[0] for el in self.elements]
        if el_drawable not in children:
            el_drawable.parents.remove(self)

        # recompute bounding box
        self.width = 0
        self.height = 0
        for (draw, x, y) in self.elements:
            self.width = max([x + draw.width, self.width])
            self.height = max([y + draw.height, self.height])

        # Our cached surface is no longer accurate
        self._alert_change()

    def draw_center(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y - draw_src.height / 2)

    def draw_left_middle(self, draw_src, x, y):
        self._add_element(draw_src, x, y - draw_src.height / 2)

    def draw_right_middle(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width, y - draw_src.height / 2)

    def draw_left_top(self, draw_src, x=0, y=0):
        self._add_element(draw_src, x, y)

    def draw_right_bottom(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width, y - draw_src.height)

    def draw_left_bottom(self, draw_src, x, y):
        self._add_element(draw_src, x, y - draw_src.height)

    def draw_center_top(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y)

    def draw_center_bottom(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y - draw_src.height)

    def is_cached(self):
        return self.cached_surface is not None

    def update_cached_surface(self):
        surf = cairo.ImageSurface(cairo.Format.ARGB32, int(preview_scale * self.width),
                                  int(preview_scale * self.height))
        ctx = cairo.Context(surf)
        ctx.save()
        ctx.scale(preview_scale, preview_scale)
        for (drawable, x, y) in self.elements:
            if not drawable.is_cached():
                drawable.update_cached_surface()
            assert (drawable.is_cached())
            drawable.draw_cached(ctx, x, y)
        self.cached_surface = surf
        ctx.restore()

    def draw_cached(self, ctx, x, y):
        ctx.save()
        ctx.scale(1. / preview_scale, 1. / preview_scale)
        ctx.set_source_surface(self.cached_surface, x, y)
        ctx.paint()
        ctx.restore()

    def is_published(self):
        return self.pub_surface is not None

    def update_published_surface(self):
        surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        ctx = cairo.Context(surf)
        for (drawable, x, y) in self.elements:
            if not drawable.is_published():
                drawable.update_published_surface()
            assert (drawable.is_published())
            drawable.draw_published(ctx, x, y)
        self.pub_surface = surf

    def draw_published(self, ctx, x, y, alpha=1.0):
        ctx.set_source_surface(self.pub_surface, x, y)
        if alpha == 1.0:
            ctx.paint()
        else:
            ctx.paint_with_alpha(alpha)


"""
A PaperGraphic corresponds to a figure or subfigure
No one should be calling is_published or is_cached on this. This doesn't get drawn onto anything either
"""



class PaperGraphic(CompositeGraphic):
    def __init__(self, name_input):
        CompositeGraphic.__init__(self, name_input)

    def publish(self, format="pdf", buffer=10):
        CompositeGraphic.update_published_surface(self)
        if format == "pdf":
            surf = cairo.PDFSurface(self.name + '.pdf', self.width + buffer, self.height + buffer)
            #surf = cairo.SVGSurface(self.name + '.svg', self.width + 10, self.height + 10)
            ctx = cairo.Context(surf)
            ctx.set_source_surface(self.pub_surface, buffer/2, buffer/2)
            #ctx.save()
            #ctx.scale(preview_scale,preview_scale)
            #ctx.set_source_surface(self.pub_surface, 0.1*self.width, 0.1*self.height)
            ctx.paint()
            #ctx.restore()
            ctx.show_page()
            #surf.write_to_png(self.name + '.png')
            #self.preview()
        elif format == "png":
            surf = cairo.ImageSurface(cairo.Format.ARGB32, int(preview_scale*self.width*1.2), int(preview_scale*self.height*1.2))
            ctx = cairo.Context(surf)
            ctx.save()
            ctx.scale(preview_scale,preview_scale)
            ctx.set_source_surface(self.pub_surface, 0.1*self.width, 0.1*self.height)
            ctx.paint()
            ctx.restore()
            surf.write_to_png(self.name + '.png')
        else:
            print("Unknown format given to publish!")
    
    def draw_publish(self, src, format="pdf", buffer=10):
        self.draw_left_top(src)
        self.publish(format, buffer)

    def preview(self):
        CompositeGraphic.update_cached_surface(self)
        surf = cairo.ImageSurface(cairo.Format.ARGB32, int(preview_scale * self.width * 1.2),
                                  int(preview_scale * self.height * 1.2))
        ctx = cairo.Context(surf)
        #ctx.translate(0.1 * self.width, 0.1 * self.height)
        ctx.set_source_surface(self.cached_surface, preview_scale * 0.1 * self.width, preview_scale * 0.1 * self.height)
        ctx.paint()
        surf.write_to_png(self.name + '.png')


black_pattern = cairo.SolidPattern(0, 0, 0)
white_pattern = cairo.SolidPattern(1, 1, 1)
gray_pattern = cairo.SolidPattern(0.66, 0.66, 0.66)
dark_gray_pattern = cairo.SolidPattern(0.33, 0.33, 0.33)

sol_blue_pattern = cairo.SolidPattern(0,0,1)
red_pattern = cairo.SolidPattern(211/256,94/256,96/256)
transp_red_pattern = cairo.SolidPattern(211/256,94/256,96/256,0.5)
blue_pattern = cairo.SolidPattern(114/256,147/256,203/256)
standard_blue_pattern = blue_pattern
blue_vert_surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, cairo.Rectangle(0,0,3,3))
blue_vert_surf_ctx = cairo.Context(blue_vert_surface)
blue_vert_surf_ctx.set_source(blue_pattern)
blue_vert_surf_ctx.move_to(0, 0)
blue_vert_surf_ctx.line_to(0, 3)
blue_vert_surf_ctx.line_to(3, 3)
blue_vert_surf_ctx.line_to(3, 0)
blue_vert_surf_ctx.line_to(0, 0)
blue_vert_surf_ctx.close_path()
blue_vert_surf_ctx.fill()
blue_vert_surf_ctx.set_source(black_pattern)
blue_vert_surf_ctx.move_to(3, 0)
blue_vert_surf_ctx.line_to(3, 3)
blue_vert_surf_ctx.stroke()
blue_pattern = cairo.SurfacePattern(blue_vert_surface)
blue_pattern.set_extend(cairo.Extend.REPEAT)

orange_pattern = cairo.SolidPattern(225/256,151/256,76/256)
standard_orange_pattern = orange_pattern
orange_horiz_surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, cairo.Rectangle(0, 0, 3, 3))
orange_horiz_surface_ctx = cairo.Context(orange_horiz_surface)
orange_horiz_surface_ctx.set_source(orange_pattern)
orange_horiz_surface_ctx.move_to(0, 0)
orange_horiz_surface_ctx.line_to(0, 3)
orange_horiz_surface_ctx.line_to(3, 3)
orange_horiz_surface_ctx.line_to(3, 0)
orange_horiz_surface_ctx.line_to(0, 0)
orange_horiz_surface_ctx.close_path()
orange_horiz_surface_ctx.fill()
orange_horiz_surface_ctx.set_source(black_pattern)
orange_horiz_surface_ctx.move_to(0, 3)
orange_horiz_surface_ctx.line_to(3, 3)
orange_horiz_surface_ctx.stroke()
orange_pattern = cairo.SurfacePattern(orange_horiz_surface)
orange_pattern.set_extend(cairo.Extend.REPEAT)

purple_pattern = cairo.SolidPattern(144/256,103/256,167/256)
purple_diag_surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, cairo.Rectangle(0, 0, 3, 3))
purple_diag_surface_ctx = cairo.Context(purple_diag_surface)
purple_diag_surface_ctx.set_source(purple_pattern)
purple_diag_surface_ctx.move_to(0, 0)
purple_diag_surface_ctx.line_to(0, 3)
purple_diag_surface_ctx.line_to(3, 3)
purple_diag_surface_ctx.line_to(3, 0)
purple_diag_surface_ctx.line_to(0, 0)
purple_diag_surface_ctx.close_path()
purple_diag_surface_ctx.fill()
purple_diag_surface_ctx.set_source(black_pattern)
purple_diag_surface_ctx.move_to(3, 0)
purple_diag_surface_ctx.line_to(3, 3)
purple_diag_surface_ctx.line_to(0, 3)
purple_diag_surface_ctx.stroke()
purple_pattern = cairo.SurfacePattern(purple_diag_surface)
purple_pattern.set_extend(cairo.Extend.REPEAT)

transp_pattern = cairo.SolidPattern(0.5, 0.5, 0.5, 0.5)
transp_white_pattern = cairo.SolidPattern(1, 1, 1, 0.25)
no_pattern = cairo.SolidPattern(0,0,0,0)
default_line_width = 0.5

green_diag_surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, cairo.Rectangle(0,0,30,30))
green_diag_surface_ctx = cairo.Context(green_diag_surface)
green_diag_surface_ctx.set_source(cairo.SolidPattern(65/256, 140/256, 34/256))
green_diag_surface_ctx.move_to(0,0)
green_diag_surface_ctx.line_to(0,30)
green_diag_surface_ctx.line_to(30,30)
green_diag_surface_ctx.line_to(30,0)
green_diag_surface_ctx.line_to(0,0)
green_diag_surface_ctx.close_path()
green_diag_surface_ctx.fill()
green_diag_surface_ctx.set_source(black_pattern)
green_diag_surface_ctx.set_line_width(2.0/math.sqrt(2))
for i in range(1,10):
    green_diag_surface_ctx.move_to(3*i,30)
    green_diag_surface_ctx.line_to(0, 30-3*i)
    green_diag_surface_ctx.stroke()
    green_diag_surface_ctx.move_to(30,3*i)
    green_diag_surface_ctx.line_to(30-3*i,0)
    green_diag_surface_ctx.stroke()
green_diag_surface_ctx.move_to(30,30)
green_diag_surface_ctx.line_to(0,0)
green_diag_surface_ctx.stroke()
green_diag_surface_ctx.move_to(3,3)
green_diag_surface_ctx.line_to(0,0)
green_diag_surface_ctx.stroke()
green_diag_pattern = cairo.SurfacePattern(green_diag_surface)
green_diag_pattern.set_extend(cairo.Extend.REPEAT)

green_pattern = cairo.SolidPattern(0,1,0)

"""
The information needed to produce a schedule figure. Assumes indexing of tasks/procs begins at 0
"""


class SchedData:
    def __init__(self, n_input, m_input, sched_end_min=0, budg_max=0, s_max = 1.0):
        self.n = n_input
        self.m = m_input
        self.executions = {}
        self.np_executions = {}
        self.releases = {}
        self.completions = {}
        self.deadlines = {}
        self.pseudo_deadlines = {}
        self.pseudo_releases = {}
        self.annotations = {}
        self.budgets = []
        self.budg_max = budg_max
        self.sched_end = sched_end_min
        self.s_max = s_max
        for i in range(self.n):
            self.executions[i] = []
            self.np_executions[i] = []
            self.releases[i] = []
            self.pseudo_releases[i] = []
            self.completions[i] = []
            self.deadlines[i] = []
            self.pseudo_deadlines[i] = []
            self.annotations[i] = []

    def add_execution(self, task_id, proc, start_time, end_time, speed=1.0):
        assert(speed <= self.s_max)
        assert (proc < self.m and task_id < self.n)
        assert (start_time < end_time and start_time >= 0)
        self.executions[task_id].append((proc, start_time, end_time, speed))

        self.sched_end = max([self.sched_end, end_time])

    def add_np_execution(self, task_id, proc, start_time, end_time, speed=1.0):
        assert (proc < self.m and task_id < self.n)
        assert (start_time < end_time and start_time >= 0)
        self.np_executions[task_id].append((proc, start_time, end_time, speed))

        self.sched_end = max([self.sched_end, end_time])

    def add_release(self, task_id, t):
        assert (task_id < self.n)
        self.releases[task_id].append(t)

        self.sched_end = max([self.sched_end, t])

    def add_pseudo_release(self, task_id, t):
        assert (task_id < self.n)
        self.pseudo_releases[task_id].append(t)

        self.sched_end = max([self.sched_end, t])

    def add_completion(self, task_id, t):
        assert (task_id < self.n)
        self.completions[task_id].append(t)

        self.sched_end = max([self.sched_end, t])

    def add_deadline(self, task_id, t):
        assert (task_id < self.n)
        self.deadlines[task_id].append(t)

        self.sched_end = max([self.sched_end, t])

    def add_pseudo_deadline(self, task_id, t):
        assert (task_id < self.n)
        self.pseudo_deadlines[task_id].append(t)

        self.sched_end = max([self.sched_end, t])
    
    def add_rd(self, task_id, t):
        self.add_release(task_id, t)
        self.add_deadline(task_id, t)

    def add_annotation(self, task_id, text, start, end=-1, justify="middle"):
        assert (task_id < self.n)
        assert (justify == "left" or justify == "middle" or justify == "right")
        if end != -1:
            assert (start < end)
        else:
            end = start
        
        self.sched_end = max([self.sched_end, end])

        self.annotations[task_id].append((text, start, end, justify))
    
    def add_budget(self, start, end, init, final):
        assert(start < end)

        self.sched_end = max([self.sched_end, end])

        self.budgets.append((start, end, init, final))
        self.budg_max = max([self.budg_max, init, final])


def draw_sched(schedule, start_from_zero=False, task_identifiers=None, scale_time_to_dots=10, time_axis_inc=5, sched_start_time=0, draw_time=True, draw_time_txt=True, transp_bg=True, title=None, draw_speed_line=False):
    sched_graphic = draw_sched_help(schedule, start_from_zero, task_identifiers, scale_time_to_dots, time_axis_inc, sched_start_time, draw_time, draw_time_txt, draw_speed_line)
    if title is not None:
        title_txt = latex_to_sprite(title, is_math=False)
        cg = CompositeGraphic()
        cg.draw_center_top(title_txt, max([title_txt.width, sched_graphic.width])/2, 0)
        cg.draw_center_top(sched_graphic, max([title_txt.width, sched_graphic.width])/2, title_txt.height)
        sched_graphic = cg
    if transp_bg:
        return sched_graphic
    tot_graphic = CompositeGraphic()
    tot_graphic.draw_left_top(rect_to_sprite(sched_graphic.height, sched_graphic.width))
    tot_graphic.draw_left_top(sched_graphic)
    return tot_graphic

"""
proc, start, end, speed
"""
def join_intervals(intervals: List[Tuple[int, float, float, float]]) -> List[Tuple[int, float,float,float]]:
    if len(intervals) <= 1:
        return intervals
    new_intervals: List[Tuple[int, float,float,float]] = []
    rm_first = intervals[1:]
    proc, start, end, speed = intervals[0]
    for proc2, start2, end2, speed2 in rm_first:
        if proc != proc2 or speed != speed2 or start2 > end:
            new_intervals.append((proc,start,end,speed))
            proc, start, end, speed = (proc2, start2, end2, speed2)
        else:
            end = end2
    new_intervals.append((proc, start, end, speed))
    return new_intervals


"""
Render a given SchedData
INPUT
    schedule - SchedData object
    start_from_zero - do task/proc subscripts start from 0 or 1
"""


def draw_sched_help(schedule: SchedData, start_from_zero=False, task_identifiers=None, scale_time_to_dots=10, time_axis_inc=5, sched_start_time=0, draw_time=True, draw_time_txt=True, draw_speed_line=False):
    sched_graphic = CompositeGraphic()

    # spaceing constants. Everything that is scaled is relative to label dimensions
    offset_0 = 5
    extra_line = 20
    exec_vertical_scale = 1.5
    vertical_spacing_scale = 1.0
    vertical_space_below_last_task = 5
    horiz_offset_scale = 1.2
    comp_T_width = 8

    # create sprites for task labels
    task_labels = []
    if task_identifiers is None:
        task_labels = [latex_to_sprite('\\tau_' + str(i if start_from_zero else i + 1)) for i in range(schedule.n)]
    else:
        task_labels = [latex_to_sprite(identifier) for identifier in task_identifiers]
    task_label_width_max = max([task_label.width for task_label in task_labels])
    task_label_height_max = max([task_label.height for task_label in task_labels])

    annot_sprites = {}
    annot_heights = {}
    for i in range(schedule.n):
        annot_sprites[i] = []
        annot_heights[i] = 0
        for (annot, start, end, just) in schedule.annotations[i]:
            latex_annot = latex_to_sprite(annot, is_math=False)
            annot_sprites[i].append((latex_annot, start, end, just))
            annot_heights[i] = max([annot_heights[i], latex_annot.height])
       

    # compute how high each horizontal axis will be
    y_botts = [0 for i in range(schedule.n)]
    y_botts[0] = (exec_vertical_scale + vertical_spacing_scale) * task_label_height_max + annot_heights[0]
    if schedule.budg_max > 0:
        #budg_label = latex_to_sprite(r"C_s")
        #task_label_height_max = max([task_label_height_max, budg_label.height])
        #task_label_width_max = max([task_label_width_max, budg_label.width])
        budg_bott = y_botts[0]
        y_botts[0] += (exec_vertical_scale + vertical_spacing_scale) * task_label_height_max

    if schedule.n > 1:
        for i in range(1, schedule.n):
            y_botts[i] = y_botts[i - 1] + (exec_vertical_scale + vertical_spacing_scale) * task_label_height_max + \
                         annot_heights[i]
    # y_botts = [(exec_vertical_scale + vertical_spacing_scale)*task_label_height_max*(i+1) for i in range(schedule.n)]
    last_axis_y = y_botts[schedule.n - 1] + vertical_space_below_last_task

    # create sprites for job releases and completions
    release_arrow = path_to_sprite([(0, task_label_height_max * (exec_vertical_scale + 0.5 * vertical_spacing_scale))], \
                                   ending_style=PathEndingStyle.BEGIN_ARROW, line_width=default_line_width * 2)
    pseudo_release_arrow = path_to_sprite([(0, task_label_height_max * (exec_vertical_scale + 0.5 * vertical_spacing_scale))], \
                                   ending_style=PathEndingStyle.BEGIN_ARROW, line_width=default_line_width * 2, line_pattern=gray_pattern)

    completion_T = path_to_sprite([(comp_T_width / 2, 0), (
    comp_T_width / 2, task_label_height_max * (exec_vertical_scale + 0.5 * vertical_spacing_scale)), \
                                   (comp_T_width / 2, 0), (comp_T_width, 0)], line_width=default_line_width * 2)
    deadline_arrow = path_to_sprite([(0, task_label_height_max * (exec_vertical_scale + 0.5 * vertical_spacing_scale))], \
                                    ending_style=PathEndingStyle.ENDING_ARROW, line_width=default_line_width * 2)

    pseudo_deadline_arrow = path_to_sprite([(0, task_label_height_max * (exec_vertical_scale + 0.5 * vertical_spacing_scale))], \
                                    ending_style=PathEndingStyle.ENDING_ARROW, line_width=default_line_width * 2, line_pattern=gray_pattern)

    # draw the main axes
    axis = path_to_sprite(
        [(0, last_axis_y - annot_heights[0]), (scale_time_to_dots * (schedule.sched_end - sched_start_time) + offset_0 + extra_line, last_axis_y - annot_heights[0])],
        ending_style=PathEndingStyle.BOTH_ARROW)
    sched_graphic.draw_left_bottom(axis, task_label_width_max * horiz_offset_scale, last_axis_y)

    # draw per task items
    for i in range(schedule.n):
        # draw the task label
        sched_graphic.draw_right_middle(task_labels[i], task_label_width_max,
                                        y_botts[i] - exec_vertical_scale / 2 * task_label_height_max)
        # draw axis for task
        task_axis = path_to_sprite([(scale_time_to_dots * (schedule.sched_end - sched_start_time) + offset_0 + extra_line, 0)],
                                   ending_style=PathEndingStyle.ENDING_ARROW, line_pattern=transp_pattern)
        
        
        # draw each block of execution for this task
        for (proc, start_time, end_time, speed) in join_intervals(schedule.executions[i]):
            sched_block = rect_to_sprite(task_label_height_max * exec_vertical_scale * speed/schedule.s_max,
                                         scale_time_to_dots * (end_time - start_time), fill_pattern=sources_dict[proc], line_pattern=None)
            sched_graphic.draw_left_bottom(sched_block,
                                           offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (start_time - sched_start_time),
                                           y_botts[i])
        for (proc, start_time, end_time, speed) in join_intervals(schedule.np_executions[i]):
            sched_block = rect_to_sprite(task_label_height_max * exec_vertical_scale * speed/schedule.s_max,
                                         scale_time_to_dots * (end_time - start_time), fill_pattern=sources_dict[proc],
                                         dash=True)
            sched_graphic.draw_left_bottom(sched_block,
                                           offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (start_time - sched_start_time),
                                           y_botts[i])
        sched_graphic.draw_left_bottom(task_axis, task_label_width_max * horiz_offset_scale, y_botts[i])
        if draw_speed_line:
            speed_axis = path_to_sprite([(scale_time_to_dots * (schedule.sched_end - sched_start_time) + offset_0 + extra_line, 0)],
                                   line_pattern=transp_pattern, dash=True)
            sched_graphic.draw_left_bottom(speed_axis, task_label_width_max*horiz_offset_scale, y_botts[i] + task_label_height_max * exec_vertical_scale * 1.0/schedule.s_max)

        # draw releases/completions
        for t in schedule.releases[i]:
            sched_graphic.draw_center_bottom(release_arrow,
                                             offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (t-sched_start_time),
                                             y_botts[i])
        for t in schedule.pseudo_releases[i]:
            sched_graphic.draw_center_bottom(pseudo_release_arrow,
                                             offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (t-sched_start_time),
                                             y_botts[i])

        for t in schedule.completions[i]:
            sched_graphic.draw_center_bottom(completion_T,
                                             offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (t-sched_start_time),
                                             y_botts[i])
        for t in schedule.deadlines[i]:
            sched_graphic.draw_center_bottom(deadline_arrow,
                                             offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (t-sched_start_time),
                                             y_botts[i])

        for t in schedule.pseudo_deadlines[i]:
            sched_graphic.draw_center_bottom(pseudo_deadline_arrow,
                                             offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (t-sched_start_time),
                                             y_botts[i])

        # draw annotations
        annot_i_y = y_botts[i] - (exec_vertical_scale + vertical_spacing_scale) * task_label_height_max
        for (annot, start, end, just) in annot_sprites[i]:
            start_x = offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (start - sched_start_time)
            end_x = offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (end - sched_start_time)
            if just == "left":
                sched_graphic.draw_left_bottom(annot, start_x, annot_i_y)
            if just == "middle":
                sched_graphic.draw_center_bottom(annot, 0.5*(start_x + end_x), annot_i_y)
            if just == "right":
                sched_graphic.draw_right_bottom(annot, end_x, annot_i_y)
            if (start != end):
                range_arrows = path_to_sprite([(scale_time_to_dots * (end - start), 0)],
                                              ending_style=PathEndingStyle.BOTH_ARROW)
                sched_graphic.draw_left_middle(range_arrows, start_x, annot_i_y)
    
    if schedule.budg_max > 0:
        # draw the budget label
        #sched_graphic.draw_right_middle(budg_label, task_label_width_max,
        #                                budg_bott - exec_vertical_scale / 2 * task_label_height_max)
        # draw axis for budget
        budg_axis = path_to_sprite([(scale_time_to_dots * (schedule.sched_end - sched_start_time) + offset_0 + extra_line, 0)],
                                   ending_style=PathEndingStyle.ENDING_ARROW, line_pattern=transp_pattern)
        sched_graphic.draw_left_bottom(budg_axis, task_label_width_max * horiz_offset_scale, budg_bott)
        for (start, end, init, final) in schedule.budgets:
            start_x = offset_0 + task_label_width_max*horiz_offset_scale + scale_time_to_dots*(start - sched_start_time)
            end_x = offset_0 + task_label_width_max*horiz_offset_scale + scale_time_to_dots*(end - sched_start_time)
            init_dy = task_label_height_max*exec_vertical_scale*init/schedule.budg_max
            final_dy = task_label_height_max*final*exec_vertical_scale/schedule.budg_max
            sched_budg = poly_to_sprite([(0,task_label_height_max*exec_vertical_scale - init_dy), 
                (0,task_label_height_max*exec_vertical_scale), 
                (end_x - start_x, task_label_height_max*exec_vertical_scale),
                (end_x - start_x, task_label_height_max*exec_vertical_scale - final_dy)], fill_pattern=sol_blue_pattern, line_pattern=None)
            sched_graphic.draw_left_bottom(sched_budg, start_x, budg_bott)

    # label the time axis
    time_height = 0
    for t in range(sched_start_time, int(schedule.sched_end + 1), time_axis_inc):
        if draw_time:
            axis_label = latex_to_sprite(str(t))
            sched_graphic.draw_center_top(axis_label,
                                          task_label_width_max * horiz_offset_scale + offset_0 + scale_time_to_dots * (t-sched_start_time),
                                       last_axis_y)
            time_height = max([time_height, axis_label.height])

    # vertical lines for each time instant
    for t in range(sched_start_time, int(schedule.sched_end + 1), time_axis_inc):
        time_vert_line = path_to_sprite([(0, last_axis_y - annot_heights[0])], line_pattern=transp_pattern, line_width=0.25)
        sched_graphic.draw_left_bottom(time_vert_line,
                                       task_label_width_max * horiz_offset_scale + offset_0 + scale_time_to_dots * (t-sched_start_time),
                                       last_axis_y)

    if draw_time_txt:
        time_text = latex_to_sprite("Time", is_math=False)
        sched_width = sched_graphic.width - task_label_width_max * horiz_offset_scale
        sched_graphic.draw_center_top(time_text, task_label_width_max * horiz_offset_scale + sched_width / 2,
                                  last_axis_y + time_height)

    return sched_graphic


"""
Folder for frequently used svgs
"""
freq_dir = '/home/sytang/vscode/fig_maker/latex_frequent/'

"""
Frequently used latex expressions mapping to svgs
There's probably a way to query svg size information, but I don't know it and I don't feel like figuring it out, so it's hardcoded whoopdeedoo
"""
latex_frequent_dict = dict([('\\tau_0', ('tau_0.svg', 9.784, 12.825)), \
                            ('\\tau_1', ('tau_1.svg', 9.784, 12.825)), \
                            ('\\tau_2', ('tau_2.svg', 9.784, 12.825)), \
                            ('\\tau_3', ('tau_3.svg', 9.784, 12.825)), \
                            ('\\tau_4', ('tau_4.svg', 9.784, 12.825)), \
                            ('\\tau_5', ('tau_5.svg', 9.784, 12.825)), \
                            ('\\tau_6', ('tau_6.svg', 9.784, 12.825)), \
                            ('\\tau_s', ('tau_s.svg', 9.784, 12.825)), \
                            ('\\tau_i', ('tau_i.svg', 9.784, 12.825)), \
                            ('\\pi_0', ('pi_0.svg', 9.784, 14.148)), \
                            ('\\pi_1', ('pi_1.svg', 9.784, 14.148)), \
                            ('\\pi_2', ('pi_2.svg', 9.784, 14.148)), \
                            ('\\pi_3', ('pi_3.svg', 9.784, 14.148)), \
                            ('\\pi_4', ('pi_4.svg', 9.784, 14.148)), \
                            ('\\pi_5', ('pi_5.svg', 9.784, 14.148)), \
                            ('\\theta_1', ('theta_1.svg', 12.413, 13.146)), \
                            ('\\theta_2', ('theta_2.svg', 12.413, 13.146)), \
                            ('\\phi_1', ('phi_1.svg', 12.856, 14.405)), \
                            ('\\phi_2', ('phi_2.svg', 12.856, 14.405)), \
                            ('\\Phi_1', ('Phi_1.svg', 12.302, 15.665)), \
                            ('\\Phi_2', ('Phi_2.svg', 12.302, 15.665)), \
                            ('0', ('0.svg', 10.42, 8.981)), \
                            ('1', ('1.svg', 10.42, 8.981)), \
                            ('2', ('2.svg', 10.42, 8.981)), \
                            ('3', ('3.svg', 10.42, 8.981)), \
                            ('4', ('4.svg', 10.42, 8.981)), \
                            ('5', ('5.svg', 10.42, 8.981)), \
                            ('6', ('6.svg', 10.42, 8.981)), \
                            ('7', ('7.svg', 10.42, 8.981)), \
                            ('8', ('8.svg', 10.42, 8.981)), \
                            ('9', ('9.svg', 10.42, 8.981)), \
                            ('10', ('10.svg', 10.42, 13.963)), \
                            ('12', ('12.svg', 10.42, 13.963)), \
                            ('14', ('14.svg', 10.42, 13.963)), \
                            ('15', ('15.svg', 10.42, 13.963)), \
                            ('16', ('16.svg', 10.42, 13.963)), \
                            ('18', ('18.svg', 10.42, 13.963)), \
                            ('20', ('20.svg', 10.42, 13.963)), \
                            ('22', ('22.svg', 10.42, 13.963)), \
                            ('24', ('24.svg', 10.42, 13.963)), \
                            ('25', ('25.svg', 10.42, 13.963)), \
                            ('30', ('30.svg', 10.42, 13.963)), \
                            ('Time', ('Time.svg', 10.808, 26.693)), \
                            ('Release', ('Release.svg', 10.918, 36.296)), \
                            ('Deadline', ('Deadline.svg', 10.918, 42.052)), \
                            ('Completion', ('Completion.svg', 12.856, 54.367)), \
                            ('Original', ('Original.svg', 12.856, 39.45)),
                            ('Ideal', ('Ideal.svg', 10.918, 25.309))])

"""
Directory for transient files
"""
tmp_dir = '/home/sytang/vscode/fig_maker/tmp_data/'

"""
This is bad and I should feel bad, but it works well enough I guess. Uses latex to create a standalone pdf of text, converts the pdf to svg with pdf2svg, reads the svg back into cairo with Rsvg. This is especially stupid because pdf2svg already reads the pdf into cairo, but that's dependent on cairo & poppler's C APIs and I'm too lazy to leave python
INPUT
    text - string of some basic latex equation
OUTPUT
    A Drawable of text or None on error
"""


def latex_to_sprite(text, is_math=True, col=black_pattern) -> Sprite:
    svgfile = ''
    height = 0
    width = 0
    if text in latex_frequent_dict.keys():
        (svgname, height, width) = latex_frequent_dict[text]
        svgfile = freq_dir + svgname
    else:
        print(text)
        with open(tmp_dir + 'tmp.tex', 'w') as tmp_tex:
            tmp_tex.write('\\documentclass[varwidth, border=2]{standalone}\n')
            tmp_tex.write('\\usepackage{amsmath}\n')
            tmp_tex.write('\\usepackage{xcolor}\n')
            tmp_tex.write('\\begin{document}\n')
            if is_math:
                tmp_tex.write('$')
            tmp_tex.write(text)
            if is_math:
                tmp_tex.write('$')
            tmp_tex.write('\n')
            tmp_tex.write('\\end{document}')

        subprocess.run(['pdflatex', '-interaction', 'batchmode', '-output-directory', tmp_dir, tmp_dir + 'tmp.tex'])

        (height, width) = get_pdf_page_size(tmp_dir + 'tmp.pdf')
        if height == -1:
            return None

        subprocess.run(['pdf2svg', tmp_dir + 'tmp.pdf', tmp_dir + 'tmp.svg'])
        svgfile = tmp_dir + 'tmp.svg'

    svg = Rsvg.Handle.new_from_file(svgfile)
    # set the dpi here to prevent scaling when the svg is rendered into cairo
    svg.set_dpi(72)

    # surface = cairo.SVGSurface(None, width, height)
    surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surface)
    svg.render_cairo(ctx)
    # close is apparently depreciated? Find out how to properly close the svg later
    svg.close()


    if col is not black_pattern:
        surf2 = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        ctx2 = cairo.Context(surf2)
        ctx2.set_source(col)
        ctx2.mask_surface(surface, 0, 0)
        ctx2.fill()
        out = Sprite(height, width, surf2, text)
    else:
        out = Sprite(height, width, surface, text)

    return out


def box(width: int, height: int, image: Drawable):
    assert(width >= image.width)
    assert(height >= image.height)
    out = CompositeGraphic()
    out.draw_left_top(rect_to_sprite(height, width))
    out.draw_center(image, width/2, height/2)
    return out

def get_pdf_page_size(pdf_name):
    height = 0
    width = 0
    with open(pdf_name, 'rb') as tmp_pdf:
        parser = pdfminer.pdfparser.PDFParser(tmp_pdf)
        doc = pdfminer.pdfdocument.PDFDocument(parser)
        for page in pdfminer.pdfpage.PDFPage.create_pages(doc):
            height = page.mediabox[3]
            width = page.mediabox[2]
            break
    if height == 0 or width == 0:
        print("0 height or width")
        return (-1, -1)

    return (height, width)


"""
legend_dict should be a dictionary of names and drawables
rownum is how many entries per row in the legend
"""


def draw_legend(legend_dict, per_row=0, horiz_padding = 0):
    keys = list(legend_dict.keys())
    assert (per_row <= len(keys))
    if per_row <= 0:
        per_row = len(keys)
    rows = int(len(keys) / per_row)
    if (len(keys) % per_row) != 0:
        rows += 1
    # 2*per_row because we need a col for the key and the symbol
    col_widths = [0 for i in range(2 * per_row)]
    row_heights = [0 for i in range(rows)]
    sprite_array_keys = [[0 for i in range(per_row)] for j in range(rows)]
    sprite_array_symbols = [[0 for i in range(per_row)] for j in range(rows)]
    # first pass computes where positions are gonna be
    for r in range(rows):
        # note, python is cool with the end index being greater than len(keys)
        row_keys = keys[per_row * r:per_row * (r + 1)]
        for c in range(len(row_keys)):
            key_sprite = latex_to_sprite(row_keys[c], is_math=False)
            sprite_array_keys[r][c] = key_sprite
            symbol_sprite = legend_dict[row_keys[c]]
            sprite_array_symbols[r][c] = symbol_sprite
            col_widths[2 * c] = max([col_widths[2 * c], key_sprite.width + horiz_padding])
            col_widths[2 * c + 1] = max([col_widths[2 * c + 1], symbol_sprite.width + horiz_padding])
            row_heights[r] = max([row_heights[r], key_sprite.height, symbol_sprite.height])
    # col_widths = [col_widths[i]*1.5 if (i is not 0 and i is not len(col_widths)-1) else col_widths[i] for i in range(len(col_widths))]
    row_heights = [row_heights[i] * 1.5 if (i != 0 and i != len(row_heights) - 1) else row_heights[i] * 1.2 for
                   i in range(len(row_heights))]
    legend_graphic = CompositeGraphic()
    total_width = sum(col_widths) + (per_row - 1) * 5 + 1
    total_height = sum(row_heights)
    bounding_box = rect_to_sprite(total_height, total_width)
    legend_graphic.draw_left_top(bounding_box, 0, 0)
    # second pass draws everything
    for r in range(rows):
        row_keys = keys[per_row * r:per_row * (r + 1)]
        for c in range(len(row_keys)):
            key_x = sum(col_widths[0:2 * c]) + col_widths[2 * c] / 2 + c * 5
            key_y = sum(row_heights[0:r]) + row_heights[r] / 2
            legend_graphic.draw_center(sprite_array_keys[r][c], key_x, key_y)
            sprite_x = sum(col_widths[0:2 * c + 1]) + col_widths[2 * c + 1] / 2 + c * 5
            legend_graphic.draw_center(sprite_array_symbols[r][c], sprite_x, key_y)

    return legend_graphic


def draw_default_legend(m, per_row=4):
    rl_sprite = path_to_sprite([(0, 15)], ending_style=PathEndingStyle.BEGIN_ARROW)
    dl_sprite = path_to_sprite([(0, 15)], ending_style=PathEndingStyle.ENDING_ARROW)
    cp_sprite = path_to_sprite([(3, 0), (3, 15), (3, 0), (6, 0)])
    #np_sprite = rect_to_sprite(15, 15, dash=True)

    #legend_dict = dict([('Release', rl_sprite), ('Deadline', dl_sprite), ('Completion', cp_sprite), ('NP', np_sprite)])
    #legend_dict = dict([('\\textbf{Legend}', rect_to_sprite(3,3,fill_pattern=white_pattern,line_pattern=white_pattern)), ('Release', rl_sprite), ('Deadline', dl_sprite), ('Completion', cp_sprite)])#, ('NP', np_sprite)])
    legend_dict = dict([('Release', rl_sprite), ('Deadline', dl_sprite), ('Completion', cp_sprite)])#, ('NP', np_sprite)])

    for j in range(m):
        proc_sprite = rect_to_sprite(15, 15, fill_pattern=sources_dict[j])
        proc_text = '$\\pi_' + str(j + 1) + '$'
        legend_dict[proc_text] = proc_sprite

    return draw_legend(legend_dict, per_row=per_row)


def draw_default_ag_legend(per_row=3):
    link_sprite = path_to_sprite([(0, 15)], ending_style=PathEndingStyle.BEGIN_ARROW)
    assign_sprite = path_to_sprite([(0, 15)], line_width=default_line_width * 10, line_pattern=transp_pattern)
    np_sprite = path_to_sprite([(0, 16)], line_width=default_line_width * 10, dash=True)
    bounding_box = Blank(15, default_line_width * 15)
    np_comp = CompositeGraphic()
    np_comp.draw_center_top(np_sprite, default_line_width * 10 / 2, 0)
    np_comp.draw_center_top(bounding_box, default_line_width * 10 / 2, 0)
    link_comp = CompositeGraphic()
    link_comp.draw_center_top(link_sprite, default_line_width * 10 / 2, 0)
    link_comp.draw_center_top(bounding_box, default_line_width * 10 / 2, 0)
    assign_comp = CompositeGraphic()
    assign_comp.draw_center_top(assign_sprite, default_line_width * 10 / 2, 0)
    assign_comp.draw_center_top(bounding_box, default_line_width * 10 / 2, 0)

    legend_dict = dict([('Linked', link_comp), ('Scheduled', assign_comp), ('NP', np_comp)])

    return draw_legend(legend_dict, per_row=per_row)





class PathEndingStyle:
    ENDING_DEFAULT = 0
    ENDING_ARROW = 1
    BOTH_ARROW = 2
    BEGIN_ARROW = 3


# def scale_vector(x,y,new_length):
#    length = math.sqrt(x*x + y*y)
#    return (new_length*x/length, new_length*y/length)

def path_to_sprite(points, line_width=default_line_width, line_pattern=black_pattern,
                   ending_style=PathEndingStyle.ENDING_DEFAULT, dash=False, offset=0, dashArr=[1,4]):

    xmin = ymin = 0
    for pt in points:
        x,y = pt
        xmin = min([x, xmin])
        ymin = min([y, ymin])

    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    max_y = 0
    max_x = 0

    ctx.set_line_width(line_width)
    ctx.set_source(line_pattern)
    if ending_style is PathEndingStyle.BEGIN_ARROW or ending_style is PathEndingStyle.BOTH_ARROW:
        (x1, y1) = points[0]
        angle = numpy.arctan2(y1, x1)
        angle1 = angle + math.pi / 6
        angle2 = angle - math.pi / 6

        (arrx1, arry1) = (math.cos(angle1) * 4 * line_width, math.sin(angle1) * 4 * line_width)
        (arrx2, arry2) = (math.cos(angle2) * 4 * line_width, math.sin(angle2) * 4 * line_width)

        # (arrx1, arry1) = scale_vector(x1*0.7 + y1*0.3, y1*0.7 - x1*0.3, 4*line_width)
        # (arrx2, arry2) = scale_vector(x1*0.7 - y1*0.3, y1*0.7 + x1*0.3, 4*line_width)
        ctx.move_to(arrx1, arry1)
        ctx.line_to(0, 0)
        ctx.line_to(arrx2, arry2)
        ctx.stroke()

    ctx.move_to(0, 0)
    if dash:
        ctx.set_dash(dashArr, offset)
    for (x, y) in points:
        ctx.line_to(x, y)
        max_y = max([max_y, y])
        max_x = max([max_x, x])
    ctx.stroke()
    if dash:
        ctx.set_dash([])

    if ending_style is PathEndingStyle.ENDING_ARROW or ending_style is PathEndingStyle.BOTH_ARROW:
        (x1, y1) = points[len(points) - 1]
        x0 = y0 = 0
        if len(points) > 1:
            (x0, y0) = points[len(points) - 2]
        (dx, dy) = (x1 - x0, y1 - y0)
        angle = numpy.arctan2(dy, dx)
        (angle1, angle2) = (angle + math.pi / 6 + math.pi, angle - math.pi / 6 + math.pi)
        (arrx1, arry1) = (math.cos(angle1) * 4 * line_width, math.sin(angle1) * 4 * line_width)
        (arrx2, arry2) = (math.cos(angle2) * 4 * line_width, math.sin(angle2) * 4 * line_width)
        # (arrx1, arry1) = scale_vector(dx*0.7 + dy*0.3, dy*0.7 - dx*0.3, 4*line_width)
        # (arrx2, arry2) = scale_vector(dx*0.7 - dy*0.3, dy*0.7 + dx*0.3, 4*line_width)
        ctx.move_to(x1 + arrx1, y1 + arry1)
        ctx.line_to(x1, y1)
        ctx.line_to(x1 + arrx2, y1 + arry2)
        ctx.stroke()
    
    if xmin < 0 or ymin < 0:
        surf2 = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        ctx = cairo.Context(surf2)
        ctx.set_source_surface(surf, -xmin, -ymin)
        ctx.paint()
        out = Sprite(max_y - ymin, max_x - xmin, surf2)
    else:
        out = Sprite(max_y, max_x, surf)
    return out

def circular_arrow(radius, begin_angle, end_angle, line_width=default_line_width, line_pattern=black_pattern,
                   ending_style=PathEndingStyle.ENDING_DEFAULT):
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    ctx.set_source(line_pattern)
    ctx.set_line_width(line_width)

    ctx.arc(radius, radius, radius, begin_angle, end_angle)
    ctx.stroke()

    if ending_style is PathEndingStyle.BEGIN_ARROW or ending_style is PathEndingStyle.BOTH_ARROW:
        angle = begin_angle + math.pi/2
        angle1 = angle + math.pi / 6
        angle2 = angle - math.pi / 6

        (arrx1, arry1) = (math.cos(angle1) * 4* line_width, math.sin(angle1) * 4 * line_width)
        (arrx2, arry2) = (math.cos(angle2) * 4 * line_width, math.sin(angle2) * 4 * line_width)

        # (arrx1, arry1) = scale_vector(x1*0.7 + y1*0.3, y1*0.7 - x1*0.3, 4*line_width)
        # (arrx2, arry2) = scale_vector(x1*0.7 - y1*0.3, y1*0.7 + x1*0.3, 4*line_width)
        (x1,y1) = (radius + radius*math.cos(begin_angle), radius + radius*math.sin((begin_angle)))
        ctx.move_to(x1 + arrx1, y1 + arry1)
        ctx.line_to(x1, y1)
        ctx.line_to(x1 + arrx2, y1 + arry2)
        ctx.stroke()

    if ending_style is PathEndingStyle.ENDING_ARROW or ending_style is PathEndingStyle.BOTH_ARROW:
        angle = end_angle + math.pi/2
        (angle1, angle2) = (angle + math.pi / 6 + math.pi, angle - math.pi / 6 + math.pi)
        (arrx1, arry1) = (math.cos(angle1) * 4 * line_width, math.sin(angle1) * 4 * line_width)
        (arrx2, arry2) = (math.cos(angle2) * 4 * line_width, math.sin(angle2) * 4 * line_width)
        # (arrx1, arry1) = scale_vector(dx*0.7 + dy*0.3, dy*0.7 - dx*0.3, 4*line_width)
        # (arrx2, arry2) = scale_vector(dx*0.7 - dy*0.3, dy*0.7 + dx*0.3, 4*line_width)
        (x1, y1) = (radius + radius*math.cos(end_angle), radius + radius*math.sin(end_angle))
        ctx.move_to(x1 + arrx1, y1 + arry1)
        ctx.line_to(x1, y1)
        ctx.line_to(x1 + arrx2, y1 + arry2)
        ctx.stroke()

    out = Sprite(2*radius, 2*radius, surf)
    return out

def circle_to_sprite(radius, line_width=default_line_width, line_pattern=black_pattern, fill_pattern=white_pattern,
                     border=True):
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    ctx.set_source(fill_pattern)
    ctx.arc(radius, radius, radius, 0, 2 * math.pi)
    ctx.close_path()
    ctx.fill()

    if border:
        ctx.set_line_width(line_width)
        ctx.set_source(black_pattern)
        ctx.arc(radius, radius, radius, 0, 2 * math.pi)
        ctx.stroke()

    out = Sprite(2 * radius, 2 * radius, surf)
    return out


def rect_to_sprite(height, width, line_width=0.5, line_pattern=black_pattern, fill_pattern=white_pattern, dash=False):
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    if fill_pattern is not None:
        ctx.set_source(fill_pattern)
        ctx.move_to(0, 0)
        ctx.line_to(width, 0)
        ctx.line_to(width, height)
        ctx.line_to(0, height)
        ctx.close_path()
        ctx.fill()

    if line_pattern is not None:
        if dash:
            ctx.set_dash([3, 1])
        ctx.set_line_width(line_width)
        ctx.set_source(line_pattern)
        ctx.move_to(0, 0)
        ctx.line_to(width, 0)
        ctx.line_to(width, height)
        ctx.line_to(0, height)
        ctx.close_path()
        ctx.stroke()
        if dash:
            ctx.set_dash([])

    out = Sprite(height, width, surf)
    return out


def poly_to_sprite(pts, line_width=0.5, line_pattern=black_pattern, fill_pattern=white_pattern, dash=False):
    assert(len(pts) > 1)
    for pt in pts:
        assert(pt[0] >= 0 and pt[1] >= 0)
    
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    max_y = 0
    max_x = 0

    if fill_pattern is not None:
        ctx.set_source(fill_pattern)
    ctx.move_to(pts[0][0], pts[0][1])
    for pt in pts[1:]:
        ctx.line_to(pt[0], pt[1])
        max_x = max([max_x, pt[0]])
        max_y = max([max_y, pt[1]])
    ctx.close_path()
    if fill_pattern is not None:
        ctx.fill()

    if line_pattern is not None:
        if dash:
            ctx.set_dash([2, 2])
        ctx.set_line_width(line_width)
        ctx.set_source(line_pattern)
        ctx.move_to(pts[0][0],pts[0][1])
        for pt in pts[1:]:
            ctx.line_to(pt[0], pt[1])
        ctx.close_path()
        ctx.stroke()
        if dash:
            ctx.set_dash([])

    out = Sprite(max_y, max_x, surf)
    return out

def textbox(text, is_math=True, width = None, height = None, line_pattern=black_pattern, line_width=default_line_width, col=black_pattern):
    txt = latex_to_sprite(text, is_math=is_math, col=col)
    if width is None:
        width = txt.width
    if height is None:
        height = txt.height
    assert(width >= txt.width)
    assert(height >= txt.height)
    out = CompositeGraphic()
    out.draw_left_top(rect_to_sprite(height, width, line_pattern=line_pattern, line_width=line_width))
    out.draw_center(txt, width/2, height/2)
    return out

class AffinityGraph:
    def __init__(self, n_input, m_input, speeds=None):
        self.n = n_input
        self.m = m_input
        if speeds is None:
            self.speeds = [1.0 for j in range(m_input)]
        else:
            self.speeds = speeds
        self.edges = []
        self.assignments = []
        self.np_assignments = []
        self.links = []

    def add_edge(self, task_id, proc_id):
        assert (task_id < self.n and proc_id < self.m)
        self.edges.append((task_id, proc_id))

    def add_edges(self, task_id, proc_ids):
        for j in proc_ids:
            self.add_edge(task_id, j)

    def make_global(self, task_id):
        for j in range(self.m):
            self.add_edge(task_id, j)

    def add_assign(self, task_id, proc_id):
        assert ((task_id, proc_id) in self.edges)
        self.assignments.append((task_id, proc_id))

    def add_np_assign(self, task_id, proc_id):
        assert ((task_id, proc_id) in self.edges)
        self.np_assignments.append((task_id, proc_id))

    def add_link(self, task_id, proc_id, rev=False):
        assert ((task_id, proc_id) in self.edges)
        self.links.append((task_id, proc_id, rev))

    def clear_assign(self):
        self.assignments = []
        self.np_assignments = []
        self.links = []
    
    def clear_link(self):
        self.links = []


def draw_ag(ag, start_from_zero=False):
    ag_graphic = CompositeGraphic()

    task_labels = [latex_to_sprite('\\tau_' + str(i if start_from_zero else i + 1)) for i in range(ag.n)]
    proc_labels = [latex_to_sprite('\\pi_' + str(i if start_from_zero else i + 1)) for i in range(ag.m)]
    task_label_width_max = max([task_label.width for task_label in task_labels])
    task_label_height_max = max([task_label.height for task_label in task_labels])
    task_length = 1.2 * max([task_label_height_max, task_label_width_max])
    proc_label_width_max = max([proc_label.width for proc_label in proc_labels])
    proc_label_height_max = max([proc_label.height for proc_label in proc_labels])
    proc_radius = 0.8 * max([proc_label_height_max, proc_label_width_max])

    assert(max(ag.speeds) == 1.0)
    proc_radius *= math.sqrt(max(ag.speeds)/min(ag.speeds))

    square = rect_to_sprite(task_length, task_length)

    # Need to know how big a circle is going to be, even though this one won't be used
    circle = circle_to_sprite(proc_radius)
    bigger_height = max([square.height, circle.height])

    total_width = max([circle.width * (ag.m) * 1.1, square.width * (ag.n) * 1.1])
    step_size_proc = float(total_width) / (ag.m)
    for j in range(ag.m):
        this_radius = proc_radius * math.sqrt(ag.speeds[j])
        circle = circle_to_sprite(this_radius, fill_pattern=sources_dict[j])
        inner_circle = circle_to_sprite(this_radius * 0.6, fill_pattern=white_pattern, border=False)
        ag_graphic.draw_center(circle, step_size_proc * (j + 0.5), proc_radius)
        ag_graphic.draw_center(inner_circle, step_size_proc * (j + 0.5), proc_radius)
        ag_graphic.draw_center(proc_labels[j], step_size_proc * (j + 0.5), proc_radius)

    step_size_task = float(total_width) / (ag.n)
    for i in range(ag.n):
        ag_graphic.draw_center(square, step_size_task * (i + 0.5), 2 * bigger_height + task_length / 2)
        ag_graphic.draw_center(task_labels[i], step_size_task * (i + 0.5), 2 * bigger_height + task_length / 2)

    for (task_id, proc_id) in ag.edges:
        x1 = step_size_task * (task_id + 0.5)
        y1 = 2 * bigger_height
        x0 = step_size_proc * (proc_id + 0.5)
        y0 = proc_radius*(1 + math.sqrt(ag.speeds[proc_id]))

        edge_sprite = path_to_sprite([(x1 - x0, y1 - y0)])
        ag_graphic.draw_center(edge_sprite, 0.5*(x0+x1), 0.5*(y0+y1))
        if (task_id, proc_id, False) in ag.links:
            edge_arrow = path_to_sprite([(x1 - x0, y1 - y0)], line_width=default_line_width*7, ending_style=PathEndingStyle.BEGIN_ARROW, line_pattern=transp_red_pattern)
            ag_graphic.draw_center(edge_arrow, 0.5*(x0+x1), 0.5*(y0+y1))
        if (task_id, proc_id, True) in ag.links:
            edge_arrow = path_to_sprite([(x1 - x0, y1 - y0)], line_width=default_line_width*7, ending_style=PathEndingStyle.ENDING_ARROW, line_pattern=red_pattern)
            ag_graphic.draw_center(edge_arrow, 0.5*(x0+x1), 0.5*(y0+y1))
        if (task_id, proc_id) in ag.np_assignments:
            edge_highlight_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=transp_pattern,
                                                   line_width=default_line_width * 10)
            ag_graphic.draw_center(edge_highlight_sprite, 0.5*(x0+x1), 0.5*(y0+y1))
            edge_dash_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=black_pattern,
                                              line_width=default_line_width * 10, dash=True, offset=0.5)
            ag_graphic.draw_center(edge_dash_sprite, 0.5*(x0+x1), 0.5*(y0+y1))
        elif (task_id, proc_id) in ag.assignments:
            edge_highlight_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=transp_pattern,
                                                   line_width=default_line_width * 10)
            ag_graphic.draw_center(edge_highlight_sprite, 0.5*(x0+x1), 0.5*(y0+y1))

    return ag_graphic


class Discont:
    OPEN = 1
    CLOSED = 2


# assume line contains multiple lines
# holes are list of tuples (x,y,Discont type)
# doesn't consider negative points
def draw_line_graph(line_data, x_axis = None, y_axis = None, holes=[], scale_x=10.0, scale_y=10.0, xmax = 0, ymax = 0, colors = None, ystep = 5, xstep = 5, stroke = default_line_width):
    line_graphic = CompositeGraphic()

    x_offset = 5
    y_offset = 5
    extra_line = 10
    hole_size = 1

    if colors is None or len(colors) != len(line_data):
        colors = [black_pattern for line in line_data]

    
    

    for line in line_data:
        for (x, y) in line:
            # xmin = min([xmin, x])
            xmax = max([xmax, x])
            # ymin = min([ymin, y])
            ymax = max([ymax, y])

    xlabels = [latex_to_sprite(str(t)) for t in range(0, xmax + 1, xstep)]
    ylabels = [latex_to_sprite(str(t)) for t in range(0, ymax + 1, ystep)]

    xlabelHeight = max([label.height for label in xlabels])
    ylabelWidth = max([label.width for label in ylabels])

    axes = path_to_sprite([(0, y_offset + ymax * scale_y + extra_line),
                           (x_offset + xmax * scale_x + extra_line, y_offset + ymax * scale_y + extra_line)],
                          ending_style=PathEndingStyle.BOTH_ARROW)

    line_graphic.draw_left_top(axes, ylabelWidth, 0)

    vert_arrow = path_to_sprite([(0, y_offset + ymax * scale_y + extra_line)], line_pattern=transp_pattern,
                                ending_style=PathEndingStyle.BEGIN_ARROW)
    horiz_arrow = path_to_sprite([(x_offset + xmax * scale_x + extra_line, 0)], line_pattern=transp_pattern,
                                 ending_style=PathEndingStyle.ENDING_ARROW)

    x_axis_tick_height = 0
    for t in range(xmax + 1):
        line_graphic.draw_center_bottom(vert_arrow, ylabelWidth + x_offset + t * scale_x,
                                        y_offset + ymax * scale_y + extra_line)
    for t in range(ymax + 1):
        line_graphic.draw_left_middle(horiz_arrow, ylabelWidth, extra_line + (ymax - t) * scale_y)
    for t in range(0, xmax + 1, xstep):
        tick_sprite = latex_to_sprite(str(t))
        line_graphic.draw_center_top(tick_sprite, ylabelWidth + x_offset + t * scale_x,
                                     y_offset + ymax * scale_y + extra_line)
        x_axis_tick_height = max([x_axis_tick_height, tick_sprite.height])
    for t in range(0, ymax + 1, ystep):
        line_graphic.draw_right_middle(latex_to_sprite(str(t)), ylabelWidth, extra_line + (ymax - t) * scale_y)

    if x_axis is not None:
        x_axis_sprite = latex_to_sprite(x_axis, is_math=False)
        line_graphic.draw_center_top(x_axis_sprite, ylabelWidth + x_offset + xmax * scale_x / 2,
                                     y_offset + ymax * scale_y + extra_line + x_axis_tick_height)

    # we're assuming that lines are ordered by x here
    for line, color in zip(line_data, colors):
        (x0, y0) = line[0]
        smallest_y = min([y for _, y in line])
        converted_line = [((x - x0) * scale_x, (y - y0) * scale_y) for (x, y) in line]
        flipped_line = [(x, -y) for (x, y) in converted_line]
        flipped_line = flipped_line[1:]
        line_path = path_to_sprite(flipped_line, line_pattern=color, line_width=stroke)
        line_graphic.draw_left_bottom(line_path, ylabelWidth + x_offset + x0 * scale_x,
                                      (ymax - smallest_y) * scale_y + extra_line)

    for (x, y, holetype) in holes:
        if holetype is Discont.OPEN:
            line_graphic.draw_center(circle_to_sprite(hole_size, fill_pattern=white_pattern),
                                     ylabelWidth + x_offset + x * scale_x, extra_line + (ymax - y) * scale_y)
        if holetype is Discont.CLOSED:
            line_graphic.draw_center(circle_to_sprite(hole_size, fill_pattern=black_pattern),
                                     ylabelWidth + x_offset + x * scale_x, extra_line + (ymax - y) * scale_y)
    
    line_graphic2 = line_graphic

    if y_axis is not None:
        line_graphic2 = CompositeGraphic()
        y_axis_sprite = latex_to_sprite(y_axis, is_math=False)
        line_graphic2.draw_left_middle(y_axis_sprite, 0, y_offset + ymax * scale_y / 2 + extra_line)
        line_graphic2.draw_left_top(line_graphic, y_axis_sprite.width, 0)

    return line_graphic2

def png_to_sprite(fname: str):
    surface = cairo.ImageSurface.create_from_png(fname)
    return Sprite(surface.get_height(), surface.get_width(), surface)


def cairo_test():
    surface = cairo.PDFSurface("test.pdf", 100, 100)
    ctx = cairo.Context(surface)
    ctx.set_source(cairo.SolidPattern(0.5, 0.5, 0))
    ctx.move_to(0, 0)
    ctx.rectangle(0, 0, 100, 100)
    ctx.fill()

    x2 = latex_to_sprite('x^2 + 2y')

    x2.draw_center(ctx, 50, 50)

    ctx.show_page()


#sources_dict = dict([(0, white_pattern), (1, gray_pattern), (2, dark_gray_pattern), (3, black_pattern)])
sources_dict = dict([(0, red_pattern), (1, blue_pattern), (2, orange_pattern), (3, purple_pattern), (4, green_diag_pattern), (5, black_pattern)])

def rotate_sprite(sprite: Sprite) -> Sprite:
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)
    ctx.rotate(-math.pi/2)
    ctx.set_source_surface(sprite.surface, -sprite.width, 0)
    ctx.paint()
    return Sprite(sprite.width, sprite.height, surf)


def scale_to_fit(draw: Drawable, width: float, height:float, scale:List = None) -> Sprite:
    width_factor = width/draw.width
    height_factor = height/draw.height
    factor = min([width_factor, height_factor])

    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)
    ctx.save()
    ctx.scale(factor, factor)
    draw.update_published_surface()
    draw.draw_published(ctx, 0, 0)
    ctx.restore()
    if scale is not None:
        scale[0] = factor
    return Sprite(draw.height*factor, draw.width*factor, surf)


def transparency(draw:Drawable, alpha: float) -> Sprite:
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)
    draw.update_published_surface()
    draw.draw_published(ctx, 0, 0, alpha=alpha)
    return Sprite(draw.height, draw.width, surf)

def pad(draw:Drawable, up: int = 0, down: int = 0, left: int = 0, right: int = 0) -> CompositeGraphic:
    cg = CompositeGraphic()
    cg.draw_left_top(draw, left, up)
    cg.height = draw.height + up + down
    cg.width = draw.width + left + right
    return cg


def init():
    #phi1 = latex_to_sprite(r"Ideal", is_math=False)
    #print(get_pdf_page_size(tmp_dir + 'tmp.pdf'))
    return 0




init()