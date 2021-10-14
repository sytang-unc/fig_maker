# used to render graphics
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

    def draw_published(self, ctx, x, y) -> None:
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

    def draw_published(self, ctx, x, y):
        ctx.set_source_surface(self.surface, x, y)
        ctx.paint()


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

    def draw_published(self, ctx, x, y):
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

    def draw_published(self, ctx, x, y):
        ctx.set_source_surface(self.pub_surface, x, y)
        ctx.paint()


"""
A PaperGraphic corresponds to a figure or subfigure
No one should be calling is_published or is_cached on this. This doesn't get drawn onto anything either
"""



class PaperGraphic(CompositeGraphic):
    def __init__(self, name_input):
        CompositeGraphic.__init__(self, name_input)

    def publish(self, format="pdf"):
        CompositeGraphic.update_published_surface(self)
        if format == "pdf":
            surf = cairo.PDFSurface(self.name + '.pdf', self.width + 10, self.height + 10)
            #surf = cairo.SVGSurface(self.name + '.svg', self.width + 10, self.height + 10)
            ctx = cairo.Context(surf)
            ctx.set_source_surface(self.pub_surface, 5, 5)
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
    
    def draw_publish(self, src, format="pdf"):
        self.draw_left_top(src)
        self.publish(format)

    def preview(self):
        CompositeGraphic.update_cached_surface(self)
        surf = cairo.ImageSurface(cairo.Format.ARGB32, int(preview_scale * self.width * 1.2),
                                  int(preview_scale * self.height * 1.2))
        ctx = cairo.Context(surf)
        #ctx.translate(0.1 * self.width, 0.1 * self.height)
        ctx.set_source_surface(self.cached_surface, preview_scale * 0.1 * self.width, preview_scale * 0.1 * self.height)
        ctx.paint()
        surf.write_to_png(self.name + '.png')


"""
The information needed to produce a schedule figure. Assumes indexing of tasks/procs begins at 0
"""


class SchedData:
    def __init__(self, n_input, m_input, sched_end_min=0, budg_max=0):
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
        for i in range(self.n):
            self.executions[i] = []
            self.np_executions[i] = []
            self.releases[i] = []
            self.pseudo_releases[i] = []
            self.completions[i] = []
            self.deadlines[i] = []
            self.pseudo_deadlines[i] = []
            self.annotations[i] = []

    def add_execution(self, task_id, proc, start_time, end_time):
        assert (proc < self.m and task_id < self.n)
        assert (start_time < end_time and start_time >= 0)
        self.executions[task_id].append((proc, start_time, end_time))

        self.sched_end = max([self.sched_end, end_time])

    def add_np_execution(self, task_id, proc, start_time, end_time):
        assert (proc < self.m and task_id < self.n)
        assert (start_time < end_time and start_time >= 0)
        self.np_executions[task_id].append((proc, start_time, end_time))

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


def draw_sched(schedule, start_from_zero=False, task_identifiers=None, scale_time_to_dots=10, time_axis_inc=5, sched_start_time=0, draw_time=True, transp_bg=True):
    sched_graphic = draw_sched_help(schedule, start_from_zero, task_identifiers, scale_time_to_dots, time_axis_inc, sched_start_time, draw_time)
    if transp_bg:
        return sched_graphic
    tot_graphic = CompositeGraphic()
    tot_graphic.draw_left_top(rect_to_sprite(sched_graphic.height, sched_graphic.width))
    tot_graphic.draw_left_top(sched_graphic)
    return tot_graphic

"""
Render a given SchedData
INPUT
    schedule - SchedData object
    start_from_zero - do task/proc subscripts start from 0 or 1
"""


def draw_sched_help(schedule, start_from_zero=False, task_identifiers=None, scale_time_to_dots=10, time_axis_inc=5, sched_start_time=0, draw_time=True):
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
        sched_graphic.draw_left_bottom(task_axis, task_label_width_max * horiz_offset_scale, y_botts[i])
        # draw each block of execution for this task
        for (proc, start_time, end_time) in schedule.executions[i]:
            sched_block = rect_to_sprite(task_label_height_max * exec_vertical_scale,
                                         scale_time_to_dots * (end_time - start_time), fill_pattern=sources_dict[proc])
            sched_graphic.draw_left_bottom(sched_block,
                                           offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (start_time - sched_start_time),
                                           y_botts[i])
        for (proc, start_time, end_time) in schedule.np_executions[i]:
            sched_block = rect_to_sprite(task_label_height_max * exec_vertical_scale,
                                         scale_time_to_dots * (end_time - start_time), fill_pattern=sources_dict[proc],
                                         dash=True)
            sched_graphic.draw_left_bottom(sched_block,
                                           offset_0 + task_label_width_max * horiz_offset_scale + scale_time_to_dots * (start_time - sched_start_time),
                                           y_botts[i])

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
                            ('\\tau_s', ('tau_s.svg', 9.784, 12.825)), \
                            ('\\pi_0', ('pi_0.svg', 9.784, 14.148)), \
                            ('\\pi_1', ('pi_1.svg', 9.784, 14.148)), \
                            ('\\pi_2', ('pi_2.svg', 9.784, 14.148)), \
                            ('\\pi_3', ('pi_3.svg', 9.784, 14.148)), \
                            ('0', ('0.svg', 10.42, 8.981)), \
                            ('5', ('5.svg', 10.42, 8.981)), \
                            ('10', ('10.svg', 10.42, 13.963)), \
                            ('15', ('15.svg', 10.42, 13.963)), \
                            ('20', ('20.svg', 10.42, 13.963)), \
                            ('25', ('25.svg', 10.42, 13.963)), \
                            ('30', ('30.svg', 10.42, 13.963)), \
                            ('Time', ('Time.svg', 10.808, 26.693)), \
                            ('Release', ('Release.svg', 10.918, 36.296)), \
                            ('Deadline', ('Deadline.svg', 10.918, 42.052)), \
                            ('Completion', ('Completion.svg', 12.856, 54.367))])

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


def latex_to_sprite(text, is_math=True):
    svgfile = ''
    height = 0
    width = 0
    if text in latex_frequent_dict.keys():
        (svgname, height, width) = latex_frequent_dict[text]
        svgfile = freq_dir + svgname
    else:
        with open(tmp_dir + 'tmp.tex', 'w') as tmp_tex:
            tmp_tex.write('\\documentclass[varwidth, border=2]{standalone}\n')
            tmp_tex.write('\\usepackage{amsmath}\n')
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

    out = Sprite(height, width, surface, text)

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


black_pattern = cairo.SolidPattern(0, 0, 0)
white_pattern = cairo.SolidPattern(1, 1, 1)
gray_pattern = cairo.SolidPattern(0.66, 0.66, 0.66)
dark_gray_pattern = cairo.SolidPattern(0.33, 0.33, 0.33)

sol_blue_pattern = cairo.SolidPattern(0,0,1)
red_pattern = cairo.SolidPattern(211/256,94/256,96/256)
blue_pattern = cairo.SolidPattern(114/256,147/256,203/256)
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

green_pattern = cairo.SolidPattern(0,1,0)


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
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    height = 0
    width = 0

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
        height = max([height, y])
        width = max([width, x])
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

    out = Sprite(height, width, surf)
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


def poly_to_sprite(pts, line_width=0.5, line_pattern=black_pattern, fill_pattern=white_pattern):
    assert(len(pts) > 1)
    for pt in pts:
        assert(pt[0] >= 0 and pt[1] >= 0)
    
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)

    max_y = 0
    max_x = 0

    ctx.set_source(fill_pattern)
    ctx.move_to(pts[0][0], pts[0][1])
    for pt in pts[1:]:
        ctx.line_to(pt[0], pt[1])
        max_x = max([max_x, pt[0]])
        max_y = max([max_y, pt[1]])
    ctx.close_path()
    ctx.fill()

    if line_pattern is not None:
        ctx.set_line_width(line_width)
        ctx.set_source(line_pattern)
        ctx.move_to(pts[0][0],pts[0][1])
        for pt in pts[1:]:
            ctx.line_to(pt[0], pt[1])
        ctx.close_path()
        ctx.stroke()

    out = Sprite(max_y, max_x, surf)
    return out

class AffinityGraph:
    def __init__(self, n_input, m_input):
        self.n = n_input
        self.m = m_input
        self.edges = []
        self.assignments = []
        self.np_assignments = []
        self.links = []

    def add_edge(self, task_id, proc_id):
        assert (task_id < self.n and proc_id < self.m)
        self.edges.append((task_id, proc_id))

    def add_assign(self, task_id, proc_id):
        assert ((task_id, proc_id) in self.edges)
        self.assignments.append((task_id, proc_id))

    def add_np_assign(self, task_id, proc_id):
        assert ((task_id, proc_id) in self.edges)
        self.np_assignments.append((task_id, proc_id))

    def add_link(self, task_id, proc_id):
        assert ((task_id, proc_id) in self.edges)
        self.links.append((task_id, proc_id))

    def clear_assign(self):
        self.assignments = []
        self.np_assignments = []
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
    proc_radius = 0.8 * max([task_label_height_max, task_label_width_max])

    square = rect_to_sprite(task_length, task_length)

    # Need to know how big a circle is going to be, even though this one won't be used
    circle = circle_to_sprite(proc_radius)
    bigger_height = max([square.height, circle.height])

    total_width = max([circle.width * (ag.m) * 1.4, square.width * (ag.n) * 1.4])
    step_size_proc = float(total_width) / (ag.m)
    for j in range(ag.m):
        circle = circle_to_sprite(proc_radius, fill_pattern=sources_dict[j])
        inner_circle = circle_to_sprite(proc_radius * 0.6, fill_pattern=white_pattern, border=False)
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
        y0 = 2 * proc_radius

        edge_sprite = path_to_sprite([(x1 - x0, y1 - y0)])
        ag_graphic.draw_left_top(edge_sprite, x0, y0)
        if (task_id, proc_id) in ag.links:
            edge_arrow = path_to_sprite([(x1 - x0, y1 - y0)], line_width=1.0, ending_style=PathEndingStyle.BEGIN_ARROW)
            ag_graphic.draw_left_top(edge_arrow, x0, y0)
        if (task_id, proc_id) in ag.np_assignments:
            edge_highlight_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=transp_pattern,
                                                   line_width=default_line_width * 10)
            ag_graphic.draw_left_top(edge_highlight_sprite, x0, y0)
            edge_dash_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=black_pattern,
                                              line_width=default_line_width * 10, dash=True, offset=0.5)
            ag_graphic.draw_left_top(edge_dash_sprite, x0, y0)
        elif (task_id, proc_id) in ag.assignments:
            edge_highlight_sprite = path_to_sprite([(x1 - x0, y1 - y0)], line_pattern=transp_pattern,
                                                   line_width=default_line_width * 10)
            ag_graphic.draw_left_top(edge_highlight_sprite, x0, y0)

    return ag_graphic


class Discont:
    OPEN = 1
    CLOSED = 2


# assume line contains multiple lines
# holes are list of tuples (x,y,Discont type)
# doesn't consider negative points
def draw_line_graph(line_data, x_axis, y_axis, holes=[], scale_x=10.0, scale_y=10.0):
    line_graphic = CompositeGraphic()

    x_offset = 5
    y_offset = 5
    extra_line = 5
    hole_size = 1

    x_axis_sprite = latex_to_sprite(x_axis, is_math=False)
    y_axis_sprite = latex_to_sprite(y_axis, is_math=False)

    xmax = ymax = 0
    for line in line_data:
        for (x, y) in line:
            # xmin = min([xmin, x])
            xmax = max([xmax, x])
            # ymin = min([ymin, y])
            ymax = max([ymax, y])

    xlabels = [latex_to_sprite(str(t)) for t in range(0, xmax + 1, 5)]
    ylabels = [latex_to_sprite(str(t)) for t in range(0, ymax + 1, 5)]

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
    for t in range(0, xmax + 1, 5):
        tick_sprite = latex_to_sprite(str(t))
        line_graphic.draw_center_top(tick_sprite, ylabelWidth + x_offset + t * scale_x,
                                     y_offset + ymax * scale_y + extra_line)
        x_axis_tick_height = max([x_axis_tick_height, tick_sprite.height])
    for t in range(0, ymax + 1, 5):
        line_graphic.draw_right_middle(latex_to_sprite(str(t)), ylabelWidth, extra_line + (ymax - t) * scale_y)

    line_graphic.draw_center_top(x_axis_sprite, ylabelWidth + x_offset + xmax * scale_x / 2,
                                 y_offset + ymax * scale_y + extra_line + x_axis_tick_height)

    # we're assuming that lines are ordered by x here
    for line in line_data:
        (x0, y0) = line[0]
        converted_line = [((x - x0) * scale_x, (y - y0) * scale_y) for (x, y) in line]
        flipped_line = [(x, -y) for (x, y) in converted_line]
        flipped_line = flipped_line[1:]
        line_path = path_to_sprite(flipped_line)
        line_graphic.draw_left_bottom(line_path, ylabelWidth + x_offset + x0 * scale_x,
                                      (ymax - y0) * scale_y + extra_line)

    for (x, y, holetype) in holes:
        if holetype is Discont.OPEN:
            line_graphic.draw_center(circle_to_sprite(hole_size, fill_pattern=white_pattern),
                                     ylabelWidth + x_offset + x * scale_x, extra_line + (ymax - y) * scale_y)
        if holetype is Discont.CLOSED:
            line_graphic.draw_center(circle_to_sprite(hole_size, fill_pattern=black_pattern),
                                     ylabelWidth + x_offset + x * scale_x, extra_line + (ymax - y) * scale_y)

    line_graphic2 = CompositeGraphic()
    line_graphic2.draw_left_middle(y_axis_sprite, 0, y_offset + ymax * scale_y / 2 + extra_line)
    line_graphic2.draw_left_top(line_graphic, y_axis_sprite.width, 0)

    return line_graphic2


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
sources_dict = dict([(0, red_pattern), (1, blue_pattern), (2, orange_pattern), (3, purple_pattern)])

def rotate_sprite(sprite: Sprite) -> Sprite:
    surf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(surf)
    ctx.rotate(-math.pi/2)
    ctx.set_source_surface(sprite.surface, -sprite.width, 0)
    ctx.paint()
    return Sprite(sprite.width, sprite.height, surf)

def init():
    return 0


init()