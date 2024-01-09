from __future__ import annotations
import copy #should be removed when updating to python >= 3.10

from typing import Callable, Dict, List, Tuple
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np


plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

SCALEDOWN_DISPLAY_DATA: float = 4

"""
The information needed to produce a schedule figure. Assumes indexing of tasks/procs begins at 0
"""
class SchedData:
    def __init__(self, n_input:int, m_input:int, sched_end_min:float=0, budg_max:float=0, budg_maxs:List[float]=None, s_max = 1.0):
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

        self.budgets = {}
        if budg_maxs is None:
            self.budg_maxs = [0 for i in range(n_input)]
        else:
            assert(len(budg_maxs) == n_input)
            self.budg_maxs = budg_maxs
        if budg_max != 0:
            self.budg_maxs[0] = budg_max
        self.b_annotations = {}

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
            self.budgets[i] = []
            self.b_annotations[i] = []
        self.annotations[-1] = []

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


    def add_annotation(self, task_id, text, start, end=-1, arrowstyle:str='<->', budget:bool=False):
        assert (task_id < self.n)
        if end != -1:
            assert (start < end)
            assert (arrowstyle == '<->' or arrowstyle == '->')
        else:
            end = start
        
        self.sched_end = max([self.sched_end, end])

        if budget:
            self.b_annotations[task_id].append((text, start, end, arrowstyle))
        else:
            self.annotations[task_id].append((text, start, end, arrowstyle))
    
    def add_budget(self, start, end, init, final, idx:int=0, fill:bool=True):
        assert(start < end)
        assert(0 <= idx < self.n)

        self.sched_end = max([self.sched_end, end])

        self.budgets[idx].append((start, end, init, final, fill))
        self.budg_maxs[idx] = max([self.budg_maxs[idx], init, final])


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

def draw_rect(ax: Axes, start:int, end:int, height:int, colidx:int) -> None:
    assert( 0 <= colidx <= 2 )
    if colidx == 0:
        ax.fill([start, end, end, start], [0, 0, height, height], facecolor='red', edgecolor='k')
    elif colidx == 1:
        ax.fill([start, end, end, start], [0, 0, height, height], facecolor='blue', hatch='//', edgecolor='k')
    elif colidx == 2:
        ax.fill([start, end, end, start], [0, 0, height, height], facecolor='orange', hatch='--', edgecolor='k')

def draw_circ(ax: Axes, center:float, radius:float, colidx:int) -> None:
    assert( 0 <= colidx <= 2 )
    if colidx == 0:
        ax.add_patch(Circle(center,radius, facecolor='red', edgecolor='k'))
    elif colidx == 1:
        ax.add_patch(Circle(center,radius, facecolor='blue', hatch='//', edgecolor='k'))
    elif colidx == 2:
        ax.add_patch(Circle(center,radius, facecolor='orange', hatch='--', edgecolor='k'))

"""
Render a given SchedData
INPUT
    schedule - SchedData object
    start_from_zero - do task/proc subscripts start from 0 or 1
    task_padding - target 4 + 3*number of chars, can't use len cuz \texttt counts toward strlen
"""
def draw_sched_help(schedule: SchedData, 
                    start_from_zero: bool =False, 
                    task_identifiers:List[str]=None, 
                    budg_identifiers:Dict[int,str]=None,
                    budg_padding:Dict[int,float]=None,
                    task_padding: List[float]=None,
                    time_axis_inc:float=5, 
                    sched_start_time:int=0, 
                    draw_time:bool=True,
                    draw_time_labels:Dict[int,List[Tuple[float,str]]]=None,
                    time_discont:Dict[int,List[Tuple[float,float]]]=None, 
                    draw_time_txt:bool=True, 
                    draw_time_txt_label:List[str]=None, 
                    filename:str=None, 
                    scale_time:float=1.0,
                    visible:List[bool]=None,
                    mscale:float=7.0,
                    force_annot_height:bool=False) -> None:
    if visible is None:
        visible = [True for _ in range(schedule.n)]
    while len(visible) < schedule.n:
        visible.append(True)

    tot_budg = sum([0] + [1 for i in range(schedule.n) if len(schedule.budgets[i])])
    annot_height = (1 if len(schedule.annotations[-1]) else 0) + sum([0] + [1 for i in range(schedule.n) if len(schedule.annotations[i]) and visible[i]])
    height = sum([1 for i in range(schedule.n) if visible[i]]) + tot_budg + annot_height
    if tot_budg or annot_height:
        height *= 1.4
    width = (schedule.sched_end - sched_start_time)/4*scale_time
    fig:Figure = plt.figure(figsize=(width,height))
    spidx: int = 0
    axs: Dict[int, Axes] = {}
    budg_ax: Dict[int, Axes] = {}
    #annot_ax: Axes = None
    first_axs: Axes = None
    last_axs: Axes = None
    sharex: bool = draw_time and (draw_time_labels is None)
    tot_axs = schedule.n + (1 if len(schedule.annotations[-1]) > 0 else 0) + tot_budg

    has_annot: bool = force_annot_height or any([len(schedule.annotations[i]) for i in range(schedule.n)])
    has_budg_annot:bool = any([len(schedule.b_annotations[i]) and len(schedule.budgets[i]) for i in range(schedule.n)])

    #don't need to scale by speed because want different speeds of tasks to be comparable
    y_height = 1.5
    if has_annot:
        y_height = 3.0
    #need to scale by budg_maxs because dont need budget of different tasks to be comparable
    budg_height_scale = 3.0 if has_budg_annot else 1.5
    budg_heights:List[float] = [budg_height_scale*schedule.budg_maxs[i] for i in range(schedule.n)]

    for i in range(schedule.n):
        if len(schedule.budgets[i]):
            spidx += 1
            if first_axs is None:
                budg_ax[i] = fig.add_subplot(tot_axs, 1, spidx)
                first_axs = budg_ax[i]
            else:
                if sharex:
                    budg_ax[i] = fig.add_subplot(tot_axs, 1, spidx, sharex=first_axs)
                else:
                    budg_ax[i] = fig.add_subplot(tot_axs, 1, spidx)
            budg_ax[i].set_ylim([0, budg_heights[i]])
            budg_ax[i].set_yticks([0, schedule.budg_maxs[i]])
            budg_ax[i].set_xlim([sched_start_time, schedule.sched_end])
            last_axs = budg_ax[i]
        if not visible[i]:
            axs[i] = None
            continue
        spidx += 1
        #axs[i] = fig.add_subplot(str(tot_axs) + '1' + str(spidx), sharex=True)
        if first_axs is None:
            axs[i] = fig.add_subplot(tot_axs, 1, spidx)
            first_axs = axs[i]
        else:
            if sharex:
                axs[i] = fig.add_subplot(tot_axs, 1, spidx, sharex=first_axs)
            else:
                axs[i] = fig.add_subplot(tot_axs, 1, spidx)
        axs[i].set_yticks([])
        axs[i].set_ylim([0, y_height])
        axs[i].grid()
        axs[i].set_xlim([sched_start_time, schedule.sched_end])
        last_axs = axs[i]

    # create sprites for task labels
    task_labels: List[str] = []
    if task_identifiers is None:
        task_labels = ['$\\tau_' + str(i if start_from_zero else i + 1) + '$' for i in range(schedule.n)]
    else:
        task_labels = task_identifiers
    if task_padding is None:
        task_padding = [10 for i in range(schedule.n)]
    else:
        #heuristic is 4 + 3*num_chars for padding. Can't use len cuz \texttt counts toward strlen
        assert(len(task_padding) == schedule.n)
    
    # create budget labels
    budg_labels: Dict[int,str] = {}
    if budg_identifiers is None:
        for i in range(schedule.n):
            budg_labels[i] = '$q_{' + str(i if start_from_zero else i + 1) + '}$'
    else:
        for i in range(schedule.n):
            assert(bool(len(schedule.budgets[i])) is bool(i in budg_identifiers.keys()))
        budg_labels = budg_identifiers
    if budg_padding is None:
        budg_padding = {}
        for i in range(schedule.n):
            if len(schedule.budgets[i]):
                budg_padding[i] = 10
    else:
        assert(all([key in budg_padding.keys() for key in budg_labels.keys()]))

    # draw per task items
    for i in range(schedule.n):
        # draw the budget for the task (if applicable)
        if len(schedule.budgets[i]):
            budg_ax[i].set_ylabel(budg_labels[i], rotation=0, labelpad=budg_padding[i])
            for (start, end, init, final, fill) in schedule.budgets[i]:
                if fill:
                    budg_ax[i].fill([start, end, end, start], [0, 0, final, init], color='aquamarine')
                else:
                    budg_ax[i].plot([start, end], [init, final], color='aquamarine', linestyle='dotted')
            
            for (annot, start, end, astyle) in schedule.b_annotations[i]:
                annot_x = start
                if (start != end):
                    annot_x:float
                    if astyle == '->':
                        annot_x = start
                    elif astyle == '<->':
                        annot_x = (start + end)/2
                    else:
                        assert(False)
                    budg_ax[i].add_patch(FancyArrowPatch([start, (1.5 + 0.2)*schedule.budg_maxs[i]], [end, (1.5 + 0.2)*schedule.budg_maxs[i]], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale, arrowstyle=astyle))
                else:
                    budg_ax[i].plot([start, start], [0, (1.5+0.2)*schedule.budg_maxs[i]], color='k', linestyle='dotted')
                budg_ax[i].text(annot_x, (1.5 + 0.3)*schedule.budg_maxs[i], annot, ha='center', va='bottom')

        if axs[i] is None:
            continue

        # draw the task label
        axs[i].set_ylabel(task_labels[i], rotation=0, labelpad=task_padding[i])
        
        # draw each block of execution for this task
        for (proc, start_time, end_time, speed) in join_intervals(schedule.executions[i]):
            draw_rect(axs[i], start_time, end_time, speed/schedule.s_max, proc)

        for (proc, start_time, end_time, speed) in join_intervals(schedule.np_executions[i]):
            draw_rect(axs[i], start_time, end_time, speed/schedule.s_max, proc)
            axs[i].plot([start_time, end_time, end_time, start_time, start_time], [0, 0, 1, 1, 0], color='k', linewidth=3.0, linestyle='dashed')

        # draw releases/completions
        #arrwidth = 0.1
        for t in schedule.releases[i]:
            if t in schedule.pseudo_deadlines[i] or t in schedule.deadlines[i]:
                axs[i].add_patch(FancyArrowPatch([t, 0], [t, 1.2], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale))
            else:
                axs[i].add_patch(FancyArrowPatch([t, 0], [t, 1.2], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale))
        for t in schedule.pseudo_releases[i]:
            if t in schedule.pseudo_deadlines[i] or t in schedule.deadlines[i]:
                axs[i].add_patch(FancyArrowPatch([t, 0], [t, 1.2], color='gray', shrinkA=0, shrinkB=0, mutation_scale=mscale))
            else:
                axs[i].add_patch(FancyArrowPatch([t, 0], [t, 1.2], color='gray', shrinkA=0, shrinkB=0, mutation_scale=mscale))
        for t in schedule.completions[i]:
            axs[i].add_patch(FancyArrowPatch([t,0],[t, 1.2], color='k', arrowstyle='-[', shrinkA=0, shrinkB=0, mutation_scale=5))
        for t in schedule.deadlines[i]:
            if t in schedule.releases[i] or t in schedule.pseudo_releases[i]:
                axs[i].add_patch(FancyArrowPatch([t,1.2],[t, 0], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale))
            else:
                axs[i].add_patch(FancyArrowPatch([t,1.2],[t, 0], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale))
        for t in schedule.pseudo_deadlines[i]:
            if t in schedule.releases[i] or t in schedule.pseudo_releases[i]:
                axs[i].add_patch(FancyArrowPatch([t,1.2],[t, 0],color='gray', shrinkA=0, shrinkB=0, mutation_scale=mscale))
            else:
                axs[i].add_patch(FancyArrowPatch([t,1.2],[t, 0],color='gray', shrinkA=0, shrinkB=0, mutation_scale=mscale))

        if time_discont is not None and i in time_discont.keys():
            disconts = time_discont[i]
            for start, end in disconts:
                axs[i].plot([start, end], [0, 0], color='white', clip_on=False, zorder=6)
                axs[i].plot([start, end], [y_height, y_height], color='white', clip_on=False, zorder=6)

        # draw annotations
        for (annot, start, end, astyle) in schedule.annotations[i]:
            annot_x = start
            if (start != end):
                annot_x:float
                if astyle == '->':
                    annot_x = start
                elif astyle == '<->':
                    annot_x = (start + end)/2
                else:
                    assert(False)
                axs[i].add_patch(FancyArrowPatch([start, 1.5 + 0.2], [end, 1.5 + 0.2], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale, arrowstyle=astyle))
            else:
                axs[i].plot([start, start], [0, 1.5+0.2], color='k', linestyle='dotted')
            axs[i].text(annot_x, 1.5 + 0.3, annot, ha='center', va='bottom')

    for (annot, start, end, astyle) in schedule.annotations[-1]:
        annot_x:float = start
        if (start != end):
            if astyle == '->':
                annot_x = start
            elif astyle == '<->':
                annot_x = (start + end)/2
            else:
                assert(False)
            first_axs.add_patch(FancyArrowPatch([start, y_height+0.6], [end, y_height+0.6], color='k', shrinkA=0, shrinkB=0, mutation_scale=mscale, arrowstyle=astyle, clip_on=False))
        first_axs.text(annot_x, y_height + 0.8, annot, ha='center', va='bottom', clip_on=False)
        
    # label the time axis
    if draw_time:
        def_ticks = np.arange(sched_start_time, schedule.sched_end, time_axis_inc)
        if draw_time_labels is not None:
            for i in range(schedule.n):
                if axs[i] is None:
                    continue
                if i in draw_time_labels.keys():
                    ticks = [tick for tick,_ in draw_time_labels[i]]
                    labels = [label for _,label in draw_time_labels[i]]
                    axs[i].set_xticks(ticks=ticks, labels=labels)
                else:
                    axs[i].set_xticks(def_ticks)
        else:
            first_axs.set_xticks(def_ticks)
    else:
        first_axs.set_xticks([])

    if draw_time_txt:
        if draw_time_txt_label is not None:
            for i in range(schedule.n):
                if axs[i] is None:
                    continue
                #test for empty string
                if len(draw_time_txt_label[i]):
                    axs[i].set_xlabel(draw_time_txt_label[i])
        else:
            assert(last_axs is not None)
            last_axs.set_xlabel('Time')
    
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def arrow_helper(ax: plt.Axes, x:float, y:float, dx:float, dy:float, color:str=None, alpha:float = 1.0):
    if color is not None:
        return ax.quiver(x, y, dx, dy, angles = 'xy', scale = 1.0, scale_units = 'xy', alpha=alpha, color = color)
    else:
        return ax.quiver(x, y, dx, dy, angles = 'xy', scale = 1.0, scale_units = 'xy', alpha=alpha)

class AffinityGraph:
    def __init__(self, n_input:int, m_input:int, speeds:List[float]=None):
        self.n:int = n_input
        self.m:int = m_input
        self.speeds:List[float]
        if speeds is None:
            self.speeds = [1.0 for j in range(m_input)]
        else:
            self.speeds = speeds
        self.edges:List[Tuple[int,int]] = []
        self.assignments:List[Tuple[int,int]] = []
        self.np_assignments:List[Tuple[int,int]] = []
        self.links:List[Tuple[int,int,bool]] = []

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

def draw_ag(ag: AffinityGraph, filename:str = None, start_from_zero=False):
    object_height:float = 0.5

    #each object gets OH and 0.1 of padding on either side, totals OH x n + 0.1 x (n+1)
    width: float = max([ag.n, ag.m])*(object_height + 0.1) + 0.1
    #tasks/procs/edges get 0.5 each, plus 0.1 padding on each side, totals OH x 3 + 0.1 x 2
    height: float = 3*object_height + 0.2
    fig = plt.figure(figsize = (width, height))
    ax = fig.add_subplot()
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.axis('off')

    assert(max(ag.speeds) == 1.0)

    for (task_id, proc_id) in ag.edges:
        x1 = task_id*width/ag.n + 0.5*width/ag.n
        y1 = 0.1 + object_height
        x0 = proc_id*width/ag.m + 0.5*width/ag.m
        y0 = 0.1 + 2*object_height

        ax.plot([x0,x1],[y0,y1], color='k')
        if (task_id, proc_id, False) in ag.links:
            ax.quiver([x0, y0], [x1-x0, y1-y0], angles='xy', scale=1.0, scale_units='xy', color='k')
        if (task_id, proc_id, True) in ag.links:
            ax.quiver([x1, y1], [x0-x1, y0-y1], angles='xy', scale=1.0, scale_units='xy', color='k')
        if (task_id, proc_id) in ag.np_assignments:
            ax.plot([x0,x1],[y0,y1], color='k', linewidth=5.0, linestyle='dashed')
        elif (task_id, proc_id) in ag.assignments:
            ax.plot([x0,x1],[y0,y1], color='k', linewidth=5.0, alpha=0.3)

    for j in range(ag.m):
        this_radius:float = object_height/2 * np.sqrt(ag.speeds[j])
        center:Tuple[float,float] = (j*width/ag.m + 0.5*width/ag.m, 0.1 + object_height*2.5)
        draw_circ(ax, center, this_radius, j)
        ax.add_patch(Circle(center, 0.6*this_radius, facecolor='white'))
        ax.text(center[0], center[1], '$\\pi_' + str(j if start_from_zero else j + 1) + '$', ha = 'center', va = 'center')

    for i in range(ag.n):
        #center of box is at i*width/ag.n + 0.5*width/ag.n
        #left of box is then that minus object_height/2, right is plus object_height/2
        center:Tuple[float,float] = (i*width/ag.n + 0.5*width/ag.n, 0.1 + object_height/2)
        cx:float = center[0]
        cy:float = center[1]
        ax.fill([cx - object_height/2, cx + object_height/2, cx + object_height/2, cx - object_height/2],\
                [cy - object_height/2, cy - object_height/2, cy + object_height/2, cy + object_height/2],\
                    facecolor='white', edgecolor='k')
        ax.text(cx, cy, '$\\tau_' + str(i if start_from_zero else i + 1) + '$', ha='center', va='center')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')


"""
Drawables maintain dimensions.
"""

class Drawable:
    def __init__(self, width_input: float, height_input: float, facecolor='white', edgecolor='k', hatch=None, zorder:int=-1):
        self.height = height_input
        self.width = width_input
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.hatch = hatch
        self.zorder = zorder
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        assert(False)

class TextGraphic(Drawable):
    def __init__(self, text: str, color='k'):
        scratch_fig: Figure = plt.figure()
        scratch_ax: Axes = scratch_fig.add_subplot()
        scratch_txt: Text = scratch_ax.text(0, 0, text)
        scratch_fig.draw_without_rendering()
        bbox: Bbox = scratch_txt.get_window_extent()

        TEXT_SCALE_MAGIC = 18

        super().__init__(bbox.bounds[2]/TEXT_SCALE_MAGIC, bbox.bounds[3]/TEXT_SCALE_MAGIC, facecolor=color, edgecolor=color, hatch=None)
        self.text = text

        #disp_width: float = bbox.bounds[2]/TEXT_SCALE_MAGIC
        #disp_height: float = bbox.bounds[3]/TEXT_SCALE_MAGIC
        #print("width: {width},\theight: {height}\n".format(width=disp_width, height=disp_height))
        plt.close(scratch_fig)
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        ax.text(x, y, self.text, color=self.edgecolor)

class RectGraphic(Drawable):
    def __init__(self, width: float, height:float, facecolor='white', edgecolor='k', hatch=None, zorder:int=-1):
        super().__init__(width, height, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, zorder=zorder)
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        if self.hatch is None:
            if self.zorder == -1:
                ax.fill([x, x + self.width, x + self.width, x], [y, y, y + self.height, y + self.height], facecolor=self.facecolor, edgecolor=self.edgecolor)
            else:
                ax.fill([x, x + self.width, x + self.width, x], [y, y, y + self.height, y + self.height], facecolor=self.facecolor, edgecolor=self.edgecolor, zorder=self.zorder)
        else:
            if self.zorder == -1:
                ax.fill([x, x + self.width, x + self.width, x], [y, y, y + self.height, y + self.height], facecolor=self.facecolor, edgecolor=self.edgecolor, hatch=self.hatch)
            else:
                ax.fill([x, x + self.width, x + self.width, x], [y, y, y + self.height, y + self.height], facecolor=self.facecolor, edgecolor=self.edgecolor, hatch=self.hatch, zorder=self.zorder)

class CircGraphic(Drawable):
    def __init__(self, radius:float, facecolor='white', edgecolor='k', hatch=None):
        super().__init__(2*radius, 2*radius, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch)
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        if self.hatch is None:
            ax.add_patch(Circle((x + self.width/2,y + self.height/2), self.width/2, edgecolor=self.edgecolor, facecolor=self.facecolor, zorder=3))
        else:
            ax.add_patch(Circle((x + self.width/2,y + self.height/2), self.width/2, edgecolor=self.edgecolor, facecolor=self.facecolor, hatch=self.hatch, zorder=3))

class PathGraphic(Drawable):
    def __init__(self, path:List[Tuple[float,float]], edgecolor='k', beg_arrow=False, end_arrow=False, close=False):
        assert(len(path) > 1)

        min_x, max_x, min_y, max_y = 0, 0, 0, 0
        for x, y in path:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        self.beg_arrow = beg_arrow
        self.end_arrow = end_arrow
        self.path = path
        self.close = close
        
        super().__init__(max_x - min_x, max_y - min_y, edgecolor=edgecolor)
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        codes = [Path.LINETO for xy in self.path]
        codes[0] = Path.MOVETO
        path = self.path.copy()
        if self.close:
            path.append(path[0])
            codes.append(Path.CLOSEPOLY)
        for i in range(len(path)):
            path[i] = (x + path[i][0], y + path[i][1])
        pltpath = Path(path, codes)
        if self.beg_arrow and self.end_arrow:
            ax.add_patch(FancyArrowPatch(path=pltpath, arrowstyle='<->', color=self.edgecolor))
        elif self.beg_arrow:
            ax.add_patch(FancyArrowPatch(path=pltpath, arrowstyle='<-', color=self.edgecolor))
        elif self.end_arrow:
            ax.add_patch(FancyArrowPatch(path=pltpath, arrowstyle='->', color=self.edgecolor, mutation_scale=7))
        else:
            ax.add_patch(FancyArrowPatch(path=pltpath, arrowstyle='-', color=self.edgecolor))

class CompositeGraphic(Drawable):
    def __init__(self):
        super().__init__(0, 0)

        # may contain repeat instances
        self.elements: List[Tuple[Drawable, float, float]] = []
    
    def draw(self, ax: Axes, x:float, y:float) -> None:
        for drawable, x2, y2 in self.elements:
            drawable.draw(ax, x + x2, y + y2)

    def _add_element(self, draw_src: Drawable, x: float, y: float):

        self.elements.append((draw_src, x, y))

        self.width = max([x + draw_src.width, self.width])
        self.height = max([y + draw_src.height, self.height])

        return True

    def remove_element(self, index:int):
        assert (0 <= index and index < len(self.elements))

        # get the drawable to be removed
        el_drawable = self.elements[0][index]

        # remove the drawable from elements
        last = len(self.elements)
        self.elements = self.elements[0:index] + self.elements[index + 1:last]

        # recompute bounding box
        self.width = 0
        self.height = 0
        for (draw, x, y) in self.elements:
            self.width = max([x + draw.width, self.width])
            self.height = max([y + draw.height, self.height])

    def draw_center(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y - draw_src.height / 2)

    def draw_left_middle(self, draw_src, x, y):
        self._add_element(draw_src, x, y - draw_src.height / 2)

    def draw_right_middle(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width, y - draw_src.height / 2)

    def draw_left_top(self, draw_src, x=0, y=0):
        self._add_element(draw_src, x, y - draw_src.height/2)

    def draw_right_bottom(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width, y)

    def draw_left_bottom(self, draw_src, x=0, y=0):
        self._add_element(draw_src, x, y)

    def draw_center_top(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y - draw_src.height/2)

    def draw_center_bottom(self, draw_src, x, y):
        self._add_element(draw_src, x - draw_src.width / 2, y)


def textbox(text: str, width:float = 0, height:float = 0, edgecolor:str = 'k', txtcolor:str = 'k') -> CompositeGraphic:
    txtGrp = TextGraphic(text, color=txtcolor)
    rect = RectGraphic(max([width, 0.5 + txtGrp.width]), max([height, 0.5 + txtGrp.height]), edgecolor=edgecolor)
    #print("rect width:{rwidth},\trect height:{rheight}".format(rwidth=rect.width, rheight=rect.height))

    ret = CompositeGraphic()
    ret.draw_left_bottom(rect, 0, 0)
    ret.draw_center(txtGrp, rect.width/2, rect.height/2)
    return ret

def cpu(cpunum:int, min_radius:float = 0) -> CompositeGraphic:
    ret = CompositeGraphic()
    assert( 0 <= cpunum <= 7)

    txt = TextGraphic(r'\texttt{cpu~' + str(cpunum) + r'}')
    in_radius = txt.width/2*1.2
    radius = max([min_radius, in_radius/0.7])

    if cpunum == 0:
        ret.draw_left_bottom(CircGraphic(radius, facecolor='red'), 0, 0)
    else:
        facecolor:str
        hatch:str
        if cpunum == 1:
            facecolor='blue'
            hatch='//'
        elif cpunum == 2:
            facecolor='orange'
            hatch='--'
        elif cpunum == 3:
            facecolor='olive'
            hatch=r'\\'
        elif cpunum == 4:
            facecolor='mediumorchid'
            hatch='||'
        elif cpunum == 5:
            facecolor='gray'
            hatch='++'
        elif cpunum == 6:
            facecolor='gold'
            hatch='xx'
        elif cpunum == 7:
            facecolor='tomato'
            hatch='..'
        ret.draw_left_bottom(CircGraphic(radius, facecolor=facecolor, hatch=hatch), 0, 0)
    
    ret.draw_center(CircGraphic(in_radius), ret.width/2, ret.height/2)
    ret.draw_center(txt, ret.width/2, ret.height/2)
    return ret


#Consider reworking this all to work as properties
class LiTask:
    cid:int = 0
    draw_deadline:bool = True
    draw_runtime:bool = False
    draw_cpusptr:bool = False
    draw_cpusmask:bool = False
    draw_static:bool = False
    draw_state:bool = False
    draw_policy:bool = False
    draw_onrq:bool = False
    def __init__(self, 
                 pid:int=-1, 
                 cpu:int=-1, 
                 onrq:int=1, 
                 deadline:int=0, 
                 dldeadline:int=0, 
                 dlperiod:int=0, 
                 dlruntime:int=0, 
                 runtime:int=0, 
                 cpusptr:List[int]=[], 
                 cpusmask:List[int]=[],
                 state:str='TASK_RUNNING',
                 policy:str=r'SCHED\_DEADLINE',
                 force_queued:bool=False,
                 force_unqueued:bool=False) -> None:
        if pid == -1:
            self.pid = LiTask.cid
            LiTask.cid += 1
        else:
            self.pid = pid
        self.deadline=deadline
        self.cpusptr=cpusptr
        self.cpusmask=cpusmask
        self.dlperiod=dlperiod
        self.dldeadline=dldeadline
        self.dlruntime=dlruntime
        self.runtime=runtime
        self.cpu = cpu
        self.onrq = onrq
        self.state = state
        self.policy = policy

        self.force_queued = force_queued
        self.force_unqueued = force_unqueued

        self.is_ghost:bool = False
        self.emphList:List[str] = []
        self._emph_task:bool = False
    
    def ghost(self) -> LiTask:
        cpy:LiTask = copy.copy(self)
        cpy.is_ghost = True
        return cpy

    def emph(self, attr:str) -> None:
        assert(hasattr(self, attr))
        assert(not (attr in self.emphList))
        self.emphList.append(attr)
    
    def deemph(self, attr:str) -> None:
        assert(attr in self.emphList)
        self.emphList.remove(attr)
    
    def emph_task(self) -> None:
        self._emph_task = not self._emph_task

#For sorting
def task_deadline(task:LiTask) -> int:
    if task.policy == r'SCHED\_DEADLINE':
        return task.deadline
    return float('inf')

def on_rq_str(task:LiTask) -> bool:
    if task.onrq == 0:
        return '0'
    elif task.onrq == 1:
        return 'TASK_ON_RQ_QUEUED'
    elif task.onrq == 2:
        return 'TASK_ON_RQ_MIGRATING'
    else:
        assert(False)
    

def draw_task(task:LiTask, task_width:float=0) -> Drawable:
    paramList:List[TextGraphic] = []

    edge_col:str = 'k'
    if task.is_ghost:
        edge_col = 'gray'
    emph_col:str = 'red'

    #Too lazy to make this not shit
    paramList.append(TextGraphic(r'\texttt{pid:~' + str(task.pid) + '}', color=edge_col))
    if LiTask.draw_state:
        paramList.append(TextGraphic(r'\texttt{__state:~' + task.state + '}', color=emph_col if 'state' in task.emphList else edge_col))
    if LiTask.draw_onrq:
        paramList.append(TextGraphic(r'\texttt{on\_rq:~' + on_rq_str(task) + '}', color=emph_col if 'onrq' in task.emphList else edge_col))
    if LiTask.draw_deadline:
        paramList.append(TextGraphic(r'\texttt{dl.deadline:~' + str(task.deadline) + '}', color=emph_col if 'deadline' in task.emphList else edge_col))
    if LiTask.draw_runtime:
        paramList.append(TextGraphic(r'\texttt{dl.runtime:~' + str(task.runtime) + '}', color=emph_col if 'runtime' in task.emphList else edge_col))
    if LiTask.draw_policy:
        paramList.append(TextGraphic(r'\texttt{policy:~' + task.policy + '}', color=emph_col if 'policy' in task.emphList else edge_col))
    if LiTask.draw_cpusptr:
        paramList.append(TextGraphic(r'\texttt{cpus_ptr:~' + str(task.cpusptr) + '}', color=emph_col if 'cpusptr' in task.emphList else edge_col))

    paramHeight = sum([txt.height + 0.5 for txt in paramList])
    out = CompositeGraphic()
    tsG = textbox(r'\texttt{task\_struct}', height=paramHeight, edgecolor=edge_col, txtcolor=edge_col)
    paramWidth = max([txt.width for txt in paramList] + [task_width - tsG.width - 0.5])
    out.draw_left_bottom(tsG)
    y_off = 0
    paramList.reverse()
    for txt in paramList:
        out.draw_left_bottom(RectGraphic(paramWidth + 0.5, txt.height + 0.5, edgecolor=edge_col), tsG.width, y_off)
        out.draw_left_bottom(txt, tsG.width + 0.25, y_off + 0.25)
        y_off += txt.height + 0.5
    if task._emph_task:
        out.draw_left_bottom(RectGraphic(out.width, out.height, edgecolor=emph_col, facecolor='none', zorder=6))
    return out

def unqueued(tasks:List[LiTask]) -> Drawable:
    assert(len(tasks))
    assert(all([task.force_unqueued or not (task.onrq and task.runtime) for task in tasks]))

    out = CompositeGraphic()
    
    tasksG:List[Drawable] = [draw_task(task) for task in tasks]
    tasks_width = max([task.width for task in tasksG])
    tasksG = [draw_task(task, tasks_width) for task in tasks]
    y_off = 0
    for task in tasksG:
        out.draw_left_bottom(task, 0, y_off)
        y_off += task.height + 0.2
    
    return out


def rq(cpunum:int, tasks:List[LiTask], stop:bool=False, emph_stop:bool=False) -> Drawable:
    out = CompositeGraphic()
    cpuG = cpu(cpunum)
    out.draw_left_bottom(cpuG, 0, 0)
    y_off = 0
    rq = CompositeGraphic()
    tasks_width = 0
    if len(tasks) or stop:
        if len(tasks):
            tasks.sort(key=task_deadline)
            tasksG:List[Drawable] = [draw_task(task) for task in tasks]
            tasks_width = max([task.width for task in tasksG])
            tasksG = [draw_task(task, tasks_width) for task in tasks]
        if stop:
            stoptb = textbox(r'\texttt{cpu_rq(' + str(cpunum) + ')->stop}', tasks_width)
            rq.draw_left_bottom(stoptb)
            if emph_stop:
                rq.draw_left_bottom(RectGraphic(stoptb.width, stoptb.height, edgecolor='red', facecolor='none', zorder=6))
            y_off += stoptb.height
        if len(tasks):
            for task in tasksG:
                rq.draw_left_bottom(task, 0, y_off)
                y_off += task.height
    else:
        idletb = textbox(r'\texttt{cpu_rq(' + str(cpunum) + ')->idle}')
        rq.draw_left_bottom(idletb)

    if cpuG.width + 0.5 < rq.width:
        rq_x_off = 0
        path_x = (cpuG.width + rq.width)/2 - cpuG.width
    else:
        rq_x_off = cpuG.width + 0.5 - rq.width
        path_x = 0.25

    y_off = cpuG.height + 0.5
    out.draw_left_bottom(rq, rq_x_off, y_off)
    pg = PathGraphic([(path_x, cpuG.height/2 + 0.5), (path_x, 0), (0, 0)], end_arrow=True)
    out.draw_left_bottom(pg, cpuG.width, cpuG.height/2)
    return out

def rqs(tasks:List[LiTask], stops:List[Tuple[int,bool]]=[], vert:bool=True, draw_unqueued:bool=True) -> Drawable:
    assert(len(tasks))
    cpus = 1 + max([task.cpu for task in tasks])
    assert(0 < cpus < 8)
    assert(all([task.cpu >= 0 for task in tasks]))
    assert(all([not (task.force_queued and task.force_unqueued) for task in tasks]))
    stops_idxs = [stop for stop,_ in stops]
    stops_emphs = [emph for _,emph in stops]
    if len(stops):
        assert(max(stops_idxs) < cpus and min(stops_idxs) >= 0)

    out = CompositeGraphic()

    unq:List[LiTask] = []
    if draw_unqueued:
        unq = [task for task in tasks if task.force_unqueued or not (task.onrq and task.runtime)]
        unq = [task for task in unq if not task.force_queued]

    if len(unq):
        unqG = unqueued(unq)

    queues = [
        rq(cpu, 
           [task for task in tasks if task.cpu == cpu and (task.force_queued or (task.onrq and task.runtime))], 
           cpu in stops_idxs, 
           (cpu in stops_idxs) and stops_emphs[stops_idxs.index(cpu)]
           ) for cpu in range(cpus)
        ]

    if vert:
        queues.reverse()
        y_off = 0
        for queue in queues:
            out.draw_left_bottom(queue, 0, y_off)
            y_off += (queue.height + 0.25)
        if len(unq):
            unq_y_mid = max([unqG.height, out.height])/2
            out.draw_left_middle(unqG, out.width + 0.5, unq_y_mid)
    else:
        x_off = 0
        for queue in queues:
            out.draw_left_bottom(queue, x_off, 0)
            x_off += queue.width + 0.25
        if len(unq):
            #assume height of unqG is less than queues
            unq_y_mid = max([unqG.height, out.height])/2
            out.draw_left_middle(unqG, x_off + 0.25, unq_y_mid)
        
    return out

def __heap(elements:List[Drawable], idx:int) -> Drawable:
    if 2*idx + 2 < len(elements):
        lchild = __heap(elements, 2*idx + 1)
        rchild = __heap(elements, 2*idx + 2)
        ret = CompositeGraphic()
        mheight = max([lchild.height, rchild.height])
        ret.draw_left_bottom(lchild, 0, mheight - lchild.height)
        rchild_x = max([lchild.width + 2, elements[idx].width - rchild.width])
        ret.draw_left_bottom(rchild, rchild_x, mheight - rchild.height)
        par_cent_x = (rchild_x + rchild.width)/2
        ret.draw_center_bottom(elements[idx], par_cent_x, mheight + 2)
        pg = PathGraphic([(lchild.width/2, mheight), (par_cent_x, mheight + 2), (rchild_x + rchild.width/2, mheight)])
        ret.draw_left_bottom(pg, 0, 0)
        return ret
    if 2*idx + 1 < len(elements):
        child = __heap(elements, 2*idx + 1)
        ret = CompositeGraphic()
        cent_x = max([elements[idx].width, child.width])/2
        ret.draw_center_bottom(child, cent_x, 0)
        ret.draw_center_bottom(elements[idx], cent_x, child.height + 2)
        pg = PathGraphic([(cent_x, child.height), (cent_x, child.height + 2)])
        ret.draw_left_bottom(pg, 0, 0)
        return ret
    return elements[idx]

def heap(elements:List[Drawable]) -> Drawable:
    assert(len(elements) > 0)
    return __heap(elements, 0)

class Field(TextGraphic):
    def __init__(self, name:str, points_to_arg: List[Field] = None, box:bool = True) -> None:
        super().__init__("\\texttt{" + name + "}")
        if points_to_arg is None:
            self.points_to: List[Field] = []
        else:
            self.points_to = points_to_arg
        self._text_width = self.width
        self._text_height = self.height
        self.width += 1
        self.height += 0.5
        self.box = box
        self._x: float
        self._y: float
        self._ax: Axes = None
    
    def draw(self, ax: Axes, x: float, y: float) -> None:
        self._ax = ax
        if self.box:
            rect = RectGraphic(self.width, self.height)
            rect.draw(ax, x, y)
        super().draw(ax, x + 0.5, y + self.height - self._text_height - 0.25)
        self._x = x
        self._y = y
        draw_callbacks.append(self.draw_pointers)
    
    def add_pointer(self, new_pointer:Field) -> None:
        self.points_to.append(new_pointer)
    
    def draw_pointers(self) -> None:
        assert(self._ax is not None)
        for pt in self.points_to:
            pg = PathGraphic([(self._x + self.width, self._y + self.height/2), (pt._x, pt._y + pt.height/2)], end_arrow=True)
            pg.draw(self._ax, 0, 0)
    
    def get_field(self, name:str) -> Field:
        assert(False)

    def _update_width(self, new_width:float) -> None:
        assert(new_width >= self.width)
        self.width = new_width

draw_callbacks:List[Callable[[],None]] = []

class Struct(Field):
    def __init__(self, name: str, points_to: List[Field] = [], fields: Dict[str, Field] = {}) -> None:
        super().__init__(name, points_to)
        self.fields = fields
        inner_width = max([self.width] + [field.width for field in fields.values()])
        self.width =  inner_width + 1
        for field in fields.values():
            field._update_width(inner_width)
        self.height += sum([field.height+0.25 for field in fields.values()]) + 0.5
    
    def draw(self, ax: Axes, x: float, y: float) -> None:
        super().draw(ax, x, y)
        y_offset = self._text_height + 0.25
        for field in self.fields.values():
            y_offset += field.height+0.25
            field.draw(ax, x + 0.5, y + self.height - y_offset)
    
    def get_field(self, name: str) -> Field:
        return self.fields[name]

def preview(drawable: Drawable, figsize:Tuple[float,float]=None) -> None:
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(drawable.width/SCALEDOWN_DISPLAY_DATA, drawable.height/SCALEDOWN_DISPLAY_DATA))
    ax = fig.add_subplot()
    ax.set_xlim([-0.1, drawable.width+0.1])
    ax.set_ylim([-0.1, drawable.height+0.1])
    ax.axis('off')
    drawable.draw(ax, 0, 0)
    for callback in draw_callbacks:
        callback()
    plt.show()

def publish(drawable: Drawable, filename: str, figsize:Tuple[float,float]=None) -> None:
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(drawable.width/SCALEDOWN_DISPLAY_DATA, drawable.height/SCALEDOWN_DISPLAY_DATA))
    ax = fig.add_subplot()
    ax.set_xlim([-0.1, drawable.width+0.1])
    ax.set_ylim([-0.1, drawable.height+0.1])
    ax.axis('off')
    drawable.draw(ax, 0, 0)
    for callback in draw_callbacks:
        callback()
    plt.savefig(filename, bbox_inches='tight')