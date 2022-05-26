from typing import List
import fig_maker as fm 
import numpy as np
import ffmpeg
import cairo

class vidData:
    def __init__(self, outfile='movie.mp4', framerate=25, width=1920, height=1080, bg=None) -> None:
        self.framerate = framerate
        self.width = width
        self.height = height
        self.bg = bg
        self.process = (
            ffmpeg
                .input('pipe:', format='rawvideo', framerate=framerate, pix_fmt='rgb24', s=('{width}x{height}'.format(height=height,width=width)))
                .output(outfile, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    
    def time_to_frames(self, sec: float) -> int:
        return int(sec*self.framerate)

    def frame(self, frame: fm.Drawable, num: int = 1) -> None:
        frame.update_published_surface()
        imgsurf = cairo.ImageSurface(cairo.Format.ARGB32, self.width, self.height)
        ctx = cairo.Context(imgsurf)
        if self.bg is not None:
            ctx.set_source(self.bg)
            ctx.move_to(0,0)
            ctx.line_to(0, self.height)
            ctx.line_to(self.width,self.height)
            ctx.line_to(self.width,0)
            ctx.close_path()
            ctx.fill()
        #ctx.save()
        #ctx.scale(fm.preview_scale,fm.preview_scale)
        frame.draw_published(ctx, self.width/2 - frame.width/2, self.height/2 - frame.height/2)
        #ctx.restore() 

        buff = imgsurf.get_data()
        array = np.ndarray(shape=(self.height, self.width, 4), dtype=np.uint8, buffer=buff)

        for i in range(num):
            self.process.stdin.write(
                array[:,:,(2,1,0)].tobytes()
            )
    

    def publish(self) -> None:

        self.process.stdin.close()
        self.process.wait()
