from typing import List
import fig_maker as fm 
import numpy as np
import ffmpeg
import cairo

class vidData:
    def __init__(self) -> None:
        self.data: List[fm.CompositeGraphic] = []
    def frame(self, frame: fm.CompositeGraphic) -> None:
        self.data.append(frame)
    def publish(self, outfile='movie.mp4', framerate=25, width=640, height=480) -> None:
        print('{width}x{height}'.format(height=height,width=width))
        
        process = (
            #vcodec='libx264'
            #pix_fmt='yuv420p'
            ffmpeg
                .input('pipe:', format='rawvideo', framerate=framerate, pix_fmt='rgb24', s=('{width}x{height}'.format(height=height,width=width)))
                .output(outfile, vcodec='libx264',pix_fmt='yuv420p')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        
        #idx = 0
    
        for frame in self.data:
            frame.update_published_surface()
            imgsurf = cairo.ImageSurface(cairo.Format.ARGB32, width, height)
            ctx = cairo.Context(imgsurf)
            ctx.set_source_surface(frame.pub_surface, 0, 0)
            ctx.paint()

            buff = imgsurf.get_data()
            array = np.ndarray(shape=(height,width,4), dtype=np.uint8, buffer=buff)
            #print(array)
            process.stdin.write(
                array[:,:,(2,1,0)].tobytes()
            )
            
            #print('Finished frame {idx}'.format(idx=idx))
            #idx += 1

        process.stdin.close()
        process.wait()
