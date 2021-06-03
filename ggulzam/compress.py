import moviepy.editor as mp
clip = mp.VideoFileClip("C:/video/coin_sketch.avi")
clip_resized = clip.resize(height=720) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile("C:/video/coin_sketch_compress.mp4")