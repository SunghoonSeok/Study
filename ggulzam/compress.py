import moviepy.editor as mps
clip = mp.VideoFileClip("C:/video/celebrity_sketch.avi")
clip_resized = clip.resize(height=720) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile("C:/video/celebrity_sketch_compress.mp4")