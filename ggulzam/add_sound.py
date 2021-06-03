from moviepy.editor import *
videoclip = VideoFileClip("c:/video/coin_sketch_compress.mp4")
audioclip = AudioFileClip("c:/video/coin_sound.mp3")
new_audioclip = CompositeAudioClip([audioclip])
videoclip.audio = new_audioclip
videoclip.write_videofile("c:/video/coin_with_sound.mp4")