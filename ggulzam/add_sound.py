from moviepy.editor import *
videoclip = VideoFileClip("c:/video/eight_acoustic_sketch_compress.mp4")
audioclip = AudioFileClip("c:/video/eight_acoustic_sound.mp3")
new_audioclip = CompositeAudioClip([audioclip])
videoclip.audio = new_audioclip
videoclip.write_videofile("c:/video/eight_acoustic_with_sound.mp4")