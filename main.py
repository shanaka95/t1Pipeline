import cropper

videos = [
    "videos/clip6_10min.mp4",
    "videos/clip5_10min.mp4", 
    "videos/clip4_10min.mp4",
    "videos/clip3_10min.mp4",
    "videos/004_t1_20230217_clip_10min.mp4",
    "videos/output_clip.mp4",
    "videos/output_10min4.mp4",
    "videos/output_10min3.mp4",
    "videos/output_10min2.mp4",
    "videos/output_10min.mp4"
]

for video in videos:
    print(cropper.get_width(video))