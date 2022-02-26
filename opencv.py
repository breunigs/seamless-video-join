#!/usr/bin/env python3

from alive_progress import alive_bar
from cmath import sqrt
from typing import Tuple, List, Optional
import copyreg
import cv2
import datetime
import multiprocessing
import os
import queue
import statistics
import sys
import threading


print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")

MAX_RESULTS = 5
SEARCH_DEPTH = 50
LOWE_RATIO_TEST = 0.6
MIN_KEYPOINT_MATCH = 60
IMG_RESIZE_WIDTH_PASS1 = 360
IMG_RESIZE_WIDTH_PASS2 = 1280

debug = False

# implementation for various algos, good reference material:
# https://github.com/whoisraibolt/Feature-Detection-and-Matching


Candidate = Tuple[float, float, float]
Candidates = List[Candidate]


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


def parse_ffmpeg_time(input):
    [h, m, rest] = input.split(":", maxsplit=3)
    [s, ms] = rest.split(".", maxsplit=2)
    return int(h)*3600+int(m)*60+int(s)+int(ms)/1000.0


def float_to_ffmpeg_time(input):
    h = int(input/3600)
    m = int((input-h*3600)/60)
    s = int(input-h*3600-m*60)
    ms = round((input-h*3600-m*60-s)*1000)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(h, m, s, ms)


def clean_filename(input):
    split = input.split("/")
    return f'{split[-2]}/{split[-1].removesuffix(".MP4")}'


def die(msg):
    print(msg)
    exit(1)


def open_video(video_path, timestamp, search_duration):
    video_path = os.path.expanduser(video_path)
    print(f'Reading video {video_path}')
    if not os.path.exists(video_path):
        die("video path doesn't exist")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        die("failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        die("cannot determine FPS")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

    frame_pos = int(round(fps*timestamp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    frame_max = frame_pos+int(round(fps*search_duration.total_seconds()))
    frame_max = min(frame_max, frame_count)
    print(f'{round(fps, 3)} fps; from frame {frame_pos} to {frame_max} ({frame_max-frame_pos} frames)')

    return (cap, frame_pos, frame_max, fps), frame_max-frame_pos


def read_frames(video, img_size, filter_timestamps, crop_ratio=0.0):
    (cap, frame_pos, frame_max, fps) = video

    brisk = cv2.BRISK_create()
    while(cap.isOpened() and frame_pos < frame_max):
        # read frame, working around incorrectly read frames
        ok, frame = cap.read()
        if not ok:
            fake_frame_skip = 1000
            while not ok and fake_frame_skip > 0:
                fake_frame_skip -= 1
                ok, frame = cap.read()
            if not ok:
                die(
                    f"WARN: video ended prematurely at frame={frame_pos}? See this issue for details: https://github.com/ultralytics/yolov5/issues/2064")

        timestamp = frame_pos/fps
        frame_pos += 1
        if len(filter_timestamps) > 0 and not timestamp in filter_timestamps:
            continue

            # grayscale, as we don't need color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scale down image for speed
        _, orig_w = frame.shape
        ratio = img_size / orig_w
        if ratio < 1.0:
            frame = cv2.resize(frame, None, fx=ratio, fy=ratio,
                               interpolation=cv2.INTER_CUBIC)

        offset = (0, 0)
        # remove border around image
        if crop_ratio > 0.0:
            orig_h, orig_w = frame.shape
            crop_h = round(orig_h*crop_ratio/2.0)
            crop_w = round(orig_w*crop_ratio/2.0)
            frame = frame[crop_h:orig_h-2*crop_h, crop_w:orig_w-2*crop_w]
            offset = (crop_h, crop_w)

        kp, des = brisk.detectAndCompute(frame, None)

        if not debug:
            frame = None
        yield(timestamp, kp, des, offset, frame)


def flann_matcher():
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)

    search_params = dict(checks=SEARCH_DEPTH)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann


def dist(kp1, offset1, kp2, offset2, elem):
    pt1 = kp1[elem.queryIdx].pt
    pt1_h = pt1[0] + offset1[0]
    pt1_w = pt1[1] + offset1[1]

    pt2 = kp2[elem.trainIdx].pt
    pt2_h = pt2[0] + offset2[0]
    pt2_w = pt2[1] + offset2[1]

    return sqrt(pow(pt1_h-pt2_h, 2)+pow(pt1_w-pt2_w, 2)).real


def score(frame1, start1, frame2, start2, img_size) -> Candidate:
    (ts1, kp1, des1, offset1, img1) = frame1
    (ts2, kp2, des2, offset2, img2) = frame2

    ts1_delta = ts1 - start1
    ts2_delta = ts2 - start2

    matches = flann_matcher().knnMatch(des1, des2, k=2)
    if len(matches) == 0:
        return (float('inf'), ts1, ts2)

    decent = []
    dists = []
    for elem in matches:
        # ORB+FLANN sometimes returns an empty match?
        if len(elem) == 0:
            continue

        # if there are two matches, ensure they are far enough apart
        if len(elem) == 2 and elem[0].distance > LOWE_RATIO_TEST * elem[1].distance:
            continue

        decent.append(elem[0])
        dists.append(dist(kp1, offset1, kp2, offset2, elem[0]))

    if len(decent) == 0:
        return (float('inf'), ts1, ts2)

    # minimize median distance between good keypoints, but penalize low match count
    score_dist = statistics.median(dists)
    score_count = MIN_KEYPOINT_MATCH/min(MIN_KEYPOINT_MATCH, len(decent))
    # give slight bonus when more frames are included. The deltas are in seconds.
    score_frames = - 5*ts1_delta + 5*ts2_delta
    score = score_dist * score_count + score_frames

    if debug:
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches1to2=decent,
                               outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        name = "___".join([f"{img_size}", float_to_ffmpeg_time(ts1), float_to_ffmpeg_time(ts2),
                           f"decent_{len(decent)}", f"dist_{score_dist}", f"count_{score_count}",
                           f"frames_{score_frames}", f"score_{score}"])
        cv2.imwrite(f"debug/{name}.jpg", img3)

    return (score, ts1, ts2)


def with_cache(gen, cache):
    if len(cache) > 0:
        for x in cache:
            yield x

        return

    for x in gen:
        cache.append(x)
        yield x


def detect(title, video1, start1, video2, start2, search, img_size, filter: Optional[Candidates] = None) -> Candidates:
    vid1, count1 = open_video(video1, start1, search)
    vid2, count2 = open_video(video2, start2, search)
    count = count1*count2

    filter1 = set()
    filter2 = set()
    filter3 = set()
    if filter:
        for _, ts1, ts2 in filter:
            filter1.add(ts1)
            filter2.add(ts2)
            filter3.add((ts1, ts2))
        count = len(filter)

    q = queue.Queue(maxsize=multiprocessing.cpu_count())
    candidates = []

    def collect():
        with alive_bar(count, spinner=None, title=title) as bar:
            while len(candidates) < count:
                candidates.append(q.get().get())
                bar()
                q.task_done()

    p = threading.Thread(target=collect)
    p.start()

    # we crop the frames of the first videos because we know we will "zoom in" most
    # likely, so the borders of the first video will never appear in the subsequent
    # video. This is a use-case specific optimization.
    frame_gen1 = read_frames(vid1, img_size, filter1, crop_ratio=0.3)
    frame_gen2 = read_frames(vid2, img_size, filter2)

    frame_cache2 = []
    pool = multiprocessing.Pool()
    for frame1 in frame_gen1:
        for frame2 in with_cache(frame_gen2, frame_cache2):
            ts1, ts2 = frame1[0], frame2[0]
            if filter and not (ts1, ts2) in filter3:
                continue

            f = pool.apply_async(
                score, (frame1, start1, frame2, start2, img_size))
            q.put(f)

    p.join()
    while not q.empty():
        candidates.append(q.get().get())

    candidates.sort(key=lambda y: y[0], reverse=False)
    return candidates


def iround(val):
    if val == float('inf'):
        return val
    return round(val, 2)


def keep_promising(candidates: Candidates) -> Candidates:
    return [x for x in candidates if x[0] <= 15*candidates[0][0] and x[0] <= 400]


def enable_debug():
    os.makedirs('debug/', exist_ok=True)
    global debug
    debug = True


# TODO: parse as args
video1 = "~/test/veloroute/videos/source/2021-08-04-fr13-fr14/GX012546.MP4"
start1 = parse_ffmpeg_time("00:00:09.000")
video2 = "~/test/veloroute/videos/source/2021-08-04-fr13-fr14/GX012546.MP4"
start2 = parse_ffmpeg_time("00:00:23.000")
# video1 = "~/test/veloroute/videos/source/2022-01-16-wedel/GX013142.MP4"
# start1 = parse_ffmpeg_time("00:00:10.000")
# video2 = "~/test/veloroute/videos/source/2022-01-16-wedel/GX013143.MP4"
# start2 = parse_ffmpeg_time("00:00:00.000")
search = datetime.timedelta(seconds=1)

candidates = detect("low resolution search", video1, start1, video2, start2,
                    search, IMG_RESIZE_WIDTH_PASS1)
candidates = keep_promising(candidates)
enable_debug()
candidates = detect("high resolution", video1, start1, video2, start2,
                    search, IMG_RESIZE_WIDTH_PASS2, filter=candidates)


for (score, ts1, ts2) in candidates[0: MAX_RESULTS]:
    ts1 = float_to_ffmpeg_time(ts1)
    ts2 = float_to_ffmpeg_time(ts2)
    print(f'''
{iround(score)} for {ts1} → {ts2}

    vv --start={ts1} --really-quiet {clean_filename(video1)} &
    vv --start={ts2} --really-quiet {clean_filename(video2)} &

    {{"{clean_filename(video1)}", ?, "{ts1}"}},
    {{"{clean_filename(video2)}", "{ts2}", ?}},
''')

print('')

for (score, ts1, ts2) in candidates[MAX_RESULTS+1:MAX_RESULTS*10]:
    print(f'{iround(score)} for {float_to_ffmpeg_time(ts1)} → {float_to_ffmpeg_time(ts2)}')
