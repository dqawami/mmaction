import time
import signal
import json
from os import makedirs, listdir
from os.path import exists, join, isfile, basename
from shutil import rmtree
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import cv2
from tqdm import tqdm


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def print_data_sources_stat(data_sources):
    print('Specified {} valid data sources:'.format(len(data_sources)))
    for data_source in data_sources:
        print('   - {}'.format(data_source))


def collect_videos(data_sources):
    out_videos = dict()
    num_duplicated_videos = 0
    for data_source in data_sources:
        data_type = basename(data_source).replace('.json', '')

        with open(data_source) as input_stream:
            data = json.load(input_stream)

        for record in data.values():
            url = record['url']
            video_name = url.split('?v=')[-1]

            label = record['annotations']['label']

            if video_name in out_videos:
                num_duplicated_videos += 1
            else:
                out_videos[video_name] = data_type, label

    if num_duplicated_videos > 0:
        print('Num duplicated videos: {}'.format(num_duplicated_videos))

    return out_videos


def filter_available(videos, videos_dir, extension):
    downloaded_video_names = [f.replace('.{}'.format(extension), '')
                              for f in listdir(videos_dir) if isfile(join(videos_dir, f)) and f.endswith(extension)]
    all_video_names = list(videos)
    valid_video_names = list(set(all_video_names).intersection(set(downloaded_video_names)))

    filtered_videos = {video_name: videos[video_name] for video_name in valid_video_names}

    return filtered_videos


def parse_completed_tasks(out_dir):
    candidate_files = [join(out_dir, f) for f in listdir(out_dir) if isfile(join(out_dir, f)) and f.endswith('.txt')]

    completed_tasks = []
    for candidate_file in candidate_files:
        with open(candidate_file) as input_stream:
            for line in input_stream:
                values = line.strip().split(' ')
                if len(values) == 3:
                    completed_tasks.append(values[0])

    return completed_tasks


def exclude_completed(videos, completed_names):
    all_video_names = list(videos)
    valid_video_names = list(set(all_video_names) - set(completed_names))

    filtered_videos = {video_name: videos[video_name] for video_name in valid_video_names}

    return filtered_videos


def make_class_map(videos):
    class_names = list(set([label_name for _, label_name in videos.values()]))
    class_map = {n: i for i, n in enumerate(class_names)}
    return class_map


def prepare_tasks(videos, videos_dir, video_extension, images_dir, class_map):
    out_tasks_queue = Queue()
    num_tasks = 0
    for video_name, video_desc in videos.items():
        video_frames_dir = join(images_dir, video_name)
        label = class_map[video_desc[1]]
        video_path = join(videos_dir, '{}.{}'.format(video_name, video_extension))
        task = dict(name=video_name,
                    type=video_desc[0],
                    label=label,
                    video_path=video_path,
                    frames_path=video_frames_dir)

        out_tasks_queue.put(task, True)
        num_tasks += 1

    return out_tasks_queue, num_tasks


def dump_frames(task, image_name_template, image_scale, frame_rate=3):
    video_capture = cv2.VideoCapture(task['video_path'])

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count is None or frame_count <= 0:
        return 0

    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_side_size = max(video_width, video_height)
    if max_side_size > image_scale:
        scale = float(image_scale) / float(max_side_size)
        trg_height, trg_width = int(video_height * scale), int(video_width * scale)
    else:
        trg_height, trg_width = int(video_height), int(video_width)

    ensure_dir_exists(task['frames_path'])

    num_read_frames = 0
    for frame_id in range(frame_count):
        success, frame = video_capture.read()
        if not success:
            break

        if frame_id % frame_rate != 0:
            continue

        resized_image = cv2.resize(frame, (trg_width, trg_height))

        out_image_path = join(task['frames_path'], image_name_template.format(num_read_frames + 1))
        cv2.imwrite(out_image_path, resized_image)

        num_read_frames += 1

    video_capture.release()

    return num_read_frames


def remove_if_exists(dir_path):
    if exists(dir_path):
        rmtree(dir_path)


def annotation_writer(completed_tasks_queue, out_streams, out_dir, pbar):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        if not completed_tasks_queue.empty():
            is_valid, data_type, rel_path, num_frames, label = completed_tasks_queue.get(True)

            if not is_valid:
                pbar.update(1)
                continue

            if data_type not in out_streams:
                out_streams[data_type] = open(join(out_dir, '{}.txt'.format(data_type)), 'a')

            converted_record = (
                str(rel_path),
                str(num_frames),
                str(label)
            )

            out_streams[data_type].write('{}\n'.format(' '.join(converted_record)))
            out_streams[data_type].flush()

            pbar.update(1)
        else:
            time.sleep(1)


def frame_writer(input_tasks_queue, completed_tasks_queue, min_length, image_name_template, image_scale):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while not input_tasks_queue.empty():
        task = input_tasks_queue.get(True)

        num_frames = dump_frames(task, image_name_template, image_scale)
        is_valid = num_frames >= min_length

        if not is_valid:
            remove_if_exists(task['frames_path'])

        completed_task = is_valid, task['type'], task['name'], num_frames, task['label']
        completed_tasks_queue.put(completed_task, True)

    print('Frame writer process has been finished.')


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--video_extension', '-ve', type=str, required=False, default='mp4')
    parser.add_argument('--image_extension', '-ie', type=str, required=False, default='jpg')
    parser.add_argument('--image_scale', '-is', type=int, required=False, default=256)
    parser.add_argument('--min_length', '-ml', type=int, required=False, default=16)
    parser.add_argument('--num_threads', '-n', type=int, required=False, default=8)
    args = parser.parse_args()

    assert exists(args.videos_dir)
    assert args.num_threads >= 1

    out_images_dir = join(args.output_dir, 'rawframes')
    ensure_dir_exists(out_images_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    all_videos = collect_videos(data_sources)
    filtered_videos = filter_available(all_videos, args.videos_dir, args.video_extension)
    print('Available {} / {} videos'.format(len(filtered_videos), len(all_videos)))

    completed_task_names = parse_completed_tasks(args.output_dir)
    if len(completed_task_names) > 0:
        filtered_videos = exclude_completed(filtered_videos, completed_task_names)
        print('Found {} completed tasks'.format(len(completed_task_names)))

    class_map = make_class_map(all_videos)
    print('Found {} unique classes'.format(len(class_map)))

    tasks_queue, num_tasks = prepare_tasks(filtered_videos, args.videos_dir, args.video_extension,
                                           out_images_dir, class_map)
    print('Prepared {} tasks'.format(num_tasks))

    image_name_template = 'img_{:05}' + '.{}'.format(args.image_extension)

    out_streams = dict()
    completed_tasks_queue = Queue()
    pbar = tqdm(total=num_tasks)
    annotation_writer_process = Process(
        target=annotation_writer, args=(completed_tasks_queue, out_streams, args.output_dir, pbar))
    annotation_writer_process.daemon = True
    annotation_writer_process.start()

    frame_writers_pool = []
    for _ in range(args.num_threads):
        frame_writer_process = Process(target=frame_writer,
                                       args=(tasks_queue, completed_tasks_queue, args.min_length,
                                             image_name_template, args.image_scale))
        frame_writer_process.daemon = True
        frame_writer_process.start()
        frame_writers_pool.append(frame_writer_process)

    for frame_writer_process in frame_writers_pool:
        frame_writer_process.join()

    annotation_writer_process.join()


if __name__ == '__main__':
    main()
