import json
from os import makedirs, listdir
from os.path import exists, join, isfile, basename
from argparse import ArgumentParser


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


def filter_invalid(videos, videos_dir, extension, min_length):
    downloaded_video_names = [f.replace('.{}'.format(extension), '')
                              for f in listdir(videos_dir) if isfile(join(videos_dir, f)) and f.endswith(extension)]
    all_video_names = list(videos)
    valid_video_names = list(set(all_video_names).intersection(set(downloaded_video_names)))

    filtered_video_names = {video_name: videos[video_name] for video_name in valid_video_names}

    return filtered_video_names


def make_class_map(videos):
    class_names = list(set([label_name for _, label_name in videos.values()]))
    class_map = {n: i for i, n in enumerate(class_names)}
    return class_map


def prepare_annotation(videos, extension, class_map):
    out_data = dict()
    for video_name, video_desc in videos.items():
        data_type, label = video_desc

        rel_video_path = '{}.{}'.format(video_name, extension)
        label = class_map[label]

        if data_type not in out_data:
            out_data[data_type] = list()

        out_data[data_type].append((rel_video_path, label))

    return out_data


def dump_annotation(annotation, output_dir):
    for data_type, data in annotation.items():
        output_file = join(output_dir, '{}.txt'.format(data_type))
        with open(output_file, 'w') as output_stream:
            for record in data:
                output_stream.write('{} {}\n'.format(*record))

        print('{} annotation stored at: {}'.format(data_type, output_file))


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='mp4')
    parser.add_argument('--min_length', '-l', type=int, required=False, default=16)
    args = parser.parse_args()

    assert exists(args.videos_dir)
    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    all_videos = collect_videos(data_sources)
    filtered_videos = filter_invalid(all_videos, args.videos_dir, args.extension, args.min_length)
    print('Available {} / {} videos'.format(len(filtered_videos), len(all_videos)))

    class_map = make_class_map(all_videos)
    print('Found {} unique classes'.format(len(class_map)))

    annotation = prepare_annotation(filtered_videos, args.extension, class_map)
    dump_annotation(annotation, args.output_dir)


if __name__ == '__main__':
    main()
