from os import listdir, remove
from os.path import exists, join, isfile
from argparse import ArgumentParser

from tqdm import tqdm


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def print_data_sources_stat(data_sources):
    print('Specified {} valid data sources:'.format(len(data_sources)))
    for data_source in data_sources:
        print('   - {}'.format(data_source))


def parse_verified_videos(data_sources):
    verified_video_names = []
    for data_source in data_sources:
        with open(data_source) as input_stream:
            for line in input_stream:
                values = line.strip().split(' ')
                if len(values) == 3:
                    verified_video_names.append(values[0])

    return verified_video_names


def remove_invalid(valid_video_names, videos_dir, extension):
    downloaded_video_names = [f.replace('.{}'.format(extension), '')
                              for f in listdir(videos_dir) if isfile(join(videos_dir, f)) and f.endswith(extension)]

    invalid_video_names = list(set(downloaded_video_names) - set(valid_video_names))

    for video_name in tqdm(invalid_video_names, desc='Removing'):
        video_path = join(videos_dir, '{}.{}'.format(video_name, extension))
        remove(video_path)

    print('Removed {} invalid videos'.format(len(invalid_video_names)))


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='mp4')
    args = parser.parse_args()

    assert exists(args.videos_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    verified_video_names = parse_verified_videos(data_sources)
    remove_invalid(verified_video_names, args.videos_dir, args.extension)


if __name__ == '__main__':
    main()
