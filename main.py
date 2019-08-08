from feature_extraction import create_song_feats
from demo1 import execute_demo1
from music import play_song, execute_spoti

import argparse
import threading
import warnings
warnings.filterwarnings("ignore")

SONG = '/home/maximiliano/datamadrid0619/FINAL_PROJECT/music_classifier/Data/unlabelled_songs/song.mp3'


def parse():
    parser = argparse.ArgumentParser(description="Music classification system")  # analizador de argumentos

    grupo = parser.add_mutually_exclusive_group()  # grupo mutuamente excluyente (solo una operacion)

    grupo.add_argument('-f', '--fext', help='Extracts features from training data',
                       action='store_true')  # action guarda el argumento
    grupo.add_argument('-t', '--train', help='Trains and save the model using training data', action='store_true')
    grupo.add_argument('-d', '--demo', help='Test model using song in path', action='store_true')
    grupo.add_argument('-s', '--spoti', help='Test model using song from spotify', action='store_true')

    parser.add_argument('-p', '--path', help='Path to local file.', type=str, default=SONG)

    print(parser)
    return parser.parse_args()


def main():
    """
    Main: either extract features from train data, train the model or test with single song (from path or spotify)
    """
    args = parse()

    # if args.fext:
    #     print("Extracting features from training data...")
    #     p = create_train_feats()
    #     print(f'Done! you may find the .csv with training data features at: {p}')
    #
    # if args.train:
    #     execute_mts()

    if args.demo:
        thread = threading.Thread(target=play_song, args=(args.path,))
        thread.start()
        print('Extracting song features...')
        create_song_feats(args.path)
        print('Predicting genre...')
        execute_demo1()
        thread.join()
        print("Done!")

    if args.spoti:
        print('SPOTI-DEMO')
        spath = execute_spoti()
        thread = threading.Thread(target=play_song, args=(spath,))
        thread.start()
        print('Extracting song features...')
        create_song_feats(spath)
        print('Predicting genre...')
        execute_demo1()
        thread.join()
        print("Done!")



if __name__ == "__main__":
    main()







