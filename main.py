from feature_extraction import *
from model_train_save import *
from demo1 import *
from music import *



import argparse
import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description="Music classification system")  # analizador de argumentos

    grupo = parser.add_mutually_exclusive_group()  # grupo mutuamente excluyente (solo una operacion)

    grupo.add_argument('-f', '--fext', help='Extracts features from training data',
                       action='store_true')  # action guarda el argumento
    grupo.add_argument('-t', '--train', help='Trains and save the model using training data', action='store_true')
    grupo.add_argument('-d', '--demo', help='Test model using song in path', action='store_true')
    grupo.add_argument('-s', '--spoti', help='Test model using song from spotify', action='store_true')

    parser.add_argument('path', help='Path to local file.', type=str)


    return parser.parse_args()


def main():
    """
    Main: either extract features from train data, train the model or test with single song (from path or spotify)
    """
    args = parse()

    if args.fext:
        print("Extracting features from training data...")
        p = create_train_feats()
        print(f'Done! you may find the .csv with training data features at: {p}')

    if args.train:
        execute_mts()

    if args.demo:
        play_song(args.path)
        print('Extracting song features...')
        create_song_feats(args.path)
        print('Predicting genre...')
        execute_demo1()
        print("Done!")

    if args.spoti:
        pass


if __name__ == "__main__":
    main()








