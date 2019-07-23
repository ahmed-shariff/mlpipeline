from mlpipeline import get_experiment


def main():
    print(get_experiment('sample_experiment.py', '', 'version5'))


if __name__ == '__main__':
    main()
