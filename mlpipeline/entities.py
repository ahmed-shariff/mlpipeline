class ExperimentModeKeys():
    '''
    Enum class that defines the keys to use to specify the execution mode of the experiment
'''
    RUN = 'Run'
    TEST = 'Test'
    EXPORT = 'Export'


class ExecutionModeKeys():
    '''
    Enum class that defines the keys to use to specify the execution mode the pipeline is currently at.
'''
    TRAIN = 'Train'
    TEST = 'Test'


class console_colors():
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLACK_FG = "\033[30m"
    RED_FG = "\033[31m"
    GREEN_FG = "\033[32m"
    YELLOW_FG = "\033[33m"
    BLUE_FG = "\033[34m"
    MEGENTA_FG = "\033[35m"
    CYAN_FG = "\033[36m"
    WHITE_FG = "\033[37m"
    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    MEGENTA_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"


class version_parameters():
    '''
    Enum class that defines eums for some of the parameters used in versions
    '''
    NAME = "name"
    DATALOADER = "dataloader"
    BATCH_SIZE = "batch_size"
    EPOCH_COUNT = "epoch_count"
    LEARNING_RATE = "learning_rate"
    EXPERIMENT_DIR_SUFFIX = "experiment_dir_suffix"
    ORDER = "order"
