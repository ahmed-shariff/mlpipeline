import os
import sys
import importlib.util
import shutil
import configparser
import logging
import socket
import string
import re
from datetime import datetime

from utils import ExecutionModeKeys
from utils import version_parameters
from utils import VersionLog
from utils import console_colors

#from helper import Model
#from utils import Versions
#from helper import DataLoader

from global_values import MODELS_DIR
from global_values import OUTPUT_FILE
from global_values import HISTORY_FILE
from global_values import TRAINING_HISTORY_LOG_FILE
from global_values import LOG_FILE
from global_values import NO_LOG
#from global_values import DATA_CODE_MAPPING
from global_values import EXECUTED_MODELS
from global_values import USE_BLACKLIST
from global_values import LISTED_MODELS
#from global_values import EPOC_COUNT

from global_values import TEST_MODE

from global_values import mtime
from global_values import version
from global_values import train_time
from global_values import vless

# from global_values import BATCH_SIZE
# from global_values import USE_ALL_CLASSES
# from global_values import CLASSES_COUNT
# from global_values import CLASSES_OFFSET
# from global_values import ALLOW_DELETE_MODEL_DIR
# from global_values import RESTART_GLOBAL_STEP
# from global_values import MODEL_DIR_SUFFIX

#tf.logging.set_verbosity(tf.logging.INFO)

formatter = logging.Formatter(fmt= "%(asctime)s:{0}{1}%(levelname)s:{2}%(name)s{3}- %(message)s" \
                              .format(console_colors.BOLD,
                                      console_colors.BLUE_FG,
                                      console_colors.GREEN_FG,
                                      console_colors.RESET),
                              datefmt="%Y-%m-%d %H:%M:%S")
                                                                              
LOGGER = logging.getLogger("mlp")
LOGGER.handlers = []
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
LOGGER.addHandler(handler)

def _main():
  # sys.path.append(os.getcwd())
  print(list((k, m[version].exectued_versions)  for k,m in EXECUTED_MODELS.items()))
    
  current_model, version_name, clean_model_dir = getNextModel()
  while current_model is not None:
    add_to_and_return_result_string("Model: {0}".format(current_model.name), True)
    add_to_and_return_result_string("Version: {0}".format(version_name))
    log("Model loaded: {0}".format(current_model.name))
    if version_name is None:
      log("No Version Specifications",
          logging.WARNING,
          modifier_1 = console_colors.RED_FG,
          modifier_2 = console_colors.BOLD)
    else:
      log("version loaded: {0}".format(version_name),
          modifier_1 = console_colors.GREEN_FG,
          modifier_2 = console_colors.BOLD)
      
    #print("\033[1;32mMode: {0}\033[0m".format(modestring))
    if TEST_MODE:
      log("Mode: {}TESTING".format(console_colors.YELLOW_FG),
          modifier_1 = console_colors.BOLD,
          modifier_2 = console_colors.GREEN_FG)
    else:
      log("{0}{1}Mode: {3}RUNNING MODEL TRAINING{4}".format(console_colors.RED_FG),
          modifier_1 = console_colors.BOLD,
          modifier_2 = console_colors.GREEN_FG)
      
    version_spec = current_model.versions.getVersion(version_name)
      
    batch_size = version_spec[version_parameters.BATCH_SIZE]
    model_dir_suffix = version_spec[version_parameters.MODEL_DIR_SUFFIX]
    dataloader = version_spec[version_parameters.DATALOADER]
    
    if TEST_MODE:
      record_training = False
      model_dir = "{0}/outputs/model_ckpts/temp".format(MODELS_DIR.rstrip("/"))
      shutil.rmtree(model_dir, ignore_errors=True)
      test__eval_steps = 1
      train_eval_steps = 1
    else:
      record_training = True
      model_dir="{0}/outputs/model_ckpts/{1}-{2}".format(MODELS_DIR.rstrip("/"),
                                                         current_model.name.split(".")[-2],
                                                         model_dir_suffix)
      test__eval_steps = dataloader.get_test_sample_count()#len(dataLoader.test_files)
      train_eval_steps = dataloader.get_train_sample_count()#len(dataLoader.train_files)
      #int(len(dataLoader.train_files)/dataLoader.batch_size)
    # Train the model
    #classifier_executed=False
    #exception_count = batch_size #maximum numeber of possible time this can loop!! if more, prolly inifinite loop
    eval_complete=False
    #training_done = False
    #while not classifier_executed:
    LOGGER.setLevel(logging.INFO)
    try:
      if clean_model_dir and current_model.allow_delete_model_dir:
        #shutil.rmtree(model_dir, ignore_errors=True)
        current_model.clean_model_dir(model_dir)
        #print("\033[1;038mclearning folder {0}\033[0m".format(model_dir))
        log("{0}{1}{}")
        # if clean_model_dir and not allow_delete_model_dir and restart_global_step:
      #   reset_gs = True
      # else:
      #   reset_gs = False
      # s = mySess()
      # hooks = [s, logging_hook] + verion_hooks
      # classifier = tf.estimator.Estimator(
      #   model_fn=current_model.get_model_fn(version_name, dataLoader.classes_count, reset_gs),
      #   model_dir=model_dir)

      current_model.pre_execution_hook(version_spec, model_dir)
      save_training_time(current_model, version_name)
      #classification_steps = getClassificationSteps(TEST_MODE, dataLoader, model_dir, epoc_count, reset_gs)
      classification_steps = getTrainingSteps(ExecutionModeKeys.TRAIN, current_model, clean_model_dir)
      log("Steps: {0}".format(classification_steps))
      if classification_steps > 0:
        # classifier.train(input_fn = dataLoader.get_train_input_fn(),
        #                  steps= classification_steps,
        #                  hooks = hooks)
        train_output = current_model.train_model(dataloader.get_train_input_fn(), classification_steps)
        log("Model traning output: {0}".format(train_output))
        log("Model trained")
        #training_done = True
      else:
        # classifier.train(input_fn = dataLoader.get_train_input_fn(),
        #                  steps= 1,
        #                  hooks = [logging_hook,s])

        log("No training. Loaded pretrained model")
        #training_done = False
      #classifier_executed = True

      # if s.tvar is not None:
      #   log("Trainable parms: {0}".format(
      #     sum([v.flatten().shape[0] for k,v in s.tvar.items()])),
      #       log_tf=True)
      #   print({k: v.flatten().shape[0] for k,v in s.tvar.items()})
      # print("*************************************************")
      # print(len([k for k,v in s.mvar.items()]))
      # print([k for k,v in s.mvar.items()])
      # print("*************************************************")
      # print(len([k for k,v in s.gvar.items()]))
      # print([k for k,v in s.gvar.items()])
      # Evaluate the model and print results
      try:
        log("Training evaluation started: {0} steps".format(train_eval_steps))
        train_results = current_model.evaluate_model(dataloader.get_train_input_fn(mode = ExecutionModeKeys.TEST),
        #classifier.evaluate(input_fn = dataLoader.get_train_input_fn(tf.estimator.ModeKeys.EVAL),# dataLoader.get_test_input_fn(),
                                               steps = train_eval_steps)
      # except tf.errors.InvalidArgumentError:
      #   tf.logging.set_verbosity(tf.logging.INFO)
      #   raise
      except Exception as e:
        #tf.logging.set_verbosity(tf.logging.INFO)
        if TEST_MODE:
          raise
        train_results = "Training evaluation failed: {0}".format(str(e))
        log(train_results, logging.ERROR)
        
      try:
        log("Testing evaluation started: {0} steps".format(test__eval_steps))
        #tf.logging.set_verbosity(tf.logging.ERROR)
        eval_results = current_model.evaluate_model(dataloader.get_test_input_fn(),
        #classifier.evaluate(input_fn = dataLoader.get_test_input_fn(),
                                                    steps = test__eval_steps)
        # except tf.errors.InvalidArgumentError:
        #   tf.logging.set_verbosity(tf.logging.INFO)
        #   raise
      except Exception as e:
        #tf.logging.set_verbosity(tf.logging.INFO)
        if TEST_MODE:
          raise
        eval_results = "Test evaluation failed: {0}".format(str(e))
        log(eval_results, logging.ERROR)
        
      log("Model evaluation complete")
    # except tf.errors.ResourceExhaustedError:
    #   dataLoader.batch_size -= 1
    #   log("ResourceExhaustedError: reducing batch_size to {0}".format(dataLoader.batch_size))
    #   print("\033[1;031mResourceExhaustedError: reducing batch_size to {0}\033[0m".format(dataLoader.batch_size))
    # except tf.errors.InvalidArgumentError as e:
    #   if not allow_delete_model_dir:
    #     log("{0}".format(str(e)))
    #     log("Not cleaning folder, skiping evaluation")
    #   else:
    #     classifier_executed = False
    #     classifier =None
    #     #shutil.rmtree(model_dir, ignore_errors=True)
    #     subprocess.run(["rm", "-rf", model_dir])
    #     log("InvalidArgumentError: clearning folder {0}".format(model_dir), logging.ERROR)
    #     print("\033[1;031mInvalidArgumentError: clearning folder {0}\033[0m".format(model_dir))
    # TODO: Nan error: reduce learning rate
    except Exception as e:
      if TEST_MODE is True:
        raise
      else:
        log("Exception: {0}".format(str(e)), logging.ERROR)
        #TODO: do this?
        # if NO_LOG:
        #   raise
    # try:
    #   print("Training result: {0}".format(train_results))
    # except                  :
    #   print("Training result: evaluation failed")
    #   eval_results="Evaluation failed due to unknown reason"
    # try:
    #   print("Evaluation result: {0}".format(eval_results))
    # except:
    #   print("Evaluation result: evaluation failed")
    # eval_results="Evaluation failed due to unknown reason"
    add_to_and_return_result_string("Eval on train set: {0}".format(train_results))
    add_to_and_return_result_string("Eval on test  set: {0}".format(eval_results))
    add_to_and_return_result_string("-------------------------------------------")
    add_to_and_return_result_string("EXECUTION SUMMERY:")
    add_to_and_return_result_string("Number of epocs: {0}".format(version_spec[version_parameters.EPOC_COUNT]))
    add_to_and_return_result_string("-------------------------------------------")
    add_to_and_return_result_string("MODEL SUMMERY:")
    add_to_and_return_result_string(current_modelxg.summery)
    add_to_and_return_result_string("-------------------------------------------")
    add_to_and_return_result_string("DATALOADER  SUMMERY:")
    add_to_and_return_result_string(dataloader.summery)
    if record_training:
      save_results_to_file(add_to_and_return_result_string(), version_name, current_model)
        #current_model, eval_results, train_results, dataLoader, training_done, model_dir)
    current_model,version_name, clean_model_dir  = getNextModel()


def getTrainingSteps(mode, model, clean_model_dir):
  if mode == ExecutionModeKeys.TEST:
    return 1
  else:
    current_version = model.get_current_version()
    complete_steps =  current_version[version_parameters.EPOC_COUNT] * \
                      current_version[version_parameters.DATALOADER].get_train_sample_count() / \
                      current_version[version_parameters.BATCH_SIZE]
    global_step = model.get_trained_step_count()
    if global_step is None or model.reset_steps:
      return complete_steps

    #TODO: why did i add the reset_step here?
    elif clean_model_dir and not model.allow_delete_model_dir and model.reset_steps:
      return complete_steps
    else:
      if complete_steps > global_step:
        return complete_steps - global_step
      else:
        return 0
      
def getNextModel(just_return_model=False):
  config_update()
  for rdir, dirs, files in os.walk(MODELS_DIR):
    for f in files:
      if f.endswith(".py"):
        file_path = os.path.join(rdir,f)
        # TODO: Should remove this check, prolly has no use!
        if True:#file_path not in EXECUTED_MODELS or EXECUTED_MODELS[file_path][mtime] + 5 < os.path.getmtime(file_path): 
          if USE_BLACKLIST and file_path in LISTED_MODELS:
            continue
          if not USE_BLACKLIST and file_path not in LISTED_MODELS:
            continue
          spec = importlib.util.spec_from_file_location(f,file_path)
          module = importlib.util.module_from_spec(spec)
          spec.loader.exec_module(module)
          clean_model_dir = False
          model = None
          try:
            # if module.IS_MODEL is False:
            #   continue
            # else:
            model = module.MODEL
            model.name = file_path
          except:
            log("{0} is not a model script. It does not contain a `MODEL` global variable".format(file_path))
            continue
          
          #TODO: why did i add this in the first place??
          if just_return_model:
            print("\033[1;33mJust returning module\033[1;0m")
            return module
          #module.__name__ = file_path
          returning_version = None
          try:
            versions = model.versions
          except:
            versions = None
          # print("\n\033[1;32mProcessing new model: {0}\033[1;0m\n".format(module.__name__))  
          log("{0}{1}Processing new model: {2}{3}".format(console_colors.BOLD,
                                                          console_colors.BLUE_FG,
                                                          model.name,
                                                          console_colors.RESET))
          with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
            t_history = [line.rstrip("\n") for line in t_hist_file]
            all_history = [t_entry.split("::") for t_entry in t_history]
            module_history = [(v,float(t)) for n,v,t in all_history if n == model.name]
          
          #print("\n\033[1;32mEvaluating model: {0}\033[1;0m\n".format(module.__name__))
          if file_path not in EXECUTED_MODELS:
            EXECUTED_MODELS[model.name] = {}
            EXECUTED_MODELS[model.name][train_time]=0
            EXECUTED_MODELS[model.name][version]=VersionLog()
            
          EXECUTED_MODELS[model.name][mtime] = os.path.getmtime(file_path)

          # if versions is None:
          #   l = [t for v,t in module_history if v == vless]
          #   if len(l) == 0 or  max(l) < os.path.getmtime(file_path):
          #     clean_model_dir= True
          #     EXECUTED_MODELS[module.__name__][version].clean()
          # else:
          reset_model_dir = True
          for v,t in module_history:
            if t > os.path.getmtime(file_path):
              reset_model_dir = False
          if reset_model_dir:
            clean_model_dir = True
            EXECUTED_MODELS[model.name][version].clean()
          else:
            t_ = os.path.getmtime(file_path)
            versions__ = [v_.name for v_ in versions.versions]
            for v,t in module_history:
              #print(v,t)
              if t > t_:
                if EXECUTED_MODELS[model.name][version].executed(v) is not VersionLog.EXECUTED and v in versions__:
                  t_ = t
                  returning_version = v
          if returning_version is None:
            #TODO: check if this like works:
            for v,k in sorted(versions.versions.items(), key=lambda x:x[1][version_parameters.ORDER]):
              if EXECUTED_MODELS[model.name][version].executed(v) is not VersionLog.EXECUTED:
                returning_version = v
                clean_model_dir = True
          log("Executed versions: {0}".format(EXECUTED_MODELS[model.name][version].exectued_versions),
              log=False)
          if returning_version is None:
            continue

          
          return model, returning_version, clean_model_dir
  return None, None, False

def add_to_and_return_result_string(message=None, reset_result_string = False, indent = True):
  global result_string
  if message is not None:
    if indent:
      message = "\t\t" + message
    if reset_result_string:
      result_string = message + "\n"
    else:
      result_string += message + "\n"
  return result_string

def save_training_time(model, version_):
  if TEST_MODE:
    return
  name = model.name
  with open(TRAINING_HISTORY_LOG_FILE, "a") as log_file:
    time = datetime.now().timestamp()
    EXECUTED_MODELS[name][version].addExecutingVersion(version_, time)
    log("Executing version: {0}".format(EXECUTED_MODELS[model.name][version].executing_version),
        log=False)
    log_file.write("{0}::{1}::{2}\n".format(name,
                                            EXECUTED_MODELS[name][version].executing_version,
                                            time))

    
def save_results_to_file(resultString, version, model):#model, result, train_result, dataloader, training_done, model_dir):
  modified_dt = datetime.isoformat(datetime.fromtimestamp(EXECUTED_MODELS[model.name][mtime]))
  result_dt = datetime.now().isoformat()
  
  #add_to_and_return_result_string("\n[{0}]:ml-pipline: output: \n".format(result_dt))
  with open(OUTPUT_FILE, 'a', encoding = "utf-8") as outfile:
    outfile.write("\n[{0}]:ml-pipline: output: \n")
    outfile.write(resultString)
    # outfile.write("[{0}]: Evaluation on test-set of model{1}: \n\t\t\t{2}\n".format(result_dt,
    #                                                         model.__name__,
    #                                                         result))
    # outfile.write("\tEvaluation on train-set of model{0}: \n\t\t\t{1}\n".format(model.__name__, train_result))
    # outfile.write("\t\tused models mtime:{0}\n".format(modified_dt))
    # outfile.write("\t\tmodel dir: {0}".format(model_dir))
    # if not training_done:
    #   outfile.write("\tNO TRAINING. ONLY EVAL\n\n")
    # try:
    #   outfile.write("\t\tModel Summery:\n{0}\n".format(model.MODEL_SUMMERY))
    # except:
    #   outfile.write("\t\tModel Summery not provided\n")
    # outfile.write("\t\tdataSummery:\n\t\t Loader summery: {0}\n".format(dataloader.summery))
    # outfile.write("\t\tClasses: {0}\n".format(dataloader.classes_count))
    # outfile.write("\t\tTraining data: {0}\n".format(len(dataloader.train_files)))
    # outfile.write("\t\tTesting data: {0}\n".format(len(dataloader.test_files)))
    # outfile.write("\t\tTraining batch size: {0}\n".format(dataloader.batch_size))
    # try:
    #   outfile.write("\t\tTraining epocs: {0}\n".format(model.EPOC_COUNT))
    # except:
    #   outfile.write("\t\tTraining epocs: {0}\n".format(EPOC_COUNT))
    # outfile.write("\n\t\tUsed labels: {0}\n\n".format({i:dataloader.DATA_CODE_MAPPING[i] for i in dataloader.used_labels}))
      
  with open(HISTORY_FILE, 'a', encoding = "utf-8") as hist_file:
    hist_file.write("{0}::{1}::{2}\n".format(model.name,
                                             EXECUTED_MODELS[model.name][mtime],
                                             EXECUTED_MODELS[model.name][version].executing_version))
  EXECUTED_MODELS[model.name][version].moveExecutingToExecuted()

def log(message, level = logging.INFO, log=True, modifier_1=None, modifier_2=None):
  # if level is not logging.INFO and level is not logging.ERROR:
  #   raise AttributeError("level cannot be other than logging.INFO or logging.ERROR, coz i am lazy to get others in here")
  if modifier_1 is None and modifier_2 is None:
    reset_string = ""
  else:
    reset_string = console_colors.RESET
    
  if modifier_1 is None:
    modifier_1 = ""
  if modifier_2 is None:
    modifier_2 = ""

  message = "{0}{1}{2}{3}".format(modifier_1, modifier_2, message, reset_string)
  
  LOGGER.log(level, message)
  if not TEST_MODE and not NO_LOG:
    with open(LOG_FILE, 'a', encoding="utf-8") as log_file:
      level = ["INFO" if level is logging.INFO else "ERROR"]
      time = datetime.now().isoformat()
      cleaned_message = re.sub("\[[0-9;m]*", "", message.translate(str.maketrans({"\x1b":None})))
      log_file.write("[{0}]::{1} - {2}\n".format(time, level[0], cleaned_message))

def config_update():
  config = configparser.ConfigParser(allow_no_value=True)
  config_file = config.read("cnn.config")
  global MODELS_DIR
  global USE_BLACKLIST
  global LISTED_MODELS
  
  global HISTORY_FILE
  global LOG_FILE
  global OUTPUT_FILE
  global TRAINING_HISTORY_LOG_FILE
  if len(config_file)==0:
    print("\033[1;031mWARNING:\033[0:031mNo 'cnn.config' file found\033[0m")
  else:
    try:
      config["CNN_PARAM"]
    except KeyError:
      print("\033[1;031mWARNING:\033[0:031mNo CNN_PARAM section in 'cnn.config' file\033[0m")
    MODELS_DIR = config.get("CNN_PARAM", "models_dir", fallback=MODELS_DIR)
    USE_BLACKLIST =  config.getboolean("CNN_PARAM", "use_blacklist", fallback=USE_BLACKLIST)
    try:
      if USE_BLACKLIST:
        LISTED_MODELS = config["BLACKLISTED_MODELS"]
      else:
        LISTED_MODELS = config["WHITELISTED_MODELS"]
      l = []
      for model in LISTED_MODELS:
        l.append(os.path.join(MODELS_DIR, model))
      LISTED_MODELS = l
      print("\033[1;036m{0}\033[0;036m: {1}\033[0m".format(
        ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0].replace("_"," "),
        LISTED_MODELS).lower())
    except KeyError:
      print("\033[1;031mWARNING:\033[0:031mNo {0} section in 'cnn.config' file\033[0m".format(
        ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0]))

  hostName = socket.gethostname()
  OUTPUT_FILE = MODELS_DIR + "/output-{0}".format(hostName)
  HISTORY_FILE = MODELS_DIR + "/history-{0}".format(hostName)
  TRAINING_HISTORY_LOG_FILE = MODELS_DIR + "/t_history-{0}".format(hostName)
  LOG_FILE = MODELS_DIR + "/log-{0}".format(hostName)
  open(OUTPUT_FILE, "a").close()
  open(HISTORY_FILE, "a").close()
  open(TRAINING_HISTORY_LOG_FILE, "a").close()
  open(LOG_FILE, "a").close()

      

def main(unused_argv):
  global TEST_MODE
  global NO_LOG
  config_update()
  if len(unused_argv)> 1:
    if any("r" in s for s in unused_argv) :
      TEST_MODE = False
    else:
      TEST_MODE = True
      
    if any("h" in s for s in unused_argv):
      if not os.path.isfile(HISTORY_FILE) and not os.path.isfile(TRAINING_HISTORY_LOG_FILE):
        print("\033[1;31mWARNING: No 'history' file in 'models' folder. No history read\033[0m")
      else:
        with open(HISTORY_FILE, 'r', encoding = "utf-8") as hist_file:
          history = [line.rstrip("\n") for line in hist_file]
          for hist_entry in history:
            hist_entry = hist_entry.split("::")
            name=hist_entry[0]
            time=hist_entry[1]
            ttime=0
            v = None
            if len(hist_entry) > 2:
              v = hist_entry[2]
            if name not in EXECUTED_MODELS:
              EXECUTED_MODELS[name] = {}
              EXECUTED_MODELS[name][mtime] = float(time)
              EXECUTED_MODELS[name][version] = VersionLog()
              if v is not None and v is not "":
                EXECUTED_MODELS[name][version].addExecutedVersion(v)
              #needs to be taken from seperate file
              #EXECUTED_MODELS[name][train_time] = float(ttime)
            else:
              if EXECUTED_MODELS[name][mtime] < float(time):
                EXECUTED_MODELS[name][mtime] = float(time)
                EXECUTED_MODELS[name][version].clean()
              if v is not None and v is not "":
                EXECUTED_MODELS[name][version].addExecutedVersion(v)
                #EXECUTED_MODELS[name][train_time] = float(ttime)
        with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
          t_history = [line.rstrip("\n") for line in t_hist_file]
          for t_entry in t_history:
            n,v,t = t_entry.split("::")
            t = float(t)
            if name in EXECUTED_MODELS:
              if EXECUTED_MODELS[name][mtime] < t and EXECUTED_MODELS[name][version].executed(v) is not VersionLog.EXECUTED:
                EXECUTED_MODELS[name][version].addExecutingVersion(v,t)
                
    if any("nl" in s for s in unused_argv):
      NO_LOG=True
    else:
      NO_LOG=False
      
    # if any("-b" in s for s in unused_argv):
    #   if not os.path.isfile(HISTORY_FILE):
    #     print("\033[1;31mWARNING: No 'blacklist' file in 'models' folder, No models blacklisted\033[0m")
    #   else:
    #     with open(HISTORY_FILE, 'r') as bl_file:
    #       BLACKLISTED_MODELS = [line.rstrip("\n") for line in bl_file]    

  # if not TEST_MODE:
  #   logging.basicConfig(filename=LOG_FILE, format = '%(asctime)s ::%(levelname)s - %(message)s')
  # else:
  #   logging.basicConfig(format = '%(acstime)s ::%(levelname)s - %(message)s')
  log("=====================CNN session started")
  _main()
  log("=====================CNN Session ended")


    
if __name__ == "__main__":
  tf.app.run()
