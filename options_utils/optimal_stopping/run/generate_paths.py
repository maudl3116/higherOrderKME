# Lint as: python3
"""
Main module to run the algorithms.
"""
import os
import pickle 
import atexit
import csv
import itertools
import multiprocessing
import socket
import random
import time
import psutil

# absl needs to be upgraded to >= 0.10.0, otherwise joblib might not work
from absl import app
from absl import flags
import numpy as np
import shutil

from optimal_stopping.utilities import configs_getter
from optimal_stopping.algorithms.backward_induction import DOS
from optimal_stopping.payoffs import payoff
from optimal_stopping.algorithms.backward_induction import LSM
from optimal_stopping.algorithms.backward_induction import RLSM
from optimal_stopping.algorithms.backward_induction import RRLSM
from optimal_stopping.data import stock_model
from optimal_stopping.algorithms.backward_induction import NLSM
from optimal_stopping.algorithms.reinforcement_learning import RFQI
from optimal_stopping.algorithms.reinforcement_learning import FQI
from optimal_stopping.algorithms.reinforcement_learning import LSPI
from optimal_stopping.run import write_figures


import joblib

# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = SendBotMessage()

NUM_PROCESSORS = multiprocessing.cpu_count()
if 'ada-' in socket.gethostname() or 'arago' in socket.gethostname():
    SERVER = True
    NB_JOBS = int(NUM_PROCESSORS) - 1
else:
    SERVER = False
    NB_JOBS = int(NUM_PROCESSORS) - 1


SEND = False
if SERVER:
    SEND = True





FLAGS = flags.FLAGS

flags.DEFINE_list("nb_stocks", None, "List of number of Stocks")
flags.DEFINE_list("algos", None, "Name of the algos to run.")
flags.DEFINE_bool("print_errors", False, "Set to True to print errors if any.")
flags.DEFINE_integer("nb_jobs", NB_JOBS, "Number of CPUs to use parallelly")
flags.DEFINE_bool("generate_pdf", False, "Whether to generate latex tables")

_CSV_HEADERS = ['algo', 'model', 'payoff', 'drift', 'volatility', 'mean',
                'speed', 'correlation', 'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'spot', 'strike', 'dividend',
                'maturity', 'nb_epochs', 'hidden_size', 'factors',
                'ridge_coeff',
                'train_ITM_only', 'use_path',
                'price', 'duration']

_PAYOFFS = {
    "MaxPut": payoff.MaxPut,
    "MaxCall": payoff.MaxCall,
    "GeometricPut": payoff.GeometricPut,
    "BasketCall": payoff.BasketCall,
    "Identity": payoff.Identity,
    "Max": payoff.Max,
    "Mean": payoff.Mean,
}

_STOCK_MODELS = {
    "BlackScholes": stock_model.BlackScholes,
    "FractionalBlackScholes": stock_model.FractionalBlackScholes,
    "FractionalBrownianMotion": stock_model.FractionalBrownianMotion,
    'FractionalBrownianMotionPathDep':
        stock_model.FractionalBrownianMotionPathDep,
    "Heston": stock_model.Heston,
}

_ALGOS = {
    "LSM": LSM.LeastSquaresPricer,
    "LSMLaguerre": LSM.LeastSquarePricerLaguerre,
    "LSMRidge": LSM.LeastSquarePricerRidge,
    "FQI": FQI.FQIFast,
    "FQILaguerre": FQI.FQIFastLaguerre,
    "LSPI": LSPI.LSPI,  # TODO: this is a slow version -> update similar to FQI

    "NLSM": NLSM.NeuralNetworkPricer,
    "DOS": DOS.DeepOptimalStopping,

    "RLSM": RLSM.ReservoirLeastSquarePricerFast,
    "RLSMRidge": RLSM.ReservoirLeastSquarePricerFastRidge,

    "RRLSM": RRLSM.ReservoirRNNLeastSquarePricer,

    "RFQI": RFQI.FQI_ReservoirFast,
    "RRFQI": RFQI.FQI_ReservoirFastRNN,
}

_NUM_FACTORS = {
    "RRLSMmix": 3,
    "RRLSM": 2,
    "RLSM": 1,
}



def init_seed():
  random.seed(0)
  np.random.seed(0)


def generate_paths():

  # to save the paths 
  name_config = ""
  for config_name, config in configs_getter.get_configs():  
      name_config = config_name 

  fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                       f'{name_config}.obj')


  nb_stocks_flag = [int(nb) for nb in FLAGS.nb_stocks or []]

  for config_name, config in configs_getter.get_configs():
    print(f'Config {config_name}', config)
    config.algos = [a for a in config.algos
                    if FLAGS.algos is None or a in FLAGS.algos]
    if nb_stocks_flag:
      config.nb_stocks = [a for a in config.nb_stocks
                          if a in nb_stocks_flag]
    combinations = list(itertools.product(
        config.algos, config.dividends, config.maturities, config.nb_dates,
        config.nb_paths, config.nb_stocks, config.payoffs, config.drift,
        config.spots, config.stock_models, config.strikes, config.volatilities,
        config.mean, config.speed, config.correlation, config.hurst,
        config.nb_epochs, config.hidden_size, config.factors,
        config.ridge_coeff,
        config.train_ITM_only, config.use_path))

    paths = []

    for params in combinations:
        stock_paths = _generate_paths(*params)
        paths.append(stock_paths.transpose(0,2,1))

  pickle.dump(paths,open(fpath,'wb'))

  return paths


def _generate_paths(
        algo, dividend, maturity, nb_dates, nb_paths,
        nb_stocks, payoff, drift, spot, stock_model, strike, volatility, mean,
        speed, correlation, hurst, nb_epochs, hidden_size=10,
        factors=(1.,1.,1.), ridge_coeff=1.,
        train_ITM_only=True, use_path=False,
        fail_on_error=False):
  """
  This functions generates paths. It is called by generate_paths()
  which is called in main(). Below the inputs are listed which have to be
  specified in the config that is passed to main().
  """
  print(spot, volatility, maturity, nb_paths, '... ', end="")

  sample_paths = _STOCK_MODELS[stock_model](
      drift=drift, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
      correlation=correlation, nb_stocks=nb_stocks,
      nb_paths=nb_paths, nb_dates=nb_dates,
      spot=spot, dividend=dividend,
      maturity=maturity).generate_paths()
 
  return sample_paths


def main(argv):
  del argv

  try:
      if SEND:
          SBM.send_notification(
              text='start running AMC2 with config:\n{}'.format(FLAGS.configs),
              chat_id="-399803347"
          )

      filepath = generate_paths()

      if FLAGS.generate_pdf:
          write_figures.write_figures()
          write_figures.generate_pdf()

      if SEND:
          time.sleep(1)
          SBM.send_notification(
              text='finished',
              files=[filepath],
              chat_id="-399803347"
          )
  except Exception as e:
      if SEND:
          SBM.send_notification(
              text='ERROR\n{}'.format(e),
              chat_id="-399803347"
          )
      else:
          print('ERROR\n{}'.format(e))




if __name__ == "__main__":
  app.run(main)
