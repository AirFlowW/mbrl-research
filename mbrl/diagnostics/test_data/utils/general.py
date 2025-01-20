import os
import datetime as dt

curr_dir = os.getcwd()
now = dt.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
path_to_save_plots = os.path.join(curr_dir,'exp','plots_test_data',timestamp,'')