

import torch
import main
import pandas as pd
import argparse
import numpy as np
from itertools import product

import experiments
import os

def borgy_submit(project_name, global_prompt, mode, config_name, 
                            metric_name, model_name,
                            dataset_name, loss_name, reset,
                            predict_method):
  command = get_borgy_command(mode, config_name, 
                            metric_name, model_name,
                            dataset_name, loss_name, reset,
                            predict_method)
        
  print(command)

  job_id, job_state = get_job_id(command)

  if job_is_running(job_state):
    return "%s - %s" % (job_state, job_id)

  else:
    if global_prompt != "y":
      prompt = input("Do you want to borgy submit the command:"
             " \n'%s' ? \n(y/n)\n" % command) 
    if global_prompt == "y" or prompt == "y":            
      # if not su.is_exist_train(main_dict):     

      submit(command, project_name=project_name, prompt=False)
      job_id, job_state = get_job_id(command)
      return  "%s - %s" % (job_state, job_id)


def borgy_status(mode, config_name, 
                        metric_name, model_name,
                        dataset_name, loss_name, reset,
                            predict_method):
    command = get_borgy_command(mode, config_name, 
                        metric_name, model_name,
                        dataset_name, loss_name, reset,
                            predict_method)
    
    # print(command)


    job_id, job_state = get_job_id(command)
    return "%s - %s" % (job_state, job_id)

def borgy_kill(mode, config_name, 
                        metric_name, model_name,
                        dataset_name, loss_name, reset,
                            predict_method):
    command = get_borgy_command(mode, config_name, 
                        metric_name, model_name,
                        dataset_name, loss_name, reset,
                            predict_method)
    
    # print(command)

    job_id, job_state = get_job_id(command)
    kill(job_id, force=True)  
    print("{}".format(job_id))
    return "KILLED %s - %s" % (job_state, job_id) 
    






### MISC
def command():
  pass

def get_borgy_script(project_name):
  borgy_script = r'''
  borgy submit --req-gpus=1 --req-cores=2 --req-ram-gbytes=20  \
              -v /mnt:/mnt -v /mnt/datasets:/mnt/datasets\
              -v /mnt/projects/counting:/mnt/projects/counting\
              -i images.borgy.elementai.lan/issam.laradji/v1 \
              -w /home/issam/Research_Ground/End2End/ \
              --name issam -- /bin/bash -c command
  '''
  
  borgy_script = borgy_script.replace("End2End", project_name)
  return borgy_script



# JOB ID

def run_bash_command(command, noSplit=True):
    if noSplit:
        command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    return str(output)


def is_same(cmd, b_cmd):
  b_cmd = b_cmd[b_cmd.find("-c")+3:]

  flag = True
  for el in ["-d", "-c", "-me", "-mode", "-l", "-m", "-model", "-metric", "-p"]:

    if extract(cmd, el) != extract(b_cmd, el):
      return False




  return flag

def get_job_id(command):
    cmdList = run_bash_command("borgy ps | grep RUNNING").split("\\n")
    
    jobid = None
    status = None
    jobid_failed = None
    status_failed = None

    for cmd in cmdList:
      try:
        tmp_jobid, tmp_status = cmd.split()[:2]
      except:
        continue 

      if tmp_status in ["RUNNING", "QUEUED", "QUEUING", "FAILED"]:
        if is_same(command, cmd):

          if tmp_status == "FAILED":
            jobid_failed = tmp_jobid
            status_failed = tmp_status

          else:
            jobid = tmp_jobid
            status = tmp_status

    if jobid is None:
      jobid = jobid_failed
      status = status_failed

    return jobid, status


def job_is_running(job_state):
  return job_state in  ["RUNNING", "QUEUED", "QUEUING"]


import subprocess
def borgy_display(jobid):
    subprocess.call(["borgy logs %s" % jobid], shell=True)


def submit(command, project_name, prompt=False):
  jobid, job_state = get_job_id(command)

  if job_is_running(job_state):
    return "Already Running"  


  borgy_script = get_borgy_script(project_name)

  cmm = borgy_script.replace("command",  '"%s"' % command)

  if prompt is False:
    subprocess.call([cmm], shell=True)
    return "Soon"

  else:
    prompt = input("Do you want to borgy submit the command:"
                   " \n'%s' ? \n(y/n)\n" % command) 
    if prompt == "y":
      subprocess.call([cmm], shell=True)
      return "Soon"

  return "Skipped"

def kill(jobid, force=False):
    if jobid is None:
        return "Not Running"
        
    if force is True:
      subprocess.call(["borgy kill %s" % jobid], shell=True)
      return "Soon"

    else:
      prompt = input("Do you want to delete : \n'%s'"
                     " ? \n(y/n)\n" % (str(jobid))) 
      if prompt == "y":
          subprocess.call(["borgy kill %s" % jobid], shell=True)
          return "Deleted"
      else:
          print("Skipped")
          return "Skipped"
     

def extract(cmd, which="-p"):
    if cmd[0] == "-":
      which = which + " "
    else:
      which = " " + which + " "
    
    sindex = cmd.rfind(which)

    if sindex == -1:
      return None

    sb = cmd[sindex+len(which):]
    next_space = sb.find(" ")

    if next_space == -1:
        return sb.strip()
    else:
        return sb[:next_space].strip()



def get_borgy_command(mode, config_name, metric_name, model_name, dataset_name, loss_name, reset,
                            predict_method):
  argList = ("-m {} -c {} -metric {} -model {}"
              " -d {} -l {} -br {} -r {} -p {}".format(
                            mode, config_name, metric_name,
                            model_name, dataset_name,
                            loss_name, 1, reset, predict_method))

  argString = " ".join(argList.split())
  

  # tmp_mode = " %s " % extract(argString, "-mode")
  # argString = argString.replace(tmp_mode, " %s " % mode)

  command = "python main.py %s" % argString

  return command 


# def get_train_command(argList):
#   return get_command(argList, mode="train")


