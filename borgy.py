import misc as ms 
import subprocess
import time
def get_borgy_script():
  borgy_script = r'''
  borgy submit --gpu=1 --cpu=4 --mem=20  \
              -v /mnt/home/issam:/mnt/home/issam\
              -v /mnt/datasets:/mnt/datasets\
              -v /mnt/projects/counting:/mnt/projects/counting\
              -i images.borgy.elementai.lan/issam.laradji/maskrcnn \
              -w work_directory \
              --restartable \
              --name issam -- /bin/bash -c borgy_command
  '''
 
  return borgy_script


def get_borgy_command(main_dict, mode):
  config_name = main_dict["config_name"]
  metric_name = main_dict["metric_name"]
  model_name = main_dict["model_name"]
  dataset_name = main_dict["dataset_name"]

  argList = ("-m {} -c {} -metric {} -model {}"
              " -d {} -br {}".format(
                            mode, config_name, metric_name,
                            model_name, dataset_name,
                             1))

  argString = " ".join(argList.split())
  

  # tmp_mode = " %s " % extract(argString, "-mode")
  # argString = argString.replace(tmp_mode, " %s " % mode)

  command = "python main.py %s" % argString  

  return command 

import os

def borgy_submit(main_dict, mode, reset, global_prompt="y"):
  
  if reset == 1:

    borgy_kill(main_dict, mode)
    ms.delete_path_save(main_dict)

    # copy code



  command = get_borgy_command(main_dict, mode)
  job_id, job_state = get_job_id(command)

  if job_is_running(job_state):
    return "%s - %s" % (job_state, job_id)

  else:
    # delete
    code_path = main_dict["path_save"] + "/code"
    if os.path.exists(code_path):
      ms.remove_dir(code_path)

  
    ms.create_dirs(code_path + "/tmp")
    copy_code = "cp -r * "\
              "%s" % code_path

    subprocess.call([copy_code], shell=True)
    time.sleep(0.5)

    work_directory = main_dict["path_save"] + "/code"

    borgy_script = get_borgy_script().replace("work_directory", work_directory)
    borgy_command = borgy_script.replace("borgy_command",  '"%s"' % command)


    subprocess.call([borgy_command], shell=True)
    job_id, job_state = get_job_id(command)

    print(command)
    return  "%s - %s" % (job_state, job_id)

   
      
      


def borgy_status(main_dict, mode):
    command = get_borgy_command(main_dict, mode)
    
    # print(command)


    job_id, job_state = get_job_id(command)
    return "%s - %s" % (job_state, job_id)

def borgy_kill(main_dict, mode):
    command = get_borgy_command(main_dict, mode)
    
    # print(command)

    job_id, job_state = get_job_id(command)
    kill(job_id, force=True)  
    print("\n KILLED: {}\n".format(job_id))
    return "KILLED %s - %s" % (job_state, job_id) 
    

def borgy_logs(main_dict, mode):
    command = get_borgy_command(main_dict, mode)
    
    # print(command)

    job_id, job_state = get_job_id(command)
    subprocess.call(["borgy logs %s" % job_id], shell=True)
    # return "KILLED %s - %s" % (job_state, job_id) 





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
  # for el in ["-d", "-c", "-mode", "-l", "-m", "-model", "-metric"]:
  for el in ["-d", "-m", "-c", "-model","-metric"]:

    if extract(cmd, el) != extract(b_cmd, el):
      return False

  return flag

def get_job_id(command):
    cmdList = run_bash_command("borgy ps").split("\\n")
    
    jobid = None
    status = None
    jobid_failed = None
    status_failed = None

    matchList = {}
    matchList["recent"] = []
    for cmd in cmdList:
      # print(cmd)
      if len(cmd.split()[:4]) == 4:
        jobid, status, user, time = cmd.split()[:4]
        # print(jobid)
    
      else:
        continue

      if is_same(command, cmd):
        if status in matchList:
          matchList[status] += [jobid]
        else:

          matchList[status] = [jobid]

        matchList["recent"] += [[jobid, status]]


    if jobid is None or len(matchList["recent"]) == 0:
      jobid = jobid_failed
      status = status_failed

    else:
      jobid, status = matchList["recent"][0]
    return jobid, status


def job_is_running(job_state):
  return job_state in  ["RUNNING", "QUEUED", "QUEUING"]



def borgy_display(jobid):
    subprocess.call(["borgy logs %s" % jobid], shell=True)


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




# def get_train_command(argList):
#   return get_command(argList, mode="train")


