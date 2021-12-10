#!/usr/bin/env python
# encoding: utf-8
"""
mixed.py

Created by Matthew Loper on 2012-06-05.
Copyright (c) 2012 MPI. All rights reserved.
"""

import marshal
import pickle
import stat
import subprocess
import sys


bashrc_fname = '/is/cluster/nsaini/.bashrc'
py3_fname = '/is/cluster/nsaini/.venvs/copenet/bin/python'
cluster_tmp = '/is/cluster/nsaini/cluster_tmp'
cluster_log = '/is/cluster/nsaini/cluster_log'

def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

list_into_chuncks = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]

# Always env is executed first and then python
# env needs to be executed first to inherit virtualenvironment
# This way python is executed from /home/user/venv/bin/python
# requirements = TARGET.CUDACapability>=7.0

condor_template = """
executable = <<BASH_SCRIPT_FNAME>>
arguments = <<PYTHON_FUNC_FNAME>> <<FUNC_FNAME>> <<INPUT_FNAME>> $(Process)
error = <<LOG_PATH>>/error.$(Process).err
output = <<LOG_PATH>>/output.$(Process).out
log = <<LOG_PATH>>/log.$(Process).log
request_memory = <<MEMORYMBS>>
request_cpus = <<CPU_COUNT>>
request_gpus = <<GPU_COUNT>>
concurrency_limits=user.mytag:<<concurrency_limits>>
requirements = <<REQUIREMENTS>>
getenv = True
on_exit_hold = (ExitCode =?= 3)
on_exit_hold_reason = "Checkpointed, will resume"
on_exit_hold_subcode = 2
periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 2) )
queue <<NJOBS>>
"""
# request_gpus = <<GPU_COUNT>>
#
#


# Usage: on login.cluster.is.localnet, type "condor_submit test.sub"

pythonscript = """
import pickle
import os
import types
import marshal
import sys

if __name__ == '__main__':

    func_fname = sys.argv[1]
    input_arg_fname = sys.argv[2]
    index = int(sys.argv[3])
    sys.stderr.write('func_fname=%s\\n'%func_fname)
    sys.stderr.write('input_arg_fname=%s\\n'%input_arg_fname)
    sys.stderr.write('index=%s\\n'%index)
    sys.stderr.write('python=%s\\n'%sys.version)

    with open(input_arg_fname, 'rb') as fp:
        input_args = pickle.load(fp)[index] # this is a list of args
        
    with open(func_fname, 'rb') as fp:
        func = types.FunctionType(marshal.loads(pickle.load(fp)), globals(), "some_func_name")

    for input_arg in input_args:
        result = func(input_arg)

"""

runscript = """

python_script_fname=$1
func_fname=$2
input_fname=$3
index=$4

source /is/cluster/nsaini/.bashrc
 
. /is/cluster/nsaini/.venvs/copenet/bin/activate

module load cuda/10.1
module load cudnn/7.5-cu10.1

/is/cluster/nsaini/.venvs/copenet/bin/python ${python_script_fname} ${func_fname} ${input_fname} ${index}

exit
"""


def mixedmap(func, seq, max_mem=8, verbose=False, jobs_per_instance=1, bid_amount=49, use_highend_gpus=False,
             log_dir=None, cpu_count=1, gpu_count=0, concurrency_limits=1, username=''):
    import time, os
    import random
    import string
    seq = list_into_chuncks(seq, jobs_per_instance)

    def eprint(s):
        if verbose:
            sys.stderr.write('parallel_internal: ' + s + '\n')

    rnd_name = ''.join([random.choice(string.ascii_lowercase) for i in range(6)])
    tmpdir = makepath(cluster_tmp, '%s_%s' % (time.strftime("%Y%m%d_%H%M"), rnd_name))
    if log_dir is None:
        log_dir = makepath(cluster_log, '%s_%s' % (time.strftime("%Y%m%d_%H%M"), rnd_name))

    # save inputs to disk
    eprint('saving inputs to disk')
    input_fname = tmpdir + '/input.pkl'
    with open(input_fname, 'wb') as fp:
        pickle.dump(seq, fp)

    # save our "func" param to disk
    eprint('saving callback function to disk')
    func_fname = tmpdir + '/callback.pkl'
    with open(func_fname, 'wb') as fp:
        pickle.dump(marshal.dumps(func.__code__), fp)

    # get filenames we will use for output
    # output_fnames = [tmpdir + '/output_%04d.pkl' % (i,) for i in xrange(len(seq))]
    output_fname = tmpdir + '/output'
    output_fnames = ['%s_%d.pkl' % (output_fname, iii) for iii in range(len(seq))]

    # construct caller.py, which will load and call the "func"
    eprint('constructing/saving caller script')

    python_fname = tmpdir + '/pythonscript.py'
    with open(python_fname, 'w') as fp:
        fp.write(pythonscript)
    os.chmod(python_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC)  # make executable

    caller_fname = tmpdir + '/caller.sh'
    with open(caller_fname, 'w') as fp:
        fp.write(runscript)
    os.chmod(caller_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC)  # make executable

    cs = condor_template
    # cs = cs.replace('<<EXE>>', 'env')
    # cs = cs.replace('<<EXE>>', '/home/gpons/venv_geist/bin/python')
    cs = cs.replace('<<BASH_SCRIPT_FNAME>>', caller_fname)
    cs = cs.replace('<<PYTHON_FUNC_FNAME>>', python_fname)
    cs = cs.replace('<<FUNC_FNAME>>', func_fname)
    cs = cs.replace('<<INPUT_FNAME>>', input_fname)
    cs = cs.replace('<<OUTPUT_ARG_FNAME>>', output_fname)
    cs = cs.replace('<<CPU_COUNT>>', str(int(cpu_count)))
    cs = cs.replace('<<GPU_COUNT>>', str(int(gpu_count)))
    cs = cs.replace('<<concurrency_limits>>', str(int(concurrency_limits)))
    cs = cs.replace('<<MEMORYMBS>>', str(int(max_mem * 1024)))
    cs = cs.replace('<<NJOBS>>', str(len(seq)))
    cs = cs.replace('<<LOG_PATH>>', str(log_dir) if log_dir else '')
    requirements = []
    if gpu_count == 0:
        requirements.append("TotalGPUs =?= 0")
    elif use_highend_gpus:
        requirements.append("TARGET.CUDACapability>=7.0")
        # requirements.append("TARGET.CUDACapability>=7.0 && TARGET.CUDAGlobalMemoryMb > 20000")

    cs = cs.replace('<<REQUIREMENTS>>', "&&".join(requirements))

    condor_fname = tmpdir + '/condor_script.sub'
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
    os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC)  # make executable

    # submit jobs
    eprint('submitting jobs')
    # cmd = 'qsub -e %s -p %d -o %s -b y -pe parallel 1 -R y -l h_vmem=%.2fG -t %d-%d -j y %s' % \
    #       (stderr_dir, priority, stdout_dir, max_mem, 1, 1+len(seq), caller_fname,)

    cmd = 'source %s; . /is/cluster/nsaini/.venvs/copenet/bin/activate; condor_submit_bid %d %s' % (bashrc_fname, bid_amount, condor_fname,)
    # cmd = 'source ~/.profile; condor_submit_bid 200 %s' % (condor_fname,)
    
    subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (username,)] + [cmd])

    return



