# Getting Started on Isambard :rocket:

The following instructions specifically deal with setting up a Conda python environment with pytorch on the [Isambard MACS](https://gw4-isambard.github.io/docs/user-guide/MACS.html) system. This implicitly assumes that you have access to Isambard as detailed in the [Isambard doc pages](https://gw4-isambard.github.io/docs/index.html)

1. **Login to MACS login-node**

```Shell
$ ssh login-01.isambard.gw4.ac.uk
```
---

2. **Activate conda** 

:exclamation: This should be a one time only step...

```Shell
[<user>@login-01 ~]$ module use /software/x86/modulefiles/
[<user>@login-01 ~]$ module load tools/anaconda3
```

Check conda is loaded sucesfully, eg. `conda info` should return info about the conda version and the locations it will use to save packages and environments

```Shell
[<user>@login-01 ~]$ conda info
```
Then set-up conda initalization in your `.bashrc` file:
```Shell
[<user>@login-01 ~]$ conda init bash
no change     /software/x86/tools/anaconda3/condabin/conda
no change     /software/x86/tools/anaconda3/bin/conda
no change     /software/x86/tools/anaconda3/bin/conda-env
no change     /software/x86/tools/anaconda3/bin/activate
no change     /software/x86/tools/anaconda3/bin/deactivate
no change     /software/x86/tools/anaconda3/etc/profile.d/conda.sh
no change     /software/x86/tools/anaconda3/etc/fish/conf.d/conda.fish
no change     /software/x86/tools/anaconda3/shell/condabin/Conda.psm1
no change     /software/x86/tools/anaconda3/shell/condabin/conda-hook.ps1
no change     /software/x86/tools/anaconda3/lib/python3.9/site-packages/xontrib/conda.xsh
no change     /software/x86/tools/anaconda3/etc/profile.d/conda.csh
modified      /home/mo-hsteptoe/.bashrc

==> For changes to take effect, close and re-open your current shell. <==
```

Restart the shell:
```Shell
[<user>@login-01 ~]$ bash
```
and now the base environment will load by default:
```Shell
(base) [<user>@login-01 ~]$
```

---

3. **Create ml-tc conda environment**

Using the lockfile from this github repo, recreate the `ml-tc` conda environment (assuming you already a cloned local copy):
```Shell
(base) [<user>@login-01 ~]$ cd ML-TC
(base) [<user>@login-01 ~]$ conda create -n ml-tc --file ml-tc-gpu-linux-64.lock
```
Packages should download and install, taking O(10 minutes), but eventually you should see `done`. Check that you can sucesfully activate the new environment
```Shell
(base) [<user>@login-01 ~]$ conda activate ml-tc
(ml-tc) [<user>@login-01 ~]$
```
Note the change in the command prompt from `(base)` to `(ml-tc)`

---

4. **Check pytorch on GPU node**

Login to a GPU node and check that pytorch has installed sucesfully and that it recognises CUDA.  In this case we use `qsub` to start an interactive session on the Pascal `pascal` node with the `-I` flag.  Some further information about interactive jobs [here](https://gw4-isambard.github.io/docs/user-guide/jobs.html#interactive-job).  Other GPU nodes are available on MACS - cross-reference :point_right: https://gw4-isambard.github.io/docs/user-guide/jobs.html#queue-configuration and https://gw4-isambard.github.io/docs/user-guide/MACS.html

```Shell
(base) [<user>@login-01 ~]$ qsub -I -q pascalq -l select=1:ngpus=2
qsub: waiting for job <job-id>.gw4head to start
qsub: job <job-id>.gw4head ready

cd /home/<user>/pbs.<job-id>.gw4head.x8z
(base) [<user>@pascal-002 ~]$ cd /home/mo-hsteptoe/pbs.<job-id>.gw4head.x8z
(base) [<user>@pascal-002 pbs.<job-id>.gw4head.x8z]$
```
:question: I don't understand why it automatically drops you into a job-id specific folder...

Now reactivate your conda `ml-tc` environment and check pytorch:

```Shell
(base) [<user>@pascal-002 pbs.<job-id>.gw4head.x8z]$ conda activate ml-tc
(ml-tc) [<user>@pascal-002 pbs.<job-id>.gw4head.x8z]$ python
```
```Python
>>> import torch
>>> torch.cuda.is_available()
True
```
If this returns `True` then pytorch can see the GPUs.

:exclamation: The default `qsub` interactive time limit is short, so be sure to start a new interactive session with a specified walltime for a longer session

## Batch Submission to GPUs

For batch submissio via `qsub` you need to explicityly set `CUDA_VISIBLE_DEVICES` environment variable for `pytorch` to access the GPUs.

This can be done in two ways.  Via your batch submission script, run before your `pytorch` code:

```Shell
export CUDA_VISIBLE_DEVICES=0,1
python3 <pytorch-script.py>
```

or at the start of your python script

```Python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```
