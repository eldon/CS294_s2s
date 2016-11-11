### XStream SU (GPU Compute Units)

> **T** = |_G_| x _t_

> where:
  - **T**: The total SUs charged.
  - |_G_|: The total number of GPUs used.
  - _t_: The total wallclock time in hours.


### Single Sign On (SSO) Hub & XStream Login

- Login via:
  ```bash
  $ ssh -l <username> login.xsede.org
  ```

- Enter your _XUP Password_ when prompted. A directory at `/home/username` will be created on your first login:
  ```bash
  #  Welcome to the XSEDE Single Sign-On (SSO) Hub!
  [username@ssohub ~]$
  ```

- By default, upon logging into the SSO Hub, the X.509 credential is obtained on your behalf and is valid only for a 12-hour period. You can check the validity of this credential via:
  ```bash
  [username@ssohub ~]$ grid-proxy-info
  ```
  and you should see the following:
  ```bash
  subject  : /C=US/O=National Center for Supercomputing Applications/CN=[YOUR NAME HERE]
  issuer   : /C=US/O=National Center for Supercomputing Applications/OU=Certificate Authorities/CN=MyProxy CA 2013
  identity : /C=US/O=National Center for Supercomputing Applications/CN=[YOUR NAME HERE]
  type     : end entity credential
  strength : 2048 bits
  path     : /tmp/x509up_u[XXXX]
  timeleft : 11:45:38
  ```
  Note the `timeleft` entry. Renew your X.509 credential while logged into the SSO Hub via the `myproxy-logon` command with your _XUP Password_:
  ```bash
  [username@ssohub ~]$ myproxy-logon
  Enter MyProxy pass phrase: [YOUR XUP PASSWORD HERE]
  A credential has been received for user [USERNAME] in /tmp/x509up_u[XXXXX].
  ```

- Once logged onto the hub, use the `gsissh` utility to login into XStream Login Node:
  ```bash
  [username@ssohub ~]$ gsissh xstream

  #     --*-*- Stanford University Research Computing Center -*-*--
  #            __  ______  _
  #            \ \/ / ___|| |_ _ __ ___  __ _ _ __ ___
  #             \  /\___ \| __| '__/ _ \/ _` | '_ ` _ \
  #             /  \ ___) | |_| | |  __/ (_| | | | | | |
  #            /_/\_\____/ \__|_|  \___|\__,_|_| |_| |_|

  [xs-username@xstream-ln0X ~]$
  ```
- Submit jobs via Slurm, do cool stuff with GPUs and then `exit`:
  ```bash
  [xs-username@xstream-ln0X ~]$ exit
  [username@ssohub ~]$ exit
  ```

### XStream Login Node

- There are 3 filesystems with each dedicated to specific tasks:

  - `$HOME`: 5GB limited space to store scripts, binaries, logs, etc.
  - `$WORK`: 1TB Lustre FS for computationally expensive I/Os (_store data here_).
  - `$LSTOR` & `$TMPDIR` (_On Compute Node_): Local Scratch Disks (up to 447GB)


- Modules a.k.a. Packages (_including TensorFlow_):

  - List all available modules:
    ```bash
    [xs-username@xstream-ln0X ~]$ module spider
    ```
  - Details on how to load the module and module support info:
    ```bash
    [xs-username@xstream-ln0X ~]$ module spider [MODULE]/[VERSION]
    ```
  - Example on how to load TensorFlow (_which automatically loads CUDA and cuDNN_):
    ```bash
    [xs-username@xstream-ln0X ~]$ ml tensorflow/0.10
    ```
    where `ml` is an alias for `module load`.

### Running Jobs on XStream with SLURM

> Only rule for job submission is CPU:GPU ratio, _r_ should be at most 5:4.

Queues and QoS:

| Slurm QoS | Max CPUs | Max GPUs | Max Jobs | Max Nodes | Job Time Limits |
|:----------|:--------:|:--------:|:--------:|:---------:|:---------------:|
|`normal`| 320/USER, 400/GROUP | 256/USER, 320/GROUP | 512/USER | 16/USER, 20/GROUP | 48 HOURS |
|`long`** | 20/USER, 80/GROUP, 200 MAX TOTAL | 16/USER, 64/GROUP, 160 MAX TOTAL | 4/USER, 64 MAX TOTAL | N/A | 7 DAYS |

 ** Enable the long QoS mode via the `--qos=long` flag when submitting jobs.

 Two steps to running jobs:
- Resource Requests (`SBATCH` prefix)
- Job Steps (`srun` command)

A few useful `SBATCH` parameters (_more in `man sbatch`_) include:
- `--job-name`: Define the name of the job.
- `--output`: Define output file for job completion information.
- `--time`: Set the runtime of the job.
- `--ntasks`: Define the number of tasks. Typically `1` for a single TensorFlow job.
- `--cpus-per-task`: Number of CPUs to be allocated.
- `--mem-per-cpu`: Total memory per CPU in MB. Max is 12800 and default is 12000.
- `--gres`: Typically used to define GPU resources e.g. `--gres gpu:2` for 2 GPUs.
- `--gres-flags`: Used to `enforce-binding` i.e. ensure that the GPUs allcoated all reside within the same CPU socket. May improve communication speed between GPUs.

Putting it all together (_A Sample SLURM scipt_):
- Create the `submit.sh` script:

 ```bash
 #!/bin/bash
 #
 #SBATCH --job-name=tf_trial
 #SBATCH --output=res_%j.txt
 #
 #SBATCH --time=12:00:00
 #SBATCH --ntasks=1
 #SBATCH --cpus-per-task=4
 #SBATCH --gres gpu:4
 #SBATCH --gres-flags=enforce-binding

 ml tensorflow/0.10 protobuf/2.6.1

 python main.py ...
 ```
- Submit the job via the `sbatch` command:
 ```bash
 [xs-username@xstream-ln0X ~]$ sbatch submit.sh
 Submitted batch job XXXX
 ```

Monitoring, Terminating and Gathering Information:
- `scancel`: Used to kill jobs e.g. `scancel [JOB ID]` or `scancel -u [USERNAME]`.
- `squeue`: View PENDING and RUNNING jobs e.g. `squeue -u [USERNAME]`.
- `scontrol show job`: Get full details about a PENDING or RUNNING job e.g. `scontrol show job [JOB ID]`
