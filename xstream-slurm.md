### XStream SU (GPU Compute Units)

> \[T = \vert G \vert \cdot t\] where $T$ is the total SUs charged, $\vert G \vert$ is the total number of GPUs used and $t$ is the total wallclock time in hours.


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
  A credential has been received for user pragaash in /tmp/x509up_u[XXXXX].
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
  - Example on how to load CUDA, cuDNN and TensorFlow:
    ```bash
    [xs-username@xstream-ln0X ~]$ ml CUDA/7.5.18 cuDNN/5.1-CUDA-7.5.18 tensorflow/0.10
    ```
    where `ml` is an alias for `module load`.