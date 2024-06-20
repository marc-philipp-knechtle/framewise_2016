# FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
# --> python 3.6.7
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# Changed back to pytorch:1.0 because of permission issues
# https://github.com/conda/conda/issues/11208
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    build-essential

# duplicate to the pytorch dockerfile definition (just as a reminder)
WORKDIR /workspace

RUN mkdir /workspace/runs
RUN mkdir /workspace/splits

# The final solutions for the joblib_cache permission issues were to link these files with CephFS
# https://github.com/conda/conda/issues/10690
# https://github.com/conda/conda/issues/11208
# https://github.com/conda/conda/issues/6852
RUN mkdir /workspace/joblib_cache && chmod 777 /workspace/joblib_cache && \
    mkdir /opt/conda/envs/framewise_2016 && chmod -R 777 /opt/conda/envs/framewise_2016 && \
    chown root:root /opt/conda/envs/framewise_2016
RUN chmod -r ./joblib_cache
RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache
# Error: Matplotlib created a temporary config/cache directory at /tmp/matplotlib-bebti0l7 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
# Solution:
RUN mkdir -p /workspace/.config/matplotlib &&  \
    chmod 777 /workspace/.config/matplotlib &&  \
    MPLCONFIGDIR=/workspace/.config/matplotlib

# conda environment preparation
RUN conda install pip
RUN conda install Cython
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "framewise_2016", "/bin/bash", "-c"]

# It's necessary to install the madmom dependency after the environment creation, because of Cython issue
# See: https://github.com/CPJKU/madmom/issues/463
RUN ["conda", "run", "-n", "framewise_2016", "pip", "install", "madmom==0.16.1"]

# This lead to version related issues with the installed
# I chose to create a environment using conda because I can specify the concrete python version with this method
# COPY requirements_00.txt .
# COPY requirements_01.txt .
# RUN conda install pip
# RUN pip install -r requirements_00.txt
# RUN pip install -r requirements_01.txt

COPY *.py .


# ENTRYPOINT ["/bin/bash"]
# ENTRYPOINT ["conda", "run", "-n", "framewise_2016", "python3", "train.py", "splits/configuration-II", "runs/VGG", "VGGNet2016"]








# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

#    Traceback (most recent call last):
#      File "/opt/conda/lib/python3.8/site-packages/conda/exceptions.py", line 1080, in __call__
#        return func(*args, **kwargs)
#      File "/opt/conda/lib/python3.8/site-packages/conda/cli/main.py", line 84, in _main
#        exit_code = do_call(args, p)
#      File "/opt/conda/lib/python3.8/site-packages/conda/cli/conda_argparse.py", line 83, in do_call
#        return getattr(module, func_name)(args, parser)
#      File "/opt/conda/lib/python3.8/site-packages/conda/cli/main_run.py", line 25, in execute
#        script_caller, command_args = wrap_subprocess_call(on_win, context.root_prefix, prefix,
#      File "/opt/conda/lib/python3.8/site-packages/conda/utils.py", line 403, in wrap_subprocess_call
#        with Utf8NamedTemporaryFile(mode='w', prefix=tmp_prefix, delete=False) as fh:
#      File "/opt/conda/lib/python3.8/site-packages/conda/auxlib/compat.py", line 88, in Utf8NamedTemporaryFile
#        return NamedTemporaryFile(
#      File "/opt/conda/lib/python3.8/tempfile.py", line 540, in NamedTemporaryFile
#        (fd, name) = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
#      File "/opt/conda/lib/python3.8/tempfile.py", line 250, in _mkstemp_inner
#        fd = _os.open(file, flags, 0o600)
#    PermissionError: [Errno 13] Permission denied: '/opt/conda/envs/framewise_2016/.tmp3ruzfr9d'
#
#`$ /opt/conda/bin/conda run -n framewise_2016 python3 train.py splits/configuration-II runs/VGG VGGNet2016`
#
#  environment variables:
#                 CIO_TEST=<not set>
#               CONDA_ROOT=/opt/conda
#           CURL_CA_BUNDLE=<not set>
#          LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#                     PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin
#                          :/bin
#       REQUESTS_CA_BUNDLE=<not set>
#            SSL_CERT_FILE=<not set>
#
#     active environment : None
#       user config file : /.condarc
# populated config files :
#          conda version : 4.11.0
#    conda-build version : 3.21.7
#         python version : 3.8.12.final.0
#       virtual packages : __cuda=12.3=0
#                          __linux=4.18.0=0
#                          __glibc=2.27=0
#                          __unix=0=0
#                          __archspec=1=x86_64
#       base environment : /opt/conda  (read only)
#      conda av data dir : /opt/conda/etc/conda
#  conda av metadata url : None
#           channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
#                          https://repo.anaconda.com/pkgs/main/noarch
#                          https://repo.anaconda.com/pkgs/r/linux-64
#                          https://repo.anaconda.com/pkgs/r/noarch
#          package cache : /opt/conda/pkgs
#                          /.conda/pkgs
#       envs directories : /.conda/envs
#                          /opt/conda/envs
#               platform : linux-64
#             user-agent : conda/4.11.0 requests/2.27.1 CPython/3.8.12 Linux/4.18.0-513.9.1.el8_9.x86_64 ubuntu/18.04.6 glibc/2.27
#                UID:GID : 1432:1001
#             netrc file : None
#           offline mode : False
#
#
#An unexpected error has occurred. Conda has prepared the above report.
