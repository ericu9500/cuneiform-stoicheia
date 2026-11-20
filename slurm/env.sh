# source this on a GH200 node
export CUN=/nobackup/proj/disk/dionysus/shared/cuneiform-stoicheia
export HF_HOME=/nobackup/proj/disk/dionysus/shared/hf_cache
export PYTHONPATH=$CUN
export TMPDIR=/tmp
export LD_PRELOAD=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}
source /nobackup/proj/disk/dionysus/shared/greek-novel/emo_analysis/envs/vllm/bin/activate
cd $CUN
