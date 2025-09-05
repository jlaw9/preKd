import subprocess


def get_job_params_str(parameters):
    job_str = ("_mp" + str(parameters.num_messages) +
               "_f" + str(parameters.mol_features) +
               "_b" + str(parameters.batch_size) +
               "_lr" + f"{parameters.learning_rate:1.0e}" +
               "_dr" + f"{parameters.dropout * 100:1.0f}" +
               "_dc" + f"{parameters.decay:1.0e}" +
               "_e" + str(parameters.epochs)
              )
    return job_str


# load the relevant training and validation data without reading in the TF GNNs
def read_model_data(loc_models):
    mm_data = []
    for i in range(10):
        data_path = loc_models / f"model_{i}/model_{i}_data.pk"
        with open(data_path, "rb") as f:
            model_data = pk.load(f)
            mm_data += [model_data]
    return mm_data


def write_submit_kestrel(out_dir,
                         mm_data_file,
                         params,
                         job_name,
                         node_idx=0,
                         n_runs_per_node=5,
                         user="jlaw",
                         submit=False):
    """ Create a slurm script file that will run five of the 10 CV jobs on each kestrel GPU
    *n*: the index of the node / GPU this job will run on
    """
    start_idx = node_idx * n_runs_per_node
    end_idx = (node_idx + 1) * n_runs_per_node
    # num_cpus_per_task = 4
    # mem_per_cpu_per_task = int(80 / (n_runs * num_cpus_per_task))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"log_n{node_idx}_nruns{n_runs_per_node}.txt"

    submit_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=robustmicrob
##SBATCH --time=1:00:00
##SBATCH --partition=debug
#SBATCH --time=2-00
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Reserve 1/4 of the GPU node's CPUs and memory for a single GPU
#SBATCH --cpus-per-task=26
#SBATCH --mem=80GB
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --open-mode=append
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jlaw@nrel.gov

module load mamba cuda/12.2 apptainer
conda activate /home/jlaw/.conda-envs/prot

echo "Job started at `date`"
for ((i = {start_idx}; i < {end_idx} ; i++)); do
apptainer run --bind $PWD:/workspace --nv \\
    /scratch/jlaw/tensorflow/tensorflow_24_05.sif \\
    python train_compound_solvent.py \\
        --kfolds $i \\
        --save_folder {out_dir} \\
        --mm_dump {mm_data_file} \\
        --n_messages {params.num_messages} \\
        --af {params.atom_features} \\
        --bf {params.bond_features} \\
        --mf {params.mol_features} \\
        --epochs {params.epochs} \\
        --batch_size {params.batch_size} \\
        --dropout {params.dropout} \\
        --learning_rate {params.learning_rate} \\
        --decay {params.decay} \\
        --smiles_col {params.smiles_col} \\
        &
done

wait
echo "Job finished at `date`"
"""

    submit_file = out_dir / f"gpu_submit_n{node_idx}_nruns{n_runs_per_node}.sh"
    print(submit_file)
    with open(submit_file, 'w') as out:
        out.write(submit_str)

    if submit:
        cmd = f"sbatch {submit_file}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    return submit_file
