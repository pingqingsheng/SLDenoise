import asyncio


async def run(cmd):
    
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    
    _, stderr = await proc.communicate()
    
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')
        
        
async def run_tinyimage():
    
    await asyncio.gather(
        run('python -W ignore run_temperature_scaling_tinyimage.py --gpus 0'), 
        run('python -W ignore run_bayes_dropout_tinyimage.py --gpus 0'), 
        run('python -W ignore run_cskd_tinyimage.py --gpus 0'), 
        run('python -W ignore run_ensemble_tinyimage.py --gpus 1'), 
        run('python -W ignore train_ours_binary_share_weighted_tinyimage_dev --gpus 1')
    )

if __name__ == '__main__':
    
    # asyncio.run(run_adaptive())
    # asyncio.run(run_coverage_vs_slrisk_svhn())
    
    # make-up experiment
    asyncio.run(run_tinyimage())