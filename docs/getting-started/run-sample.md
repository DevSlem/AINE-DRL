---
sort: 1
---

# Run Sample

Run a sample script in [samples](samples/) directory. Enter the following command:

```bash
python samples/<FILE_NAME>
```

Example:

```bash
python samples/cartpole_v1_ppo.py
```

then, you can see the training information in your shell:

```
+----------------------------------------------+
| AINE-DRL Training Start!                     |
|==============================================|
| ID: CartPole-v1_PPO                          |
| Output Path: results/CartPole-v1_PPO         |
|----------------------------------------------|
| Training INFO:                               |
|     number of environments: 3                |
|     total time steps: 20000                  |
|     summary frequency: 1000                  |
|     agent save frequency: 10000              |
|----------------------------------------------|
| Agent INFO:                                  |
|     name: PPO                                |
|     device: cpu                              |
+----------------------------------------------+

[AINE-DRL] training time: 1.30, time steps: 1000, cumulated reward: 30.00
[AINE-DRL] training time: 2.57, time steps: 2000, cumulated reward: 115.00
[AINE-DRL] training time: 3.88, time steps: 3000, cumulated reward: 133.00
```

When the training is finished, you can see the results (tensorboard, log message, agent save file) in `results` directory.

Open the tensorboard result by entering the command:

```bash
tensorboard --logdir=results
```

or

```bash
tensorboard --logdir=results/<SUB_DIRECTORY>
```

If you want to change the inference export format like gif, png (default: real-time rendering), you need to change the `Inference` setting in the configuration file. Follow the example:

```yaml
Inference:
  Config:
    export: gif # default: render_only
```

`export` detail options: `None`, `render_only`, `gif`, `png`