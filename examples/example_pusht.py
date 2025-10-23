if __name__ == "__main__":
    import datasets
    import numpy as np
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=25,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    world.set_policy(swm.policy.RandomPolicy())
    world.record_dataset(
        "example-pusht",
        episodes=10,
        seed=2347,
        options=None,
    )

    world.record_video_from_dataset(
        "./",
        "example-pusht",
        episode_idx=[0, 1],
    )

    ################
    ##  Pretrain  ##
    ################

    # SKIP TRAINING - Assume we have a pretrained checkpoint
    # swm.pretraining(
    #     "scripts/train/dinowm.py",
    #     dataset_name="example-pusht",
    #     output_model_name="dummy_pusht",
    #     dump_object=True,
    # )
    
    # Create a dummy checkpoint for testing (simple world model)
    import torch
    cache_dir = swm.data.get_cache_dir()
    
    # Debug: print action space info
    print(f"Action space (vectorized): {world.envs.action_space.shape}")
    print(f"Single action space: {world.envs.single_action_space.shape}")
    print(f"Single action space shape[1:]: {world.envs.single_action_space.shape[1:]}")
    
    # The action dimension calculation should match what the solver will use
    # The solver uses: int(np.prod(action_space.shape[1:]))
    # where action_space is the VECTORIZED action space (2, 2) -> shape[1:] = (2,)
    base_action_dim = int(np.prod(world.envs.action_space.shape[1:]))
    action_block = 5  # Must match the PlanConfig below
    action_dim = base_action_dim * action_block
    
    print(f"base_action_dim: {base_action_dim}, action_block: {action_block}, total action_dim: {action_dim}")
    
    dummy_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
    checkpoint_path = cache_dir / "dummy_pusht_object.ckpt"
    torch.save(dummy_model, checkpoint_path)
    print(f"Created dummy checkpoint at {checkpoint_path}")

    #########################
    ##  Transform/Process  ##
    #########################

    def img_transform():
        transform = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform

    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = swm.data.get_cache_dir() / "example-pusht"
    dataset = datasets.load_from_disk(str(dataset_path)).with_format("numpy")

    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset["action"][:])

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset["proprio"][:])

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    ################
    ##  Evaluate  ##
    ################

    # Use CPU instead of CUDA (Mac doesn't have CUDA support)
    device = "cpu"  # or "mps" for Apple Silicon GPU acceleration
    model = swm.policy.AutoCostModel("dummy_pusht").to(device)
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device=device)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    world.set_policy(policy)
    results = world.evaluate(episodes=3, seed=2347)

    print(results)
