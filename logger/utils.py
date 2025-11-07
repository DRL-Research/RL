

def log_training_results_to_neptune(logger, training_results):
    """
    Logs training results (reward/losses per episode) to Neptune.
    """
    episode_rewards = training_results.get("episode_rewards", [])
    master_policy_losses = training_results.get("master_policy_losses", [])
    master_value_losses = training_results.get("master_value_losses", [])
    master_total_losses = training_results.get("master_total_losses", [])
    agent_policy_losses = training_results.get("agent_policy_losses", [])
    agent_value_losses = training_results.get("agent_value_losses", [])
    agent_total_losses = training_results.get("agent_total_losses", [])

    for i in range(len(episode_rewards)):
        logger.log_metric("episode/reward", episode_rewards[i])
        if i < len(master_policy_losses) and master_policy_losses[i] is not None:
            logger.log_metric("master/policy_loss", master_policy_losses[i])
            logger.log_metric("master/value_loss", master_value_losses[i])
            logger.log_metric("master/total_loss", master_total_losses[i])
        if i < len(agent_policy_losses) and agent_policy_losses[i] is not None:
            logger.log_metric("agent/policy_loss", agent_policy_losses[i])
            logger.log_metric("agent/value_loss", agent_value_losses[i])
            logger.log_metric("agent/total_loss", agent_total_losses[i])




def close_logger(logger):
    try:
        logger.close()
    except Exception:
        pass
