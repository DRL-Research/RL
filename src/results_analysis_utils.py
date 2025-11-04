import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedResultsAnalyzer:
    """
    Comprehensive analysis and visualization tool for enhanced training results.
    """

    def __init__(self, results_directory: str = "analysis_results"):
        self.results_directory = results_directory
        self.ensure_directory_exists()

        # Analysis cache
        self._analysis_cache = {}

        # Visualization settings
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.color_palette = sns.color_palette("husl", 10)

    def ensure_directory_exists(self):
        """Ensure analysis results directory exists."""
        os.makedirs(self.results_directory, exist_ok=True)

    def analyze_training_session(self, training_results: Dict[str, List],
                                 experiment_config, save_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single training session.

        Args:
            training_results: Dictionary containing training metrics
            experiment_config: Experiment configuration object
            save_plots: Whether to save generated plots

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting comprehensive training session analysis...")

        analysis = {
            'basic_metrics': self._analyze_basic_metrics(training_results),
            'performance_trends': self._analyze_performance_trends(training_results),
            'training_stability': self._analyze_training_stability(training_results),
            'agent_scaling_analysis': self._analyze_agent_scaling(training_results),
            'curriculum_effectiveness': self._analyze_curriculum_effectiveness(training_results),
            'loss_analysis': self._analyze_loss_patterns(training_results),
        }

        # Enhanced metrics if available
        if 'success_rates' in training_results:
            analysis['success_rate_analysis'] = self._analyze_success_rates(training_results)

        if 'episode_lengths' in training_results:
            analysis['efficiency_analysis'] = self._analyze_efficiency(training_results)

        # Generate visualizations
        if save_plots:
            self._generate_comprehensive_plots(training_results, experiment_config, analysis)

        # Save analysis results
        self._save_analysis_results(analysis, experiment_config)

        logger.info("Training session analysis completed")
        return analysis

    def _analyze_basic_metrics(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze basic training metrics."""
        rewards = results.get('episode_rewards', [])
        if not rewards:
            return {'status': 'no_data'}

        return {
            'total_episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'improvement': np.mean(rewards[-50:]) - np.mean(rewards[:50]) if len(rewards) >= 100 else 0,
            'reward_variance_trend': self._calculate_variance_trend(rewards)
        }

    def _analyze_performance_trends(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        rewards = results.get('episode_rewards', [])
        if len(rewards) < 10:
            return {'status': 'insufficient_data'}

        # Calculate moving averages
        window_sizes = [10, 25, 50, 100]
        moving_averages = {}

        for window in window_sizes:
            if len(rewards) >= window:
                ma = pd.Series(rewards).rolling(window=window).mean().tolist()
                moving_averages[f'ma_{window}'] = ma

        # Trend analysis
        recent_episodes = min(100, len(rewards))
        trend_slope, _ = np.polyfit(range(recent_episodes), rewards[-recent_episodes:], 1)

        # Performance phases
        phases = self._identify_performance_phases(rewards)

        return {
            'moving_averages': moving_averages,
            'trend_slope': trend_slope,
            'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
            'performance_phases': phases,
            'convergence_episode': self._find_convergence_point(rewards),
            'plateau_episodes': self._find_plateau_episodes(rewards)
        }

    def _analyze_training_stability(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze training stability across different components."""
        stability_metrics = {}

        # Reward stability
        rewards = results.get('episode_rewards', [])
        if rewards:
            stability_metrics['reward_stability'] = 1.0 / (1.0 + np.std(rewards[-50:]))

        # Loss stability
        for loss_type in ['master_total_losses', 'agent_total_losses']:
            losses = [l for l in results.get(loss_type, []) if l is not None]
            if losses:
                recent_losses = losses[-20:]
                stability_metrics[f'{loss_type}_stability'] = 1.0 / (1.0 + np.std(recent_losses))

        # Training phase consistency
        phases = results.get('training_phases', [])
        if phases:
            phase_changes = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1])
            stability_metrics['training_phase_stability'] = 1.0 / (1.0 + phase_changes / len(phases))

        return stability_metrics

    def _analyze_agent_scaling(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze the effectiveness of agent scaling."""
        agent_counts = results.get('agent_counts', [])
        rewards = results.get('episode_rewards', [])

        if not agent_counts or not rewards:
            return {'status': 'no_scaling_data'}

        # Performance by agent count
        performance_by_agents = defaultdict(list)
        for count, reward in zip(agent_counts, rewards):
            performance_by_agents[count].append(reward)

        scaling_analysis = {}
        for count, reward_list in performance_by_agents.items():
            scaling_analysis[f'agents_{count}'] = {
                'mean_reward': np.mean(reward_list),
                'std_reward': np.std(reward_list),
                'episodes': len(reward_list),
                'success_rate': len([r for r in reward_list if r > 0]) / len(reward_list)
            }

        # Scaling efficiency
        agent_counts_unique = sorted(performance_by_agents.keys())
        if len(agent_counts_unique) > 1:
            scaling_efficiency = []
            for i in range(1, len(agent_counts_unique)):
                prev_perf = scaling_analysis[f'agents_{agent_counts_unique[i - 1]}']['mean_reward']
                curr_perf = scaling_analysis[f'agents_{agent_counts_unique[i]}']['mean_reward']
                efficiency = curr_perf / prev_perf if prev_perf != 0 else 0
                scaling_efficiency.append(efficiency)
            scaling_analysis['scaling_efficiency'] = scaling_efficiency

        return scaling_analysis

    def _analyze_curriculum_effectiveness(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze curriculum learning effectiveness."""
        curriculum_levels = results.get('curriculum_levels', [])
        rewards = results.get('episode_rewards', [])

        if not curriculum_levels or not rewards:
            return {'status': 'no_curriculum_data'}

        # Performance by curriculum level
        performance_by_level = defaultdict(list)
        for level, reward in zip(curriculum_levels, rewards):
            performance_by_level[level].append(reward)

        curriculum_analysis = {}
        for level, reward_list in performance_by_level.items():
            curriculum_analysis[f'level_{level}'] = {
                'mean_reward': np.mean(reward_list),
                'std_reward': np.std(reward_list),
                'episodes': len(reward_list),
                'improvement_rate': self._calculate_improvement_rate(reward_list)
            }

        # Curriculum progression effectiveness
        levels = sorted(performance_by_level.keys())
        if len(levels) > 1:
            progression_effectiveness = []
            for i in range(1, len(levels)):
                prev_perf = curriculum_analysis[f'level_{levels[i - 1]}']['mean_reward']
                curr_perf = curriculum_analysis[f'level_{levels[i]}']['mean_reward']
                effectiveness = (curr_perf - prev_perf) / abs(prev_perf) if prev_perf != 0 else 0
                progression_effectiveness.append(effectiveness)
            curriculum_analysis['progression_effectiveness'] = progression_effectiveness

        return curriculum_analysis

    def _analyze_loss_patterns(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze loss patterns and convergence."""
        loss_analysis = {}

        loss_types = ['master_policy_losses', 'master_value_losses', 'master_total_losses',
                      'agent_policy_losses', 'agent_value_losses', 'agent_total_losses']

        for loss_type in loss_types:
            losses = [l for l in results.get(loss_type, []) if l is not None]
            if losses:
                loss_analysis[loss_type] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'min': np.min(losses),
                    'max': np.max(losses),
                    'convergence_episode': self._find_loss_convergence(losses),
                    'trend': np.polyfit(range(len(losses)), losses, 1)[0] if len(losses) > 1 else 0,
                    'stability': 1.0 / (1.0 + np.std(losses[-20:]))
                }

        return loss_analysis

    def _analyze_success_rates(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze success rate patterns."""
        success_rates = results.get('success_rates', [])
        if not success_rates:
            return {'status': 'no_success_data'}

        return {
            'mean_success_rate': np.mean(success_rates),
            'final_success_rate': success_rates[-1] if success_rates else 0,
            'peak_success_rate': np.max(success_rates),
            'success_rate_trend': np.polyfit(range(len(success_rates)), success_rates, 1)[0],
            'time_to_50_percent': self._find_threshold_episode(success_rates, 0.5),
            'time_to_80_percent': self._find_threshold_episode(success_rates, 0.8),
            'success_rate_stability': 1.0 / (1.0 + np.std(success_rates[-20:]))
        }

    def _analyze_efficiency(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Analyze episode efficiency metrics."""
        episode_lengths = results.get('episode_lengths', [])
        rewards = results.get('episode_rewards', [])

        if not episode_lengths or not rewards:
            return {'status': 'no_efficiency_data'}

        # Calculate efficiency (reward per step)
        efficiency = [r / l if l > 0 else 0 for r, l in zip(rewards, episode_lengths)]

        return {
            'mean_efficiency': np.mean(efficiency),
            'std_efficiency': np.std(efficiency),
            'efficiency_trend': np.polyfit(range(len(efficiency)), efficiency, 1)[0],
            'mean_episode_length': np.mean(episode_lengths),
            'episode_length_trend': np.polyfit(range(len(episode_lengths)), episode_lengths, 1)[0],
            'efficiency_improvement': np.mean(efficiency[-25:]) - np.mean(efficiency[:25]) if len(
                efficiency) >= 50 else 0
        }

    def _generate_comprehensive_plots(self, results: Dict[str, List],
                                      experiment_config, analysis: Dict[str, Any]):
        """Generate comprehensive visualization plots."""
        logger.info("Generating comprehensive visualization plots...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # Plot 1: Episode Rewards with Moving Average
        plt.subplot(3, 4, 1)
        self._plot_episode_rewards(results)

        # Plot 2: Success Rates
        plt.subplot(3, 4, 2)
        self._plot_success_rates(results)

        # Plot 3: Loss Patterns
        plt.subplot(3, 4, 3)
        self._plot_loss_patterns(results)

        # Plot 4: Agent Scaling
        plt.subplot(3, 4, 4)
        self._plot_agent_scaling(results)

        # Plot 5: Episode Lengths
        plt.subplot(3, 4, 5)
        self._plot_episode_lengths(results)

        # Plot 6: Training Phases
        plt.subplot(3, 4, 6)
        self._plot_training_phases(results)

        # Plot 7: Efficiency Analysis
        plt.subplot(3, 4, 7)
        self._plot_efficiency_analysis(results)

        # Plot 8: Curriculum Levels
        plt.subplot(3, 4, 8)
        self._plot_curriculum_levels(results)

        # Plot 9: Performance Distribution
        plt.subplot(3, 4, 9)
        self._plot_performance_distribution(results)

        # Plot 10: Convergence Analysis
        plt.subplot(3, 4, 10)
        self._plot_convergence_analysis(results)

        # Plot 11: Stability Metrics
        plt.subplot(3, 4, 11)
        self._plot_stability_metrics(analysis)

        # Plot 12: Summary Statistics
        plt.subplot(3, 4, 12)
        self._plot_summary_statistics(analysis)

        plt.tight_layout()

        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"comprehensive_analysis_{experiment_config.EXPERIMENT_ID}_{timestamp}.png"
        plot_path = os.path.join(self.results_directory, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comprehensive plots saved to {plot_path}")

        # Generate individual detailed plots
        self._generate_detailed_plots(results, experiment_config, analysis)

    def _plot_episode_rewards(self, results):
        """Plot episode rewards with moving averages."""
        rewards = results.get('episode_rewards', [])
        if not rewards:
            plt.text(0.5, 0.5, 'No reward data', ha='center', va='center')
            return

        plt.plot(rewards, alpha=0.3, color='blue', label='Episode Rewards')

        # Moving averages
        if len(rewards) >= 25:
            ma_25 = pd.Series(rewards).rolling(25).mean()
            plt.plot(ma_25, color='red', label='MA 25')

        if len(rewards) >= 100:
            ma_100 = pd.Series(rewards).rolling(100).mean()
            plt.plot(ma_100, color='green', label='MA 100')

        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

    def _plot_success_rates(self, results):
        """Plot success rates over time."""
        success_rates = results.get('success_rates', [])
        if not success_rates:
            plt.text(0.5, 0.5, 'No success rate data', ha='center', va='center')
            return

        plt.plot(success_rates, color='green', linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Target')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% Target')
        plt.title('Success Rate Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

    def _plot_loss_patterns(self, results):
        """Plot training loss patterns."""
        master_losses = [l for l in results.get('master_total_losses', []) if l is not None]
        agent_losses = [l for l in results.get('agent_total_losses', []) if l is not None]

        if master_losses:
            plt.plot(master_losses, label='Master Loss', color='blue')
        if agent_losses:
            plt.plot(agent_losses, label='Agent Loss', color='red')

        plt.title('Training Loss Patterns')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

    def _plot_agent_scaling(self, results):
        """Plot agent scaling over time."""
        agent_counts = results.get('agent_counts', [])
        if not agent_counts:
            plt.text(0.5, 0.5, 'No agent scaling data', ha='center', va='center')
            return

        plt.plot(agent_counts, color='purple', linewidth=2, marker='o', markersize=3)
        plt.title('Agent Scaling Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Number of Agents')
        plt.grid(True)

    def _plot_episode_lengths(self, results):
        """Plot episode lengths over time."""
        lengths = results.get('episode_lengths', [])
        if not lengths:
            plt.text(0.5, 0.5, 'No episode length data', ha='center', va='center')
            return

        plt.plot(lengths, alpha=0.5, color='orange')
        if len(lengths) >= 25:
            ma = pd.Series(lengths).rolling(25).mean()
            plt.plot(ma, color='red', linewidth=2, label='MA 25')

        plt.title('Episode Lengths Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.grid(True)

    def _plot_training_phases(self, results):
        """Plot training phases over time."""
        phases = results.get('training_phases', [])
        if not phases:
            plt.text(0.5, 0.5, 'No training phase data', ha='center', va='center')
            return

        # Convert phases to numerical values for plotting
        phase_map = {'both': 2, 'master': 1, 'agent': 0, 'none': -1}
        numeric_phases = [phase_map.get(phase, -1) for phase in phases]

        plt.plot(numeric_phases, color='brown', linewidth=2)
        plt.title('Training Phases Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Training Phase')
        plt.yticks([-1, 0, 1, 2], ['None', 'Agent', 'Master', 'Both'])
        plt.grid(True)

    def _plot_efficiency_analysis(self, results):
        """Plot efficiency analysis."""
        rewards = results.get('episode_rewards', [])
        lengths = results.get('episode_lengths', [])

        if not rewards or not lengths:
            plt.text(0.5, 0.5, 'No efficiency data', ha='center', va='center')
            return

        efficiency = [r / l if l > 0 else 0 for r, l in zip(rewards, lengths)]
        plt.plot(efficiency, alpha=0.6, color='teal')

        if len(efficiency) >= 25:
            ma = pd.Series(efficiency).rolling(25).mean()
            plt.plot(ma, color='darkred', linewidth=2, label='MA 25')

        plt.title('Efficiency (Reward/Step)')
        plt.xlabel('Episode')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.grid(True)

    def _plot_curriculum_levels(self, results):
        """Plot curriculum levels over time."""
        levels = results.get('curriculum_levels', [])
        if not levels:
            plt.text(0.5, 0.5, 'No curriculum data', ha='center', va='center')
            return

        plt.plot(levels, color='magenta', linewidth=2, marker='s', markersize=3)
        plt.title('Curriculum Levels Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Curriculum Level')
        plt.grid(True)

    def _plot_performance_distribution(self, results):
        """Plot performance distribution."""
        rewards = results.get('episode_rewards', [])
        if not rewards:
            plt.text(0.5, 0.5, 'No reward data', ha='center', va='center')
            return

        plt.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

    def _plot_convergence_analysis(self, results):
        """Plot convergence analysis."""
        rewards = results.get('episode_rewards', [])
        if len(rewards) < 50:
            plt.text(0.5, 0.5, 'Insufficient data for convergence analysis', ha='center', va='center')
            return

        # Calculate rolling standard deviation to show convergence
        rolling_std = pd.Series(rewards).rolling(50).std()
        plt.plot(rolling_std, color='darkgreen', linewidth=2)
        plt.title('Convergence Analysis (Rolling Std)')
        plt.xlabel('Episode')
        plt.ylabel('Rolling Standard Deviation')
        plt.grid(True)

    def _plot_stability_metrics(self, analysis):
        """Plot stability metrics."""
        stability = analysis.get('training_stability', {})
        if not stability:
            plt.text(0.5, 0.5, 'No stability data', ha='center', va='center')
            return

        metrics = list(stability.keys())
        values = list(stability.values())

        plt.barh(metrics, values, color='lightcoral')
        plt.title('Training Stability Metrics')
        plt.xlabel('Stability Score')
        plt.xlim(0, 1)
        plt.grid(True)

    def _plot_summary_statistics(self, analysis):
        """Plot summary statistics."""
        basic = analysis.get('basic_metrics', {})
        if not basic or basic.get('status') == 'no_data':
            plt.text(0.5, 0.5, 'No summary data', ha='center', va='center')
            return

        # Create text summary
        summary_text = f"""
        Total Episodes: {basic.get('total_episodes', 'N/A')}
        Mean Reward: {basic.get('mean_reward', 0):.2f}
        Final 100 Mean: {basic.get('final_100_mean', 0):.2f}
        Improvement: {basic.get('improvement', 0):.2f}
        Max Reward: {basic.get('max_reward', 0):.2f}
        Min Reward: {basic.get('min_reward', 0):.2f}
        """

        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.title('Summary Statistics')
        plt.axis('off')

    def _generate_detailed_plots(self, results, experiment_config, analysis):
        """Generate detailed individual plots."""
        # Additional detailed plots can be generated here
        pass

    # Helper methods for analysis
    def _calculate_variance_trend(self, data):
        """Calculate variance trend over time."""
        if len(data) < 50:
            return 0
        windows = [data[i:i + 25] for i in range(0, len(data) - 24, 25)]
        variances = [np.var(window) for window in windows]
        if len(variances) > 1:
            return np.polyfit(range(len(variances)), variances, 1)[0]
        return 0

    def _identify_performance_phases(self, rewards):
        """Identify distinct performance phases."""
        if len(rewards) < 50:
            return []

        # Simple phase detection based on trend changes
        window_size = 25
        trends = []
        for i in range(window_size, len(rewards) - window_size):
            before = rewards[i - window_size:i]
            after = rewards[i:i + window_size]
            trend_before = np.polyfit(range(len(before)), before, 1)[0]
            trend_after = np.polyfit(range(len(after)), after, 1)[0]
            trends.append(trend_after - trend_before)

        # Identify significant trend changes
        threshold = np.std(trends) * 0.5
        phase_changes = [i for i, trend in enumerate(trends) if abs(trend) > threshold]

        return phase_changes

    def _find_convergence_point(self, data):
        """Find the episode where training converged."""
        if len(data) < 100:
            return None

        # Look for point where rolling standard deviation stabilizes
        rolling_std = pd.Series(data).rolling(50).std().tolist()

        # Find where std stops decreasing significantly
        for i in range(50, len(rolling_std) - 50):
            if all(abs(rolling_std[j] - rolling_std[i]) < 0.1 * rolling_std[i]
                   for j in range(i, min(i + 50, len(rolling_std)))):
                return i
        return None

    def _find_plateau_episodes(self, data):
        """Find episodes where performance plateaued."""
        if len(data) < 100:
            return []

        ma = pd.Series(data).rolling(25).mean().tolist()
        plateaus = []

        for i in range(25, len(ma) - 25):
            if all(abs(ma[j] - ma[i]) < 0.05 * abs(ma[i])
                   for j in range(i, min(i + 25, len(ma)))):
                plateaus.append(i)

        return plateaus

    def _calculate_improvement_rate(self, data):
        """Calculate improvement rate over time."""
        if len(data) < 10:
            return 0
        return (np.mean(data[-5:]) - np.mean(data[:5])) / len(data)

    def _find_loss_convergence(self, losses):
        """Find where loss converged."""
        if len(losses) < 50:
            return None

        # Look for sustained period of low variance
        rolling_var = pd.Series(losses).rolling(20).var().tolist()
        threshold = np.mean(rolling_var) * 0.1

        for i in range(20, len(rolling_var)):
            if rolling_var[i] < threshold:
                return i
        return None

    def _find_threshold_episode(self, data, threshold):
        """Find first episode where data exceeds threshold."""
        for i, value in enumerate(data):
            if value >= threshold:
                return i
        return None

    def _save_analysis_results(self, analysis, experiment_config):
        """Save analysis results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{experiment_config.EXPERIMENT_ID}_{timestamp}.json"
        filepath = os.path.join(self.results_directory, filename)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {key: recursive_convert(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)

        converted_analysis = recursive_convert(analysis)

        with open(filepath, 'w') as f:
            json.dump(converted_analysis, f, indent=2)

        logger.info(f"Analysis results saved to {filepath}")


def compare_experiments(experiment_results: List[Dict], experiment_names: List[str] = None):
    """
    Compare results from multiple experiments.

    Args:
        experiment_results: List of training results dictionaries
        experiment_names: Optional list of experiment names
    """
    if not experiment_names:
        experiment_names = [f"Experiment_{i + 1}" for i in range(len(experiment_results))]

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Final Performance Comparison
    final_rewards = []
    for results in experiment_results:
        rewards = results.get('episode_rewards', [])
        final_rewards.append(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))

    axes[0, 0].bar(experiment_names, final_rewards, color=sns.color_palette("viridis", len(experiment_names)))
    axes[0, 0].set_title('Final Performance Comparison')
    axes[0, 0].set_ylabel('Average Reward (Last 100 Episodes)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Learning Curves
    for i, (results, name) in enumerate(zip(experiment_results, experiment_names)):
        rewards = results.get('episode_rewards', [])
        if rewards and len(rewards) >= 25:
            ma = pd.Series(rewards).rolling(25).mean()
            axes[0, 1].plot(ma, label=name, alpha=0.8)

    axes[0, 1].set_title('Learning Curves Comparison')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Moving Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Success Rates
    final_success_rates = []
    for results in experiment_results:
        success_rates = results.get('success_rates', [])
        if success_rates:
            final_success_rates.append(success_rates[-1])
        else:
            # Calculate from rewards if success_rates not available
            rewards = results.get('episode_rewards', [])
            if rewards:
                success_count = len([r for r in rewards[-100:] if r > 0])
                final_success_rates.append(success_count / min(100, len(rewards)))
            else:
                final_success_rates.append(0)

    axes[0, 2].bar(experiment_names, final_success_rates, color=sns.color_palette("plasma", len(experiment_names)))
    axes[0, 2].set_title('Final Success Rates')
    axes[0, 2].set_ylabel('Success Rate')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Plot 4: Training Stability
    stability_scores = []
    for results in experiment_results:
        rewards = results.get('episode_rewards', [])
        if len(rewards) >= 50:
            stability = 1.0 / (1.0 + np.std(rewards[-50:]))
            stability_scores.append(stability)
        else:
            stability_scores.append(0)

    axes[1, 0].bar(experiment_names, stability_scores, color=sns.color_palette("crest", len(experiment_names)))
    axes[1, 0].set_title('Training Stability')
    axes[1, 0].set_ylabel('Stability Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 5: Episode Length Comparison
    avg_lengths = []
    for results in experiment_results:
        lengths = results.get('episode_lengths', [])
        if lengths:
            avg_lengths.append(np.mean(lengths))
        else:
            avg_lengths.append(0)

    axes[1, 1].bar(experiment_names, avg_lengths, color=sns.color_palette("flare", len(experiment_names)))
    axes[1, 1].set_title('Average Episode Length')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Plot 6: Performance Variance
    variances = []
    for results in experiment_results:
        rewards = results.get('episode_rewards', [])
        if rewards:
            variances.append(np.var(rewards))
        else:
            variances.append(0)

    axes[1, 2].bar(experiment_names, variances, color=sns.color_palette("rocket", len(experiment_names)))
    axes[1, 2].set_title('Performance Variance')
    axes[1, 2].set_ylabel('Reward Variance')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"experiment_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("Experiment comparison completed and saved")


def generate_experiment_report(analysis_results: Dict[str, Any], experiment_config) -> str:
    """
    Generate a comprehensive text report from analysis results.

    Args:
        analysis_results: Results from EnhancedResultsAnalyzer
        experiment_config: Experiment configuration

    Returns:
        Formatted report string
    """
    basic = analysis_results.get('basic_metrics', {})
    trends = analysis_results.get('performance_trends', {})
    stability = analysis_results.get('training_stability', {})

    report = f"""
🎯 ENHANCED TRAINING EXPERIMENT REPORT
═══════════════════════════════════════════════════════════════

📋 Experiment Details:
• Experiment ID: {experiment_config.EXPERIMENT_ID}
• Training Mode: {'Enhanced Serial + Progressive + Curriculum' if hasattr(experiment_config, 'SERIAL_TRAINING_MODE') and experiment_config.SERIAL_TRAINING_MODE else 'Traditional'}
• Total Cycles: {experiment_config.CYCLES}
• Episodes per Cycle: {experiment_config.EPISODES_PER_CYCLE}
• Initial Agents: {getattr(experiment_config, 'INITIAL_AGENTS', experiment_config.CARS_AMOUNT)}
• Max Agents: {getattr(experiment_config, 'MAX_AGENTS', experiment_config.CARS_AMOUNT)}
• Embedding Size: {experiment_config.EMBEDDING_SIZE}

📊 Performance Summary:
• Total Episodes: {basic.get('total_episodes', 'N/A')}
• Final Performance: {basic.get('final_100_mean', 0):.2f} (last 100 episodes)
• Overall Improvement: {basic.get('improvement', 0):.2f}
• Best Episode Reward: {basic.get('max_reward', 0):.2f}
• Performance Trend: {trends.get('trend_direction', 'Unknown')} ({trends.get('trend_slope', 0):.4f} slope)

🎯 Success Metrics:
"""

    success_analysis = analysis_results.get('success_rate_analysis', {})
    if success_analysis.get('status') != 'no_success_data':
        report += f"""• Final Success Rate: {success_analysis.get('final_success_rate', 0):.1%}
• Peak Success Rate: {success_analysis.get('peak_success_rate', 0):.1%}
• Time to 50% Success: {success_analysis.get('time_to_50_percent', 'N/A')} episodes
• Time to 80% Success: {success_analysis.get('time_to_80_percent', 'N/A')} episodes
"""

    report += f"""
🔄 Agent Scaling Analysis:
"""

    scaling_analysis = analysis_results.get('agent_scaling_analysis', {})
    if scaling_analysis.get('status') != 'no_scaling_data':
        for key, value in scaling_analysis.items():
            if key.startswith('agents_'):
                num_agents = key.split('_')[1]
                report += f"""• {num_agents} Agents: {value['mean_reward']:.2f} avg reward, {value['success_rate']:.1%} success ({value['episodes']} episodes)
"""

    report += f"""
🎓 Curriculum Learning:
"""

    curriculum_analysis = analysis_results.get('curriculum_effectiveness', {})
    if curriculum_analysis.get('status') != 'no_curriculum_data':
        for key, value in curriculum_analysis.items():
            if key.startswith('level_'):
                level = key.split('_')[1]
                report += f"""• Level {level}: {value['mean_reward']:.2f} avg reward, {value['improvement_rate']:.4f} improvement rate
"""

    report += f"""
⚡ Training Stability:
"""

    for metric, value in stability.items():
        if isinstance(value, (int, float)):
            report += f"""• {metric.replace('_', ' ').title()}: {value:.3f}
"""

    efficiency = analysis_results.get('efficiency_analysis', {})
    if efficiency.get('status') != 'no_efficiency_data':
        report += f"""
📈 Efficiency Metrics:
• Average Efficiency: {efficiency.get('mean_efficiency', 0):.4f} reward/step
• Episode Length Trend: {efficiency.get('episode_length_trend', 0):.4f}
• Efficiency Improvement: {efficiency.get('efficiency_improvement', 0):.4f}
"""

    convergence_episode = trends.get('convergence_episode')
    if convergence_episode:
        report += f"""
🎯 Convergence Analysis:
• Convergence Episode: {convergence_episode}
• Training Efficiency: {(convergence_episode / basic.get('total_episodes', 1)):.1%} of total episodes
"""

    report += f"""
💡 Key Insights & Recommendations:
"""

    # Generate recommendations based on analysis
    if basic.get('improvement', 0) > 50:
        report += "• ✅ Strong learning progress observed\n"
    elif basic.get('improvement', 0) < -20:
        report += "• ⚠️ Performance degradation detected - consider parameter adjustment\n"

    if success_analysis.get('final_success_rate', 0) > 0.8:
        report += "• ✅ Excellent final success rate achieved\n"
    elif success_analysis.get('final_success_rate', 0) < 0.5:
        report += "• ⚠️ Low success rate - consider curriculum adjustment or architecture changes\n"

    if scaling_analysis.get('scaling_efficiency'):
        avg_efficiency = np.mean(scaling_analysis['scaling_efficiency'])
        if avg_efficiency > 1.0:
            report += "• ✅ Agent scaling shows positive results\n"
        else:
            report += "• ⚠️ Agent scaling may need optimization\n"

    stability_avg = np.mean(list(stability.values())) if stability else 0
    if stability_avg > 0.8:
        report += "• ✅ Training demonstrates high stability\n"
    elif stability_avg < 0.5:
        report += "• ⚠️ Training instability detected - consider learning rate adjustment\n"

    report += f"""
═══════════════════════════════════════════════════════════════
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report