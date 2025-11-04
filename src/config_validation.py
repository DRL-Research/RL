"""
Enhanced Configuration Validation System
Validates experiment configurations and provides recommendations for optimal settings.
"""

import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import fields
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """
    Comprehensive validation system for experiment configurations.
    """

    def __init__(self):
        self.validation_rules = {
            'critical': [],  # Issues that will prevent training
            'warning': [],  # Issues that may cause suboptimal performance
            'info': [],  # Recommendations for improvement
        }

        self.parameter_ranges = {
            'LEARNING_RATE': (1e-5, 1e-1),
            'N_STEPS': (16, 2048),
            'BATCH_SIZE': (4, 512),
            'EPISODES_PER_CYCLE': (10, 1000),
            'CYCLES': (1, 20),
            'EMBEDDING_SIZE': (2, 64),
            'EXPLORATION_EXPLOITATION_THRESHOLD': (10, 200),
            'CARS_AMOUNT': (1, 10),
            'INITIAL_AGENTS': (1, 5),
            'MAX_AGENTS': (1, 10),
            'AGENTS_INCREMENT_FREQUENCY': (10, 200),
            'SCALING_SUCCESS_THRESHOLD': (0.3, 0.95),
            'EPISODES_PER_SCENARIO': (25, 500),
            'SCENARIO_VARIATIONS': (2, 10),
            'POSITION_VARIATION_RANGE': (5.0, 50.0),
        }

        self.optimal_combinations = {
            'learning_rate_batch_size': {
                # Learning rate recommendations based on batch size
                'small_batch': (32, 0.003),
                'medium_batch': (64, 0.001),
                'large_batch': (128, 0.0005),
            },
            'embedding_size_agents': {
                # Embedding size recommendations based on max agents
                'few_agents': (3, 4),
                'medium_agents': (5, 8),
                'many_agents': (7, 16),
            }
        }

    def validate_configuration(self, experiment_config) -> Dict[str, List[str]]:
        """
        Comprehensive validation of experiment configuration.

        Args:
            experiment_config: Experiment configuration object

        Returns:
            Dictionary with validation results categorized by severity
        """
        logger.info("Starting comprehensive configuration validation...")

        self.validation_rules = {'critical': [], 'warning': [], 'info': []}

        # Basic parameter validation
        self._validate_basic_parameters(experiment_config)

        # Enhanced feature validation
        self._validate_enhanced_features(experiment_config)

        # Parameter combination validation
        self._validate_parameter_combinations(experiment_config)

        # Performance optimization recommendations
        self._generate_optimization_recommendations(experiment_config)

        # Training stability checks
        self._validate_training_stability(experiment_config)

        # Resource utilization checks
        self._validate_resource_utilization(experiment_config)

        # Log validation summary
        self._log_validation_summary()

        return self.validation_rules

    def _validate_basic_parameters(self, config):
        """Validate basic parameter ranges and types."""
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if hasattr(config, param_name):
                value = getattr(config, param_name)

                if not isinstance(value, (int, float)):
                    self.validation_rules['critical'].append(
                        f"{param_name} must be a number, got {type(value)}"
                    )
                    continue

                if value < min_val or value > max_val:
                    if param_name in ['LEARNING_RATE', 'N_STEPS', 'BATCH_SIZE']:
                        self.validation_rules['critical'].append(
                            f"{param_name} = {value} is outside safe range [{min_val}, {max_val}]"
                        )
                    else:
                        self.validation_rules['warning'].append(
                            f"{param_name} = {value} is outside recommended range [{min_val}, {max_val}]"
                        )

        # Check required fields
        required_fields = ['EXPERIMENT_ID', 'CYCLES', 'EPISODES_PER_CYCLE', 'LEARNING_RATE']
        for field_name in required_fields:
            if not hasattr(config, field_name) or getattr(config, field_name) is None:
                self.validation_rules['critical'].append(f"Required field {field_name} is missing")

    def _validate_enhanced_features(self, config):
        """Validate enhanced feature configurations."""
        # Progressive scaling validation
        if getattr(config, 'PROGRESSIVE_SCALING_MODE', False):
            initial_agents = getattr(config, 'INITIAL_AGENTS', 1)
            max_agents = getattr(config, 'MAX_AGENTS', 5)
            increment_freq = getattr(config, 'AGENTS_INCREMENT_FREQUENCY', 50)

            if initial_agents >= max_agents:
                self.validation_rules['warning'].append(
                    f"INITIAL_AGENTS ({initial_agents}) should be less than MAX_AGENTS ({max_agents})"
                )

            if increment_freq < 20:
                self.validation_rules['warning'].append(
                    f"AGENTS_INCREMENT_FREQUENCY ({increment_freq}) is very low, may cause unstable scaling"
                )

        # Serial training validation
        if getattr(config, 'SERIAL_TRAINING_MODE', False):
            episodes_per_scenario = getattr(config, 'EPISODES_PER_SCENARIO', 100)
            scenario_variations = getattr(config, 'SCENARIO_VARIATIONS', 5)
            total_episodes = config.CYCLES * config.EPISODES_PER_CYCLE

            min_episodes_needed = episodes_per_scenario * scenario_variations
            if total_episodes < min_episodes_needed:
                self.validation_rules['warning'].append(
                    f"Total episodes ({total_episodes}) may be insufficient for {scenario_variations} "
                    f"scenarios with {episodes_per_scenario} episodes each"
                )

        # Parameter search validation
        if getattr(config, 'PARAMETER_SEARCH_MODE', False):
            search_iterations = getattr(config, 'SEARCH_ITERATIONS', 10)
            if search_iterations > 20:
                self.validation_rules['warning'].append(
                    f"SEARCH_ITERATIONS ({search_iterations}) is high, may take very long time"
                )

            if not hasattr(config, 'PARAM_SEARCH_RANGES'):
                self.validation_rules['critical'].append(
                    "PARAM_SEARCH_RANGES is required when PARAMETER_SEARCH_MODE is enabled"
                )

    def _validate_parameter_combinations(self, config):
        """Validate parameter combinations for optimal performance."""
        # Learning rate vs batch size
        lr = getattr(config, 'LEARNING_RATE', 0.003)
        batch_size = getattr(config, 'BATCH_SIZE', 64)

        if lr > 0.01 and batch_size < 32:
            self.validation_rules['warning'].append(
                f"High learning rate ({lr}) with small batch size ({batch_size}) may cause instability"
            )

        if lr < 0.0001 and batch_size > 128:
            self.validation_rules['warning'].append(
                f"Very low learning rate ({lr}) with large batch size ({batch_size}) may slow convergence"
            )

        # Embedding size vs max agents
        embedding_size = getattr(config, 'EMBEDDING_SIZE', 4)
        max_agents = getattr(config, 'MAX_AGENTS', getattr(config, 'CARS_AMOUNT', 5))

        if embedding_size < max_agents:
            self.validation_rules['info'].append(
                f"EMBEDDING_SIZE ({embedding_size}) is less than MAX_AGENTS ({max_agents}). "
                f"Consider increasing to {max_agents * 2} for better representation capacity"
            )

        # N_STEPS vs EPISODES_PER_CYCLE
        n_steps = getattr(config, 'N_STEPS', 64)
        episodes_per_cycle = getattr(config, 'EPISODES_PER_CYCLE', 200)

        if n_steps > episodes_per_cycle * 50:  # Assuming ~50 steps per episode average
            self.validation_rules['warning'].append(
                f"N_STEPS ({n_steps}) is very large compared to expected episode lengths"
            )

    def _generate_optimization_recommendations(self, config):
        """Generate performance optimization recommendations."""
        # Learning rate recommendations
        batch_size = getattr(config, 'BATCH_SIZE', 64)
        current_lr = getattr(config, 'LEARNING_RATE', 0.003)

        if batch_size <= 32:
            recommended_lr = 0.003
        elif batch_size <= 64:
            recommended_lr = 0.001
        else:
            recommended_lr = 0.0005

        if abs(current_lr - recommended_lr) > recommended_lr * 0.5:
            self.validation_rules['info'].append(
                f"For batch size {batch_size}, consider learning rate around {recommended_lr:.4f} "
                f"(current: {current_lr:.4f})"
            )

        # Architecture recommendations
        max_agents = getattr(config, 'MAX_AGENTS', getattr(config, 'CARS_AMOUNT', 5))
        embedding_size = getattr(config, 'EMBEDDING_SIZE', 4)

        if max_agents <= 3 and embedding_size > 8:
            self.validation_rules['info'].append(
                f"For {max_agents} agents, embedding size {embedding_size} may be oversized. "
                f"Consider reducing to 4-6 for efficiency"
            )
        elif max_agents >= 7 and embedding_size < 8:
            self.validation_rules['info'].append(
                f"For {max_agents} agents, embedding size {embedding_size} may be undersized. "
                f"Consider increasing to 12-16 for better capacity"
            )

        # Training schedule recommendations
        cycles = getattr(config, 'CYCLES', 3)
        if cycles < 5 and getattr(config, 'PROGRESSIVE_SCALING_MODE', False):
            self.validation_rules['info'].append(
                f"With progressive scaling, consider increasing CYCLES to at least 5 "
                f"for better scaling progression (current: {cycles})"
            )

    def _validate_training_stability(self, config):
        """Validate configuration for training stability."""
        # Check for potentially unstable combinations
        exploration_threshold = getattr(config, 'EXPLORATION_EXPLOITATION_THRESHOLD', 50)
        if exploration_threshold < 20:
            self.validation_rules['warning'].append(
                f"Very low exploration threshold ({exploration_threshold}) may cause premature convergence"
            )
        elif exploration_threshold > 150:
            self.validation_rules['warning'].append(
                f"Very high exploration threshold ({exploration_threshold}) may prevent convergence"
            )

        # Check curriculum learning settings
        if getattr(config, 'CURRICULUM_LEARNING', False):
            difficulty_scaling = getattr(config, 'DIFFICULTY_SCALING_FACTOR', 1.2)
            if difficulty_scaling > 2.0:
                self.validation_rules['warning'].append(
                    f"High difficulty scaling factor ({difficulty_scaling}) may cause training instability"
                )
            elif difficulty_scaling < 1.1:
                self.validation_rules['info'].append(
                    f"Low difficulty scaling factor ({difficulty_scaling}) may not provide enough challenge"
                )

    def _validate_resource_utilization(self, config):
        """Validate configuration for resource efficiency."""
        # Estimate memory usage
        max_agents = getattr(config, 'MAX_AGENTS', getattr(config, 'CARS_AMOUNT', 5))
        embedding_size = getattr(config, 'EMBEDDING_SIZE', 4)
        n_steps = getattr(config, 'N_STEPS', 64)

        # Rough memory estimation (MB)
        estimated_memory = (max_agents * embedding_size * n_steps * 4) / (1024 * 1024)  # 4 bytes per float

        if estimated_memory > 100:  # > 100MB
            self.validation_rules['info'].append(
                f"Estimated memory usage: ~{estimated_memory:.1f}MB. "
                f"Consider reducing N_STEPS or EMBEDDING_SIZE if memory is limited"
            )

        # Estimate training time
        total_episodes = config.CYCLES * config.EPISODES_PER_CYCLE
        if getattr(config, 'PARAMETER_SEARCH_MODE', False):
            search_iterations = getattr(config, 'SEARCH_ITERATIONS', 10)
            total_episodes *= search_iterations

        if total_episodes > 5000:
            self.validation_rules['info'].append(
                f"Training will run {total_episodes} episodes. "
                f"Consider reducing for faster experimentation"
            )

    def _log_validation_summary(self):
        """Log validation summary."""
        critical_count = len(self.validation_rules['critical'])
        warning_count = len(self.validation_rules['warning'])
        info_count = len(self.validation_rules['info'])

        if critical_count > 0:
            logger.error(f"❌ {critical_count} critical validation issues found")
            for issue in self.validation_rules['critical']:
                logger.error(f"  • {issue}")

        if warning_count > 0:
            logger.warning(f"⚠️  {warning_count} warnings found")
            for issue in self.validation_rules['warning']:
                logger.warning(f"  • {issue}")

        if info_count > 0:
            logger.info(f"💡 {info_count} optimization recommendations")
            for issue in self.validation_rules['info']:
                logger.info(f"  • {issue}")

        if critical_count == 0 and warning_count == 0:
            logger.info("✅ Configuration validation passed!")

    def generate_configuration_report(self, config) -> str:
        """Generate a comprehensive configuration report."""
        validation_results = self.validate_configuration(config)

        report = f"""
🔧 EXPERIMENT CONFIGURATION VALIDATION REPORT
═══════════════════════════════════════════════════════════════

📋 Configuration Summary:
• Experiment ID: {getattr(config, 'EXPERIMENT_ID', 'N/A')}
• Training Mode: {'Enhanced' if getattr(config, 'SERIAL_TRAINING_MODE', False) else 'Traditional'}
• Total Episodes: {getattr(config, 'CYCLES', 0) * getattr(config, 'EPISODES_PER_CYCLE', 0)}
• Learning Rate: {getattr(config, 'LEARNING_RATE', 'N/A')}
• Batch Size: {getattr(config, 'BATCH_SIZE', 'N/A')}
• Embedding Size: {getattr(config, 'EMBEDDING_SIZE', 'N/A')}

🎯 Enhanced Features:
• Serial Training: {'✅ Enabled' if getattr(config, 'SERIAL_TRAINING_MODE', False) else '❌ Disabled'}
• Progressive Scaling: {'✅ Enabled' if getattr(config, 'PROGRESSIVE_SCALING_MODE', False) else '❌ Disabled'}
• Curriculum Learning: {'✅ Enabled' if getattr(config, 'CURRICULUM_LEARNING', False) else '❌ Disabled'}
• Parameter Search: {'✅ Enabled' if getattr(config, 'PARAMETER_SEARCH_MODE', False) else '❌ Disabled'}

"""

        if getattr(config, 'PROGRESSIVE_SCALING_MODE', False):
            report += f"""
📈 Progressive Scaling Settings:
• Initial Agents: {getattr(config, 'INITIAL_AGENTS', 'N/A')}
• Max Agents: {getattr(config, 'MAX_AGENTS', 'N/A')}
• Scaling Frequency: Every {getattr(config, 'AGENTS_INCREMENT_FREQUENCY', 'N/A')} episodes
• Success Threshold: {getattr(config, 'SCALING_SUCCESS_THRESHOLD', 'N/A')}

"""

        if getattr(config, 'SERIAL_TRAINING_MODE', False):
            report += f"""
🔄 Serial Training Settings:
• Episodes per Scenario: {getattr(config, 'EPISODES_PER_SCENARIO', 'N/A')}
• Scenario Variations: {getattr(config, 'SCENARIO_VARIATIONS', 'N/A')}
• Position Variation Range: ±{getattr(config, 'POSITION_VARIATION_RANGE', 'N/A')}m

"""

        # Validation results
        critical_issues = validation_results.get('critical', [])
        warnings = validation_results.get('warning', [])
        recommendations = validation_results.get('info', [])

        if critical_issues:
            report += "❌ CRITICAL ISSUES:\n"
            for issue in critical_issues:
                report += f"  • {issue}\n"
            report += "\n"

        if warnings:
            report += "⚠️  WARNINGS:\n"
            for warning in warnings:
                report += f"  • {warning}\n"
            report += "\n"

        if recommendations:
            report += "💡 OPTIMIZATION RECOMMENDATIONS:\n"
            for rec in recommendations:
                report += f"  • {rec}\n"
            report += "\n"

        # Overall assessment
        if critical_issues:
            status = "❌ CONFIGURATION ISSUES DETECTED"
            recommendation = "Please fix critical issues before training"
        elif warnings:
            status = "⚠️  CONFIGURATION NEEDS ATTENTION"
            recommendation = "Consider addressing warnings for optimal performance"
        else:
            status = "✅ CONFIGURATION VALIDATED"
            recommendation = "Configuration looks good for training"

        report += f"""
🎯 OVERALL ASSESSMENT: {status}
📝 RECOMMENDATION: {recommendation}

═══════════════════════════════════════════════════════════════
"""

        return report

    def suggest_optimal_configuration(self, base_config, target_performance: str = "balanced") -> Dict[str, Any]:
        """
        Suggest optimal configuration based on target performance.

        Args:
            base_config: Base configuration to modify
            target_performance: "fast", "balanced", or "thorough"

        Returns:
            Dictionary with suggested parameter values
        """
        suggestions = {}

        if target_performance == "fast":
            # Fast training for experimentation
            suggestions.update({
                'CYCLES': 3,
                'EPISODES_PER_CYCLE': 100,
                'LEARNING_RATE': 0.005,
                'N_STEPS': 64,
                'BATCH_SIZE': 32,
                'EPISODES_PER_SCENARIO': 50,
                'SCENARIO_VARIATIONS': 3,
                'EMBEDDING_SIZE': 4,
                'INITIAL_AGENTS': 2,
                'MAX_AGENTS': 3,
            })

        elif target_performance == "thorough":
            # Thorough training for best results
            suggestions.update({
                'CYCLES': 8,
                'EPISODES_PER_CYCLE': 300,
                'LEARNING_RATE': 0.001,
                'N_STEPS': 256,
                'BATCH_SIZE': 128,
                'EPISODES_PER_SCENARIO': 200,
                'SCENARIO_VARIATIONS': 7,
                'EMBEDDING_SIZE': 16,
                'INITIAL_AGENTS': 2,
                'MAX_AGENTS': 5,
            })

        else:  # balanced
            # Balanced approach
            suggestions.update({
                'CYCLES': 5,
                'EPISODES_PER_CYCLE': 200,
                'LEARNING_RATE': 0.003,
                'N_STEPS': 128,
                'BATCH_SIZE': 64,
                'EPISODES_PER_SCENARIO': 150,
                'SCENARIO_VARIATIONS': 5,
                'EMBEDDING_SIZE': 8,
                'INITIAL_AGENTS': 2,
                'MAX_AGENTS': 5,
            })

        # Add enhanced features for all configurations
        suggestions.update({
            'SERIAL_TRAINING_MODE': True,
            'PROGRESSIVE_SCALING_MODE': True,
            'CURRICULUM_LEARNING': True,
            'PROGRESSIVE_REWARD_SCALING': True,
            'AGENTS_INCREMENT_FREQUENCY': 50,
            'SCALING_SUCCESS_THRESHOLD': 0.8,
        })

        return suggestions


def validate_and_optimize_config(experiment_config, target_performance: str = "balanced",
                                 auto_fix: bool = False):
    """
    Convenience function to validate and optionally optimize configuration.

    Args:
        experiment_config: Experiment configuration object
        target_performance: Target performance profile
        auto_fix: Whether to automatically apply suggested fixes

    Returns:
        Tuple of (validation_results, suggested_config)
    """
    validator = ConfigurationValidator()

    # Validate current configuration
    validation_results = validator.validate_configuration(experiment_config)

    # Generate optimization suggestions
    suggestions = validator.suggest_optimal_configuration(experiment_config, target_performance)

    # Auto-fix critical issues if requested
    if auto_fix and validation_results['critical']:
        logger.info("Auto-fixing critical configuration issues...")

        for param, value in suggestions.items():
            if hasattr(experiment_config, param):
                old_value = getattr(experiment_config, param)
                setattr(experiment_config, param, value)
                logger.info(f"Auto-fixed {param}: {old_value} -> {value}")

        # Re-validate after fixes
        validation_results = validator.validate_configuration(experiment_config)

    # Generate report
    report = validator.generate_configuration_report(experiment_config)
    logger.info(f"\n{report}")

    return validation_results, suggestions


# Export main functions
__all__ = [
    'ConfigurationValidator',
    'validate_and_optimize_config'
]