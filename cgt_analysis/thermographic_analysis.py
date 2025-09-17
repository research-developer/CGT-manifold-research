"""
Thermographic Analysis and Visualization

This module provides comprehensive thermographic analysis and visualization
capabilities for combinatorial games, with special focus on the temperature
evolution patterns predicted by the 2^n×k principle.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from .cgt_position import CGTPosition
from .temperature_analysis import TemperatureCalculator

class ThermographAnalyzer:
    """
    Comprehensive thermographic analysis and visualization toolkit.
    
    This class provides tools for creating thermographs, analyzing
    temperature evolution, and visualizing the patterns that support
    the 2^n×k structural resonance principle.
    """
    
    def __init__(self):
        """Initialize the thermograph analyzer"""
        self.temp_calculator = TemperatureCalculator()
        self.figure_cache: Dict[str, plt.Figure] = {}
        
    def create_single_thermograph(self, position: CGTPosition, 
                                 temperature_range: Tuple[float, float] = (0.0, 5.0),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a thermograph for a single position.
        
        Args:
            position: CGTPosition to analyze
            temperature_range: (min_temp, max_temp) for the graph
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Generate thermograph data
        thermograph_data = self.temp_calculator.generate_thermograph(
            position, temperature_range, num_points=200
        )
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        temperatures = thermograph_data['temperatures']
        left_values = thermograph_data['left_values']
        right_values = thermograph_data['right_values']
        
        # Plot left and right trajectories
        ax.plot(temperatures, left_values, 'b-', linewidth=2, label='Left Value', alpha=0.8)
        ax.plot(temperatures, right_values, 'r-', linewidth=2, label='Right Value', alpha=0.8)
        
        # Fill the area between curves
        ax.fill_between(temperatures, left_values, right_values, alpha=0.3, color='gray', label='Game Range')
        
        # Mark critical temperatures
        for critical_temp in thermograph_data['critical_temperatures']:
            ax.axvline(x=critical_temp, color='orange', linestyle='--', alpha=0.7)
        
        # Mark the actual temperature
        actual_temp = thermograph_data['actual_temperature']
        ax.axvline(x=actual_temp, color='green', linewidth=3, label=f'Temperature = {actual_temp:.3f}')
        
        # Mark mean value
        mean_val = thermograph_data['mean_value']
        ax.axhline(y=mean_val, color='purple', linestyle=':', alpha=0.7, label=f'Mean Value = {mean_val:.3f}')
        
        # Formatting
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Game Value', fontsize=12)
        ax.set_title(f'Thermograph: {position.position_name}\\nType: {thermograph_data["thermograph_type"]}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate(f'Actual Temperature: {actual_temp:.3f}', 
                   xy=(actual_temp, mean_val), xytext=(actual_temp + 0.5, mean_val + 0.2),
                   arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_temperature_evolution_plot(self, positions: List[CGTPosition], 
                                        deck_sizes: List[int], 
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a plot showing temperature evolution across deck sizes.
        
        This is crucial for demonstrating the 2^n×k principle.
        
        Args:
            positions: List of positions (one per deck size)
            deck_sizes: Corresponding deck sizes (e.g., [44, 48, 52])
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if len(positions) != len(deck_sizes):
            raise ValueError("Number of positions must match number of deck sizes")
        
        # Compute temperatures and mean values
        temperatures = [self.temp_calculator.compute_temperature(pos) for pos in positions]
        mean_values = [self.temp_calculator.compute_mean_value(pos) for pos in positions]
        
        # Create subplot with two y-axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Temperature evolution plot
        ax1.plot(deck_sizes, temperatures, 'o-', linewidth=3, markersize=8, color='red', label='Temperature')
        ax1.fill_between(deck_sizes, temperatures, alpha=0.3, color='red')
        
        # Highlight the 48-card peak
        if 48 in deck_sizes:
            idx_48 = deck_sizes.index(48)
            ax1.plot(48, temperatures[idx_48], 'o', markersize=12, color='gold', 
                    markeredgecolor='red', markeredgewidth=2, label='48-Card Optimum')
        
        ax1.set_xlabel('Deck Size', fontsize=12)
        ax1.set_ylabel('Temperature', fontsize=12)
        ax1.set_title('Temperature Evolution Across Deck Sizes\\n(2^n×k Structural Resonance)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for key insights
        max_temp_idx = temperatures.index(max(temperatures))
        ax1.annotate(f'Peak Temperature\\n{deck_sizes[max_temp_idx]} cards: {temperatures[max_temp_idx]:.3f}',
                    xy=(deck_sizes[max_temp_idx], temperatures[max_temp_idx]),
                    xytext=(deck_sizes[max_temp_idx] + 2, temperatures[max_temp_idx] + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
        
        # Mean value preservation plot
        ax2.plot(deck_sizes, mean_values, 's-', linewidth=3, markersize=8, color='blue', label='Mean Value')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Perfect Balance')
        ax2.fill_between(deck_sizes, mean_values, alpha=0.3, color='blue')
        
        ax2.set_xlabel('Deck Size', fontsize=12)
        ax2.set_ylabel('Mean Value', fontsize=12)
        ax2.set_title('Mean Value Preservation\\n(Demonstrates Balance Maintenance)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistical annotations
        mean_deviation = np.mean([abs(mv) for mv in mean_values])
        ax2.text(0.02, 0.98, f'Avg. Deviation from Zero: {mean_deviation:.4f}', 
                transform=ax2.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparative_thermographs(self, positions: List[CGTPosition], 
                                      labels: List[str],
                                      temperature_range: Tuple[float, float] = (0.0, 5.0),
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparative thermographs for multiple positions.
        
        Args:
            positions: List of positions to compare
            labels: Labels for each position
            temperature_range: Temperature range for analysis
            save_path: Optional save path
            
        Returns:
            matplotlib Figure object
        """
        if len(positions) != len(labels):
            raise ValueError("Number of positions must match number of labels")
        
        # Generate thermograph data for all positions
        thermograph_data = []
        for pos in positions:
            data = self.temp_calculator.generate_thermograph(pos, temperature_range, num_points=200)
            thermograph_data.append(data)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(positions)))
        
        for i, (data, label, color) in enumerate(zip(thermograph_data, labels, colors)):
            temperatures = data['temperatures']
            left_values = data['left_values']
            right_values = data['right_values']
            
            # Plot thermograph boundaries
            ax.plot(temperatures, left_values, '-', color=color, linewidth=2, alpha=0.8, label=f'{label} (Left)')
            ax.plot(temperatures, right_values, '--', color=color, linewidth=2, alpha=0.8, label=f'{label} (Right)')
            
            # Fill between curves
            ax.fill_between(temperatures, left_values, right_values, alpha=0.2, color=color)
            
            # Mark actual temperature
            actual_temp = data['actual_temperature']
            ax.axvline(x=actual_temp, color=color, linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Game Value', fontsize=12)
        ax.set_title('Comparative Thermographs\\nShowing Position Differences', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_periodicity_heatmap(self, position_sequences: Dict[str, List[CGTPosition]],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing temperature periodicity patterns.
        
        This visualization is key to demonstrating the 16-card periodicity
        predicted by the 2^n×k principle.
        
        Args:
            position_sequences: Dict mapping sequence names to position lists
            save_path: Optional save path
            
        Returns:
            matplotlib Figure object
        """
        # Compute temperature sequences
        temp_sequences = {}
        for seq_name, positions in position_sequences.items():
            temps = [self.temp_calculator.compute_temperature(pos) for pos in positions]
            temp_sequences[seq_name] = temps
        
        # Create matrix for heatmap
        max_length = max(len(seq) for seq in temp_sequences.values())
        sequence_names = list(temp_sequences.keys())
        
        # Pad sequences to same length
        temp_matrix = []
        for seq_name in sequence_names:
            seq = temp_sequences[seq_name]
            # Pad with NaN for missing values
            padded_seq = seq + [np.nan] * (max_length - len(seq))
            temp_matrix.append(padded_seq)
        
        temp_matrix = np.array(temp_matrix)
        
        # Create the heatmap
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Use a perceptually uniform colormap
        sns.heatmap(temp_matrix, 
                   xticklabels=range(1, max_length + 1),
                   yticklabels=sequence_names,
                   cmap='viridis',
                   cbar_kws={'label': 'Temperature'},
                   ax=ax)
        
        ax.set_xlabel('Position in Sequence', fontsize=12)
        ax.set_ylabel('Position Type', fontsize=12)
        ax.set_title('Temperature Periodicity Heatmap\\nRevealing 16-Card Cycles', fontsize=14)
        
        # Add vertical lines to show potential 16-card periods
        for period_start in range(16, max_length, 16):
            ax.axvline(x=period_start - 0.5, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_analysis_dashboard(self, analysis_data: Dict,
                                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard showing all key CGT analysis results.
        
        Args:
            analysis_data: Dictionary containing all analysis results
            save_path: Optional save path
            
        Returns:
            matplotlib Figure object with multiple subplots
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x3 grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Temperature evolution across deck sizes
        ax1 = fig.add_subplot(gs[0, 0])
        if 'temperature_evolution' in analysis_data:
            data = analysis_data['temperature_evolution']
            ax1.plot(data['deck_sizes'], data['temperatures'], 'o-', linewidth=2, markersize=6)
            ax1.set_title('Temperature vs Deck Size')
            ax1.set_xlabel('Deck Size')
            ax1.set_ylabel('Temperature')
            ax1.grid(True, alpha=0.3)
        
        # 2. Mean value preservation
        ax2 = fig.add_subplot(gs[0, 1])
        if 'mean_values' in analysis_data:
            data = analysis_data['mean_values']
            ax2.plot(data['deck_sizes'], data['mean_values'], 's-', linewidth=2, markersize=6, color='blue')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Mean Value Preservation')
            ax2.set_xlabel('Deck Size')
            ax2.set_ylabel('Mean Value')
            ax2.grid(True, alpha=0.3)
        
        # 3. Grundy number distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'grundy_distribution' in analysis_data:
            data = analysis_data['grundy_distribution']
            grundy_nums = list(data.keys())
            counts = list(data.values())
            ax3.bar(grundy_nums, counts, alpha=0.7, color='green')
            ax3.set_title('Grundy Number Distribution')
            ax3.set_xlabel('Grundy Number')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
        
        # 4. Position comparison thermograph
        ax4 = fig.add_subplot(gs[1, :])
        if 'comparative_thermographs' in analysis_data:
            data = analysis_data['comparative_thermographs']
            for pos_data in data:
                temps = pos_data['temperatures']
                left_vals = pos_data['left_values']
                right_vals = pos_data['right_values']
                label = pos_data['position_name']
                
                ax4.plot(temps, left_vals, '-', label=f'{label} (L)', alpha=0.8)
                ax4.plot(temps, right_vals, '--', label=f'{label} (R)', alpha=0.8)
            
            ax4.set_title('Comparative Thermographs')
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Game Value')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        # 5. Periodicity analysis
        ax5 = fig.add_subplot(gs[2, 0])
        if 'periodicity_analysis' in analysis_data:
            data = analysis_data['periodicity_analysis']
            periods = data.get('detected_periods', [])
            if periods:
                ax5.bar(range(len(periods)), periods, alpha=0.7, color='orange')
                ax5.set_title('Detected Periods')
                ax5.set_xlabel('Period Index')
                ax5.set_ylabel('Period Length')
                ax5.grid(True, alpha=0.3)
        
        # 6. Game value distribution
        ax6 = fig.add_subplot(gs[2, 1])
        if 'game_values' in analysis_data:
            values = analysis_data['game_values']
            ax6.hist(values, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax6.set_title('Game Value Distribution')
            ax6.set_xlabel('Game Value')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics
        ax7 = fig.add_subplot(gs[2, 2])
        if 'summary_stats' in analysis_data:
            stats = analysis_data['summary_stats']
            stat_names = list(stats.keys())
            stat_values = list(stats.values())
            
            # Create a text-based summary
            ax7.axis('off')
            summary_text = "\\n".join([f"{name}: {value:.4f}" if isinstance(value, float) 
                                     else f"{name}: {value}" for name, value in stats.items()])
            ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            ax7.set_title('Summary Statistics')
        
        # Overall title
        fig.suptitle('Comprehensive CGT Analysis Dashboard\\n2^n×k Structural Resonance in War Card Game', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_analysis_data(self, positions: List[CGTPosition], 
                           filename: str) -> Dict[str, any]:
        """
        Export comprehensive analysis data to file.
        
        Args:
            positions: List of positions to analyze
            filename: Output filename (JSON format)
            
        Returns:
            Dictionary with all analysis data
        """
        import json
        
        analysis_data = {}
        
        # Compute all analysis data
        for i, position in enumerate(positions):
            pos_name = position.position_name
            
            # Temperature analysis
            temp_data = self.temp_calculator.generate_thermograph(position)
            analysis_data[f'{pos_name}_thermograph'] = {
                'temperatures': temp_data['temperatures'],
                'left_values': temp_data['left_values'],
                'right_values': temp_data['right_values'],
                'actual_temperature': temp_data['actual_temperature'],
                'mean_value': temp_data['mean_value']
            }
            
            # Position summary
            analysis_data[f'{pos_name}_summary'] = position.get_analysis_summary()
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        return analysis_data