"""
Create visualizations for the CGT analysis results
"""

import sys
import os
sys.path.append('/workspace')

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_analysis_results():
    """Load the analysis results"""
    with open('/workspace/data/simplified_analysis_results.json', 'r') as f:
        return json.load(f)

def create_temperature_evolution_plot():
    """Create temperature evolution plot across deck sizes"""
    results = load_analysis_results()
    
    # Extract temperature data
    deck_sizes = [44, 48, 52]
    positions = ['A', 'B', 'C', 'D', 'E']
    
    # Create data for plotting
    plot_data = []
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            for pos in positions:
                if pos in results[str(deck_size)]:
                    temp = results[str(deck_size)][pos]['temperature']
                    plot_data.append({
                        'Deck_Size': deck_size,
                        'Position': pos,
                        'Temperature': temp
                    })
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual position temperatures
    for pos in positions:
        pos_data = df[df['Position'] == pos]
        ax1.plot(pos_data['Deck_Size'], pos_data['Temperature'], 
                'o-', linewidth=2, markersize=8, label=f'Position {pos}')
    
    ax1.set_xlabel('Deck Size')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature by Position and Deck Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average temperatures
    avg_temps = df.groupby('Deck_Size')['Temperature'].mean()
    ax2.plot(avg_temps.index, avg_temps.values, 'ro-', linewidth=3, markersize=10)
    ax2.fill_between(avg_temps.index, avg_temps.values, alpha=0.3, color='red')
    
    # Highlight the 48-card point
    ax2.plot(48, avg_temps[48], 'go', markersize=15, markeredgecolor='darkgreen', 
            markeredgewidth=3, label='48-Card Deck')
    
    ax2.set_xlabel('Deck Size')
    ax2.set_ylabel('Average Temperature')
    ax2.set_title('Average Temperature Across All Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation for 2^n×k principle
    max_temp_deck = avg_temps.idxmax()
    if max_temp_deck == 48:
        ax2.text(48, avg_temps[48] + 0.05, '2^n×k Peak\\n(Confirmed)', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax2.text(max_temp_deck, avg_temps[max_temp_deck] + 0.05, 
                f'Actual Peak\\n({max_temp_deck} cards)', 
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/workspace/data/temperature_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_temps

def create_grundy_analysis_plot():
    """Create Grundy number analysis plot"""
    results = load_analysis_results()
    
    deck_sizes = [44, 48, 52]
    positions = ['A', 'B', 'C', 'D', 'E']
    
    # Create Grundy number matrix
    grundy_matrix = []
    for deck_size in deck_sizes:
        row = []
        for pos in positions:
            if str(deck_size) in results and pos in results[str(deck_size)]:
                row.append(results[str(deck_size)][pos]['grundy_number'])
            else:
                row.append(0)
        grundy_matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    sns.heatmap(grundy_matrix, 
               xticklabels=positions,
               yticklabels=[f'{ds} cards' for ds in deck_sizes],
               annot=True, 
               fmt='d',
               cmap='viridis',
               cbar_kws={'label': 'Grundy Number'},
               ax=ax)
    
    ax.set_title('Grundy Numbers Across Positions and Deck Sizes')
    ax.set_xlabel('Position Type')
    ax.set_ylabel('Deck Size')
    
    plt.tight_layout()
    plt.savefig('/workspace/data/grundy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_game_value_analysis():
    """Create game value analysis plot"""
    results = load_analysis_results()
    
    deck_sizes = [44, 48, 52]
    positions = ['A', 'B', 'C', 'D', 'E']
    
    # Extract game values
    plot_data = []
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            for pos in positions:
                if pos in results[str(deck_size)]:
                    game_val = results[str(deck_size)][pos]['game_value']
                    plot_data.append({
                        'Deck_Size': deck_size,
                        'Position': pos,
                        'Game_Value': game_val
                    })
    
    df = pd.DataFrame(plot_data)
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Group by position
    positions_to_plot = df[df['Game_Value'] != 0]['Position'].unique()  # Only non-zero values
    
    if len(positions_to_plot) > 0:
        x = np.arange(len(deck_sizes))
        width = 0.15
        
        for i, pos in enumerate(positions_to_plot):
            pos_data = df[df['Position'] == pos]
            values = [pos_data[pos_data['Deck_Size'] == ds]['Game_Value'].iloc[0] 
                     if len(pos_data[pos_data['Deck_Size'] == ds]) > 0 else 0 
                     for ds in deck_sizes]
            ax.bar(x + i*width, values, width, label=f'Position {pos}')
        
        ax.set_xlabel('Deck Size')
        ax.set_ylabel('Game Value')
        ax.set_title('Game Values by Position and Deck Size')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{ds} cards' for ds in deck_sizes])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/workspace/data/game_values.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard():
    """Create a comprehensive analysis dashboard"""
    results = load_analysis_results()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    deck_sizes = [44, 48, 52]
    positions = ['A', 'B', 'C', 'D', 'E']
    
    # 1. Temperature evolution
    ax1 = fig.add_subplot(gs[0, 0])
    temp_data = []
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            temps = [results[str(deck_size)][pos]['temperature'] 
                    for pos in positions if pos in results[str(deck_size)]]
            temp_data.append(np.mean(temps) if temps else 0)
    
    ax1.plot(deck_sizes, temp_data, 'ro-', linewidth=3, markersize=10)
    ax1.fill_between(deck_sizes, temp_data, alpha=0.3, color='red')
    ax1.set_title('Average Temperature Evolution')
    ax1.set_xlabel('Deck Size')
    ax1.set_ylabel('Temperature')
    ax1.grid(True, alpha=0.3)
    
    # Highlight 48-card deck
    max_idx = np.argmax(temp_data)
    if deck_sizes[max_idx] == 48:
        ax1.plot(48, temp_data[1], 'go', markersize=15, markeredgecolor='darkgreen', markeredgewidth=3)
        ax1.text(48, temp_data[1] + 0.02, '48-Card\\nOptimum', ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 2. Grundy number distribution
    ax2 = fig.add_subplot(gs[0, 1])
    grundy_counts = {}
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            grundy_nums = [results[str(deck_size)][pos]['grundy_number'] 
                          for pos in positions if pos in results[str(deck_size)]]
            for g in grundy_nums:
                grundy_counts[g] = grundy_counts.get(g, 0) + 1
    
    if grundy_counts:
        ax2.bar(grundy_counts.keys(), grundy_counts.values(), alpha=0.7, color='green')
        ax2.set_title('Grundy Number Distribution')
        ax2.set_xlabel('Grundy Number')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Position A temperature detail (shows the clearest pattern)
    ax3 = fig.add_subplot(gs[1, 0])
    pos_a_temps = []
    pos_a_games = []
    for deck_size in deck_sizes:
        if str(deck_size) in results and 'A' in results[str(deck_size)]:
            pos_a_temps.append(results[str(deck_size)]['A']['temperature'])
            pos_a_games.append(results[str(deck_size)]['A']['game_value'])
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(deck_sizes, pos_a_temps, 'bo-', linewidth=2, label='Temperature')
    line2 = ax3_twin.plot(deck_sizes, pos_a_games, 'rs-', linewidth=2, label='Game Value')
    
    ax3.set_title('Position A: Temperature vs Game Value')
    ax3.set_xlabel('Deck Size')
    ax3.set_ylabel('Temperature', color='blue')
    ax3_twin.set_ylabel('Game Value', color='red')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. Win rate analysis
    ax4 = fig.add_subplot(gs[1, 1])
    win_rate_data = []
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            p1_rates = []
            for pos in positions:
                if pos in results[str(deck_size)]:
                    p1_rates.append(results[str(deck_size)][pos]['simulation_stats']['p1_win_rate'])
            win_rate_data.append(np.mean(p1_rates) if p1_rates else 0.5)
    
    ax4.bar(range(len(deck_sizes)), win_rate_data, alpha=0.7, color='purple')
    ax4.axhline(y=0.5, color='red', linestyle='--', label='Perfect Balance')
    ax4.set_title('Average P1 Win Rate by Deck Size')
    ax4.set_xlabel('Deck Size')
    ax4.set_ylabel('Player 1 Win Rate')
    ax4.set_xticks(range(len(deck_sizes)))
    ax4.set_xticklabels([f'{ds} cards' for ds in deck_sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Game length analysis
    ax5 = fig.add_subplot(gs[2, 0])
    game_lengths = []
    for deck_size in deck_sizes:
        if str(deck_size) in results:
            lengths = []
            for pos in positions:
                if pos in results[str(deck_size)]:
                    lengths.append(results[str(deck_size)][pos]['simulation_stats']['average_game_length'])
            game_lengths.append(np.mean(lengths) if lengths else 50)
    
    ax5.plot(deck_sizes, game_lengths, 'go-', linewidth=2, markersize=8)
    ax5.fill_between(deck_sizes, game_lengths, alpha=0.3, color='green')
    ax5.set_title('Average Game Length by Deck Size')
    ax5.set_xlabel('Deck Size')
    ax5.set_ylabel('Average Moves')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate key statistics
    max_temp_deck = deck_sizes[np.argmax(temp_data)]
    principle_confirmed = max_temp_deck == 48
    
    summary_text = f"""
    ANALYSIS SUMMARY
    
    Total Positions Analyzed: {len(positions) * len(deck_sizes)}
    
    Temperature Analysis:
    • 44 cards: {temp_data[0]:.3f}
    • 48 cards: {temp_data[1]:.3f} {'🏆' if principle_confirmed else ''}
    • 52 cards: {temp_data[2]:.3f}
    
    2^n×k Principle: {'✅ CONFIRMED' if principle_confirmed else '❌ NOT CONFIRMED'}
    Peak at: {max_temp_deck} cards
    
    Grundy Numbers: {len(grundy_counts)} unique values
    Most common: {max(grundy_counts.keys(), key=grundy_counts.get) if grundy_counts else 'N/A'}
    
    Game Balance:
    Avg P1 Win Rate: {np.mean(win_rate_data):.1%}
    Deviation from 50%: {abs(np.mean(win_rate_data) - 0.5):.1%}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Overall title
    fig.suptitle('Comprehensive CGT Analysis Dashboard\\nWar Card Game: 2^n×k Structural Resonance Study', 
                fontsize=16, fontweight='bold')
    
    plt.savefig('/workspace/data/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all visualizations"""
    print("Creating visualizations...")
    
    # Create individual plots
    avg_temps = create_temperature_evolution_plot()
    print("✓ Temperature evolution plot created")
    
    create_grundy_analysis_plot()
    print("✓ Grundy number heatmap created")
    
    create_game_value_analysis()
    print("✓ Game value analysis created")
    
    create_comprehensive_dashboard()
    print("✓ Comprehensive dashboard created")
    
    # Print summary
    print("\\n" + "="*50)
    print("VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Average temperatures: {dict(avg_temps)}")
    print(f"Peak temperature at: {avg_temps.idxmax()} cards")
    print(f"2^n×k principle: {'✅ CONFIRMED' if avg_temps.idxmax() == 48 else '❌ NOT CONFIRMED'}")
    print("\\nAll visualizations saved to /workspace/data/")
    print("="*50)

if __name__ == "__main__":
    main()